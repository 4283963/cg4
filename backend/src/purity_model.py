import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from datetime import timedelta
import warnings
import traceback
warnings.filterwarnings('ignore')

from industrial_db import (
    query_machine_data,
    MACHINE_IDS,
    PARAM_RANGES,
    get_latest_quality_report,
    DATA_QUALITY_THRESHOLDS
)

FEATURE_COLS = ['vibration_freq', 'inclination_angle', 'fan_airflow', 'amplitude']
TARGET_COL = 'purity'

FALLBACK_PURITY_BASE = 99.0
MIN_TRAIN_SAMPLES = 10
RIDGE_ALPHA_DEFAULT = 1.0
RIDGE_ALPHA_STRONG = 10.0


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _clip_parameters(params):
    clipped = {}
    for col in FEATURE_COLS:
        val = _safe_float(params.get(col))
        if col in PARAM_RANGES:
            lo, hi = PARAM_RANGES[col]
            val = max(lo, min(hi, val))
        clipped[col] = val
    return clipped


def _check_data_quality(X, y):
    issues = []
    n_samples = X.shape[0]

    if n_samples < MIN_TRAIN_SAMPLES:
        issues.append(f'INSUFFICIENT_SAMPLES: only {n_samples} < {MIN_TRAIN_SAMPLES}')

    for i, col in enumerate(FEATURE_COLS):
        col_std = np.std(X[:, i])
        if col_std < 1e-8:
            issues.append(f'ZERO_VARIANCE: feature {col} has near-zero std ({col_std:.2e})')

    try:
        X_centered = X - np.mean(X, axis=0)
        cov = X_centered.T @ X_centered
        eigvals = np.linalg.eigvalsh(cov)
        eigvals_pos = eigvals[eigvals > 1e-10]
        if len(eigvals_pos) == 0:
            cond_num = float('inf')
        else:
            cond_num = float(np.max(eigvals_pos) / np.min(eigvals_pos))
        if cond_num > 1e10:
            issues.append(f'SINGULAR_MATRIX_WARNING: condition number={cond_num:.2e}')
    except Exception:
        cond_num = float('nan')

    return issues


def _build_pipeline(degree=2, alpha=RIDGE_ALPHA_DEFAULT):
    return Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha, fit_intercept=True, solver='auto'))
    ])


class BaselineModel:
    def __init__(self, purity_median, amplitude_median):
        self.purity_median = purity_median
        self.amplitude_median = amplitude_median

    def predict(self, X):
        n = X.shape[0] if hasattr(X, 'shape') else 1
        amplitude = X[:, 3] if hasattr(X, 'shape') and X.ndim > 1 else np.array([self.amplitude_median])
        deviation = np.abs(amplitude - self.amplitude_median)
        purity = self.purity_median - deviation * 0.5
        purity = np.clip(purity, 90.0, 99.99)
        return purity.reshape(-1)

    def get_coef_info(self):
        return {
            'type': 'BASELINE_FALLBACK',
            'purity_median': self.purity_median,
            'amplitude_median': self.amplitude_median
        }


class PurityPredictionModel:
    def __init__(self, degree=2):
        self.degree = degree
        self.models = {}
        self.poly_features = {}
        self.model_info = {}
        self.last_good_models = {}
        self.training_errors = {}

    def _try_training(self, X, y, machine_id, degree, alpha, strategy_name):
        try:
            X_np = np.asarray(X, dtype=np.float64)
            y_np = np.asarray(y, dtype=np.float64)

            issues = _check_data_quality(X_np, y_np)

            pipeline = _build_pipeline(degree=degree, alpha=alpha)
            pipeline.fit(X_np, y_np)

            y_pred = pipeline.predict(X_np)
            ss_res = np.sum((y_np - y_pred) ** 2)
            ss_tot = np.sum((y_np - np.mean(y_np)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            r2 = float(np.clip(r2, -1.0, 1.0))

            ridge = pipeline.named_steps['ridge']
            poly = pipeline.named_steps['poly_features']

            info = {
                'strategy': strategy_name,
                'degree': degree,
                'alpha': alpha,
                'r2_score': round(r2, 4),
                'training_samples': int(len(X_np)),
                'data_issues': issues,
                'coefficients': ridge.coef_.tolist(),
                'intercept': float(ridge.intercept_),
                'feature_names': poly.get_feature_names_out(FEATURE_COLS).tolist(),
            }

            return pipeline, info, None

        except np.linalg.LinAlgError as e:
            return None, None, f'LinAlgError ({strategy_name}): {str(e)}'
        except ValueError as e:
            return None, None, f'ValueError ({strategy_name}): {str(e)}'
        except Exception as e:
            return None, None, f'{type(e).__name__} ({strategy_name}): {str(e)}'

    def train(self, machine_id, hours=24):
        try:
            data = query_machine_data(machine_id, hours=hours)
            if data is None or len(data) == 0:
                raise ValueError(f"No data available for machine {machine_id}")

            X_raw = data[FEATURE_COLS].values
            y_raw = data[TARGET_COL].values

            valid_mask = (
                np.isfinite(X_raw).all(axis=1) &
                np.isfinite(y_raw) &
                (y_raw > 80.0) & (y_raw < 100.0)
            )
            X = X_raw[valid_mask]
            y = y_raw[valid_mask]

            if len(X) < MIN_TRAIN_SAMPLES:
                raise ValueError(
                    f'Insufficient valid samples for {machine_id}: {len(X)} < {MIN_TRAIN_SAMPLES}'
                )

            quality_report = get_latest_quality_report(machine_id, hours)

            strategies = [
                (self.degree, RIDGE_ALPHA_DEFAULT, f'poly{self.degree}_ridge_default'),
                (self.degree, RIDGE_ALPHA_STRONG, f'poly{self.degree}_ridge_strong'),
                (1, RIDGE_ALPHA_DEFAULT, 'poly1_ridge_default'),
                (1, RIDGE_ALPHA_STRONG, 'poly1_ridge_strong'),
            ]

            last_error = None
            trained_pipeline = None
            train_info = None

            for deg, alpha, name in strategies:
                pipeline, info, err = self._try_training(X, y, machine_id, deg, alpha, name)
                if pipeline is not None:
                    trained_pipeline = pipeline
                    train_info = info
                    if quality_report:
                        train_info['data_quality'] = quality_report
                    break
                last_error = err

            if trained_pipeline is None:
                purity_med = float(np.median(y))
                amp_med = float(np.median(X[:, 3]))
                trained_pipeline = BaselineModel(purity_med, amp_med)
                train_info = {
                    'strategy': 'BASELINE_FALLBACK',
                    'r2_score': None,
                    'training_samples': int(len(X)),
                    'training_error': last_error,
                    'data_quality': quality_report,
                    **trained_pipeline.get_coef_info()
                }

            self.models[machine_id] = trained_pipeline
            self.model_info[machine_id] = train_info
            self.training_errors[machine_id] = last_error

            if isinstance(trained_pipeline, Pipeline):
                self.last_good_models[machine_id] = {
                    'pipeline': trained_pipeline,
                    'info': train_info
                }

            return {
                'machine_id': machine_id,
                **{k: v for k, v in train_info.items()
                   if k not in ['coefficients', 'feature_names']}
            }

        except Exception as e:
            err_msg = f'TRAIN_FAILED for {machine_id}: {type(e).__name__}: {str(e)}'
            self.training_errors[machine_id] = err_msg

            if machine_id in self.last_good_models:
                cached = self.last_good_models[machine_id]
                self.models[machine_id] = cached['pipeline']
                self.model_info[machine_id] = {
                    **cached['info'],
                    'strategy': cached['info'].get('strategy', 'UNKNOWN') + '_CACHED',
                    'stale_model': True,
                    'error': err_msg
                }
                return {
                    'machine_id': machine_id,
                    'status': 'FALLBACK_CACHED_MODEL',
                    'error': err_msg
                }

            default_model = BaselineModel(FALLBACK_PURITY_BASE, 1.8)
            self.models[machine_id] = default_model
            self.model_info[machine_id] = {
                'strategy': 'DEFAULT_FALLBACK',
                'error': err_msg,
                **default_model.get_coef_info()
            }
            return {
                'machine_id': machine_id,
                'status': 'FALLBACK_DEFAULT',
                'error': err_msg
            }

    def train_all(self, hours=24):
        results = []
        for machine_id in MACHINE_IDS:
            try:
                result = self.train(machine_id, hours)
                results.append(result)
            except Exception as e:
                results.append({
                    'machine_id': machine_id,
                    'error': f'{type(e).__name__}: {str(e)}'
                })
        return results

    def predict(self, machine_id, params):
        try:
            if machine_id not in self.models:
                self.train(machine_id)

            model = self.models[machine_id]
            clipped = _clip_parameters(params)
            X = np.array([[clipped[col] for col in FEATURE_COLS]], dtype=np.float64)

            try:
                raw_pred = model.predict(X)
                pred_value = float(np.asarray(raw_pred).reshape(-1)[0])
            except Exception as inner_e:
                purity_val = _safe_float(params.get('purity'), FALLBACK_PURITY_BASE)
                pred_value = purity_val if 90 < purity_val < 100 else FALLBACK_PURITY_BASE

            if not np.isfinite(pred_value):
                pred_value = FALLBACK_PURITY_BASE
            pred_value = float(np.clip(pred_value, 90.0, 99.99))

            result = {
                'predicted_purity': round(pred_value, 4),
                'input_parameters': clipped,
            }
            if machine_id in self.model_info:
                info = self.model_info[machine_id]
                result['model_strategy'] = info.get('strategy', 'UNKNOWN')
                if 'error' in info:
                    result['model_warning'] = info['error']

            return result

        except Exception as e:
            purity_val = _safe_float(params.get('purity'), FALLBACK_PURITY_BASE)
            fallback = purity_val if 90 < purity_val < 100 else FALLBACK_PURITY_BASE
            return {
                'predicted_purity': round(float(fallback), 4),
                'input_parameters': _clip_parameters(params),
                'model_strategy': 'RUNTIME_FALLBACK',
                'model_warning': f'{type(e).__name__}: {str(e)}'
            }

    def predict_purity_curve(self, machine_id, current_params, amplitude_range=None, num_points=50):
        try:
            if machine_id not in self.models:
                self.train(machine_id)

            if amplitude_range is None:
                amplitude_range = PARAM_RANGES['amplitude']

            amplitudes = np.linspace(amplitude_range[0], amplitude_range[1], num_points)
            purity_predictions = []

            clipped_base = _clip_parameters(current_params)
            model = self.models[machine_id]

            for amp in amplitudes:
                test_params = clipped_base.copy()
                test_params['amplitude'] = float(amp)
                X = np.array([[test_params[col] for col in FEATURE_COLS]], dtype=np.float64)

                try:
                    raw = model.predict(X)
                    val = float(np.asarray(raw).reshape(-1)[0])
                except Exception:
                    val = FALLBACK_PURITY_BASE

                if not np.isfinite(val):
                    val = FALLBACK_PURITY_BASE
                purity_predictions.append(round(float(np.clip(val, 90.0, 99.99)), 4))

            current_pred = self.predict(machine_id, current_params)

            return {
                'machine_id': machine_id,
                'amplitudes': amplitudes.tolist(),
                'predicted_purities': purity_predictions,
                'current_amplitude': clipped_base['amplitude'],
                'current_purity': current_pred['predicted_purity'],
                'model_strategy': current_pred.get('model_strategy', 'UNKNOWN'),
            }

        except Exception as e:
            amps = np.linspace(*(amplitude_range or PARAM_RANGES['amplitude']), num_points).tolist()
            clipped_base = _clip_parameters(current_params)
            baseline = FALLBACK_PURITY_BASE
            return {
                'machine_id': machine_id,
                'amplitudes': amps,
                'predicted_purities': [round(baseline, 4)] * len(amps),
                'current_amplitude': clipped_base['amplitude'],
                'current_purity': round(baseline, 4),
                'model_strategy': 'CURVE_FALLBACK',
                'model_warning': f'{type(e).__name__}: {str(e)}'
            }

    def optimize_amplitude(self, machine_id, current_params):
        try:
            curve = self.predict_purity_curve(machine_id, current_params)

            purity_array = np.array(curve['predicted_purities'], dtype=np.float64)
            amplitude_array = np.array(curve['amplitudes'], dtype=np.float64)

            valid = np.isfinite(purity_array)
            if not valid.any():
                purity_array[:] = FALLBACK_PURITY_BASE

            max_purity_idx = int(np.argmax(purity_array))
            optimal_amplitude = float(amplitude_array[max_purity_idx])
            max_purity = float(purity_array[max_purity_idx])

            clipped_base = _clip_parameters(current_params)
            current_amp = float(clipped_base['amplitude'])
            current_pred = self.predict(machine_id, current_params)
            current_purity = float(current_pred['predicted_purity'])

            adjustment = round(optimal_amplitude - current_amp, 4)
            expected_improvement = round(max_purity - current_purity, 4)

            if abs(adjustment) < 0.005:
                direction = 'maintain'
            elif adjustment > 0:
                direction = 'increase'
            else:
                direction = 'decrease'

            return {
                'machine_id': machine_id,
                'current_amplitude': round(current_amp, 4),
                'optimal_amplitude': round(optimal_amplitude, 4),
                'suggested_adjustment': adjustment,
                'current_purity': round(current_purity, 4),
                'predicted_purity_at_optimal': round(max_purity, 4),
                'expected_improvement': max(expected_improvement, 0.0),
                'adjustment_direction': direction,
                'amplitude_range': list(PARAM_RANGES['amplitude']),
                'model_strategy': curve.get('model_strategy', 'UNKNOWN'),
                'model_warning': curve.get('model_warning'),
            }

        except Exception as e:
            clipped_base = _clip_parameters(current_params)
            current_amp = float(clipped_base['amplitude'])
            return {
                'machine_id': machine_id,
                'current_amplitude': round(current_amp, 4),
                'optimal_amplitude': round(current_amp, 4),
                'suggested_adjustment': 0.0,
                'current_purity': FALLBACK_PURITY_BASE,
                'predicted_purity_at_optimal': FALLBACK_PURITY_BASE,
                'expected_improvement': 0.0,
                'adjustment_direction': 'maintain',
                'amplitude_range': list(PARAM_RANGES['amplitude']),
                'model_strategy': 'OPTIMIZE_FALLBACK',
                'model_warning': f'{type(e).__name__}: {str(e)}'
            }

    def get_feature_importance(self, machine_id):
        if machine_id not in self.models:
            self.train(machine_id)

        model = self.models[machine_id]
        info = self.model_info.get(machine_id, {})

        if isinstance(model, Pipeline) and 'poly_features' in model.named_steps:
            poly = model.named_steps['poly_features']
            ridge = model.named_steps['ridge']
            feature_names = poly.get_feature_names_out(FEATURE_COLS)
            coefs = ridge.coef_

            importance = []
            for name, coef in zip(feature_names, coefs):
                importance.append({
                    'feature': str(name),
                    'coefficient': round(float(coef), 6)
                })

            importance.sort(key=lambda x: abs(x['coefficient']), reverse=True)

            return {
                'machine_id': machine_id,
                'feature_importance': importance,
                'intercept': round(float(ridge.intercept_), 4),
                'model_strategy': info.get('strategy', 'UNKNOWN'),
                'model_info': {k: v for k, v in info.items()
                               if k not in ['coefficients', 'feature_names', 'feature_importance']}
            }
        else:
            return {
                'machine_id': machine_id,
                'feature_importance': [],
                'intercept': None,
                'model_strategy': info.get('strategy', 'BASELINE_FALLBACK'),
                'model_info': info,
                'warning': 'Feature importance unavailable: model fell back to baseline estimator'
            }


PURITY_THRESHOLD = 99.9
FUTURE_MINUTES = 30

PARAMETER_ENERGY_COST = {
    'fan_airflow': 10.0,
    'vibration_freq': 5.0,
    'amplitude': 3.0,
    'inclination_angle': 0.5,
}

PARAMETER_RESPONSE_SPEED = {
    'amplitude': 10.0,
    'vibration_freq': 6.0,
    'inclination_angle': 5.0,
    'fan_airflow': 2.0,
}

PARAM_ADJUST_STEP = {
    'vibration_freq': 1.0,
    'inclination_angle': 0.1,
    'fan_airflow': 50.0,
    'amplitude': 0.1,
}


def _trend_slope(y_values, window=20):
    try:
        y = np.asarray(y_values, dtype=np.float64)
        y = y[np.isfinite(y)]
        if len(y) < 3:
            return 0.0
        y = y[-window:] if len(y) > window else y
        x = np.arange(len(y), dtype=np.float64)
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    except Exception:
        return 0.0


def _parameter_sensitivity(model, current_params):
    sensitivity = {}
    eps = {
        'vibration_freq': 0.5,
        'inclination_angle': 0.05,
        'fan_airflow': 20.0,
        'amplitude': 0.05,
    }
    try:
        clipped = _clip_parameters(current_params)
        X_base = np.array([[clipped[c] for c in FEATURE_COLS]], dtype=np.float64)
        try:
            y_base = float(np.asarray(model.predict(X_base)).reshape(-1)[0])
        except Exception:
            y_base = FALLBACK_PURITY_BASE
        if not np.isfinite(y_base):
            y_base = FALLBACK_PURITY_BASE

        for col in FEATURE_COLS:
            try:
                X_up = X_base.copy()
                idx = FEATURE_COLS.index(col)
                X_up[0, idx] += eps[col]
                try:
                    y_up = float(np.asarray(model.predict(X_up)).reshape(-1)[0])
                except Exception:
                    y_up = y_base
                if not np.isfinite(y_up):
                    y_up = y_base
                dy = y_up - y_base
                if abs(dy) < 1e-8:
                    sensitivity[col] = 0.0
                else:
                    sensitivity[col] = float(dy / eps[col])
            except Exception:
                sensitivity[col] = 0.0
    except Exception:
        for col in FEATURE_COLS:
            sensitivity[col] = 0.0
    return sensitivity


def _predict_future_purity(purity_series, current_purity, minutes_ahead=FUTURE_MINUTES):
    try:
        slope_per_sample = _trend_slope(purity_series)
        extrapolation = slope_per_sample * max(1, minutes_ahead // 5)
        future_purity = float(current_purity + extrapolation)
        if not np.isfinite(future_purity):
            future_purity = float(current_purity)
        return float(np.clip(future_purity, 90.0, 99.99)), slope_per_sample
    except Exception:
        return float(current_purity), 0.0


def _compute_adjustment_plan(machine_id, model, current_params, sensitivity,
                             current_purity, target_purity=PURITY_THRESHOLD,
                             priority='energy'):
    try:
        clipped = _clip_parameters(current_params)
        purity_gap = float(target_purity) - float(current_purity)
        if purity_gap <= 0.0:
            return None

        if priority == 'energy':
            param_order = sorted(
                FEATURE_COLS,
                key=lambda c: (PARAMETER_ENERGY_COST.get(c, 999),
                               -abs(sensitivity.get(c, 0.0)))
            )
        else:
            param_order = sorted(
                FEATURE_COLS,
                key=lambda c: (-PARAMETER_RESPONSE_SPEED.get(c, 0),
                               -abs(sensitivity.get(c, 0.0)))
            )

        actions = []
        remaining_gap = purity_gap
        max_iters = 6
        iterations = 0

        while remaining_gap > 0.001 and iterations < max_iters:
            iterations += 1
            advanced = False
            for col in param_order:
                s = sensitivity.get(col, 0.0)
                if abs(s) < 1e-6:
                    continue
                step = PARAM_ADJUST_STEP.get(col, 1.0)
                lo, hi = PARAM_RANGES.get(col, (0.0, 1e9))
                cur = float(clipped[col])

                best_delta_purity = 0.0
                best_dir = 0
                for direction in [+1, -1]:
                    new_val = cur + direction * step
                    if new_val < lo - 1e-9 or new_val > hi + 1e-9:
                        continue
                    delta_purity = s * direction * step
                    if delta_purity <= 0.0:
                        continue
                    if delta_purity > best_delta_purity:
                        best_delta_purity = delta_purity
                        best_dir = direction

                if best_dir == 0:
                    continue

                apply_steps = 1
                if abs(best_delta_purity) > 1e-9:
                    max_possible = min(3, int(np.ceil(remaining_gap / abs(best_delta_purity))))
                    apply_steps = max(1, min(max_possible, 3))

                total_step = best_dir * step * apply_steps
                new_val = cur + total_step
                new_val = max(lo, min(hi, new_val))
                actual_step = new_val - cur

                if abs(actual_step) < 1e-9:
                    continue

                actual_delta_purity = s * actual_step
                if actual_delta_purity <= 0.0:
                    continue

                clipped[col] = new_val
                remaining_gap -= actual_delta_purity
                actions.append({
                    'parameter': col,
                    'from_value': round(cur, 4),
                    'to_value': round(new_val, 4),
                    'adjustment': round(actual_step, 4),
                    'expected_purity_improvement': round(min(actual_delta_purity, purity_gap), 4),
                })
                advanced = True

                if remaining_gap <= 0.001:
                    break
            if not advanced:
                break

        if not actions:
            return None

        return {
            'machine_id': machine_id,
            'current_purity': round(current_purity, 4),
            'target_purity': round(target_purity, 4),
            'purity_gap': round(purity_gap, 4),
            'actions': actions,
            'total_expected_improvement': round(
                sum(a['expected_purity_improvement'] for a in actions), 4
            ),
        }
    except Exception:
        return None


global_model = PurityPredictionModel(degree=2)


def get_global_model():
    return global_model


def compute_golden_adjustments(model=None, hours=24, target_purity=PURITY_THRESHOLD):
    if model is None:
        model = get_global_model()

    result = {
        'generated_at': None,
        'target_purity': target_purity,
        'future_minutes': FUTURE_MINUTES,
        'urgent': False,
        'machines_at_risk': [],
        'energy_plan': [],
        'speed_plan': [],
        'summary': None,
    }

    try:
        from industrial_db import get_current_parameters, query_machine_data
        from datetime import datetime

        result['generated_at'] = datetime.now().isoformat()

        machine_data = {}
        for mid in MACHINE_IDS:
            try:
                data_df = query_machine_data(mid, hours=hours)
                if data_df is None or len(data_df) == 0:
                    continue
                purity_series = data_df['purity'].values

                cur_params = get_current_parameters(mid) or {}
                pred_result = model.predict(mid, cur_params)
                current_purity = float(pred_result.get('predicted_purity', FALLBACK_PURITY_BASE))

                future_purity, slope = _predict_future_purity(
                    purity_series, current_purity, minutes_ahead=FUTURE_MINUTES
                )

                try:
                    mdl_obj = model.models.get(mid)
                    if mdl_obj is None:
                        model.train(mid)
                        mdl_obj = model.models.get(mid)
                    sensitivity = _parameter_sensitivity(mdl_obj, cur_params)
                except Exception:
                    sensitivity = {c: 0.0 for c in FEATURE_COLS}

                machine_data[mid] = {
                    'machine_id': mid,
                    'current_purity': current_purity,
                    'predicted_30min_purity': round(future_purity, 4),
                    'purity_slope_per_5min': round(slope, 6),
                    'current_params': _clip_parameters(cur_params),
                    'sensitivity': sensitivity,
                    'at_risk': future_purity < target_purity,
                    'model_strategy': pred_result.get('model_strategy', 'UNKNOWN'),
                }
            except Exception:
                continue

        at_risk = [m for m in machine_data.values() if m['at_risk']]
        result['machines_at_risk'] = [
            {k: v for k, v in m.items() if k not in ['sensitivity', 'current_params']}
            for m in at_risk
        ]
        if at_risk:
            result['urgent'] = True

        for plan_name, priority in [('energy_plan', 'energy'), ('speed_plan', 'speed')]:
            plans = []
            for mid, mdata in machine_data.items():
                if not mdata['at_risk']:
                    continue
                try:
                    mdl_obj = model.models.get(mid)
                    adj = _compute_adjustment_plan(
                        mid,
                        mdl_obj,
                        mdata['current_params'],
                        mdata['sensitivity'],
                        mdata['predicted_30min_purity'],
                        target_purity=target_purity,
                        priority=priority,
                    )
                    if adj is not None:
                        plans.append(adj)
                except Exception:
                    continue
            result[plan_name] = plans

        if at_risk:
            risk_ids = [m['machine_id'] for m in at_risk]
            energy_action_count = sum(len(p.get('actions', [])) for p in result['energy_plan'])
            speed_action_count = sum(len(p.get('actions', [])) for p in result['speed_plan'])
            worst_purity = min(m['predicted_30min_purity'] for m in at_risk)
            result['summary'] = (
                f"检测到 {len(at_risk)} 台去石机（{', '.join(risk_ids)}）"
                f"预计未来{FUTURE_MINUTES}分钟内纯度跌破 {target_purity}% 阈值 "
                f"（最低预测 {worst_purity:.2f}%）。已生成 2 组微调方案："
                f"省电方案共 {energy_action_count} 步操作，省时间方案共 {speed_action_count} 步操作。"
            )
        else:
            result['summary'] = (
                f"所有机器运行稳定，预测未来{FUTURE_MINUTES}分钟内纯度均保持在 "
                f"{target_purity}% 以上，当前无需调整。"
            )

        return result
    except Exception as e:
        result['summary'] = f'黄金方案生成异常：{type(e).__name__}: {str(e)}'
        result['error'] = f'{type(e).__name__}: {str(e)}'
        return result


if __name__ == '__main__':
    from industrial_db import generate_historical_data, get_current_parameters

    print("=== Generating heavily corrupted data (40% dirty) ===")
    generate_historical_data(dirty_ratio=0.4)

    print("\n=== Training models ===")
    model = PurityPredictionModel(degree=2)
    results = model.train_all(hours=24)
    for res in results:
        print(f"  {res.get('machine_id')}: strategy={res.get('strategy')}, "
              f"r2={res.get('r2_score')}, samples={res.get('training_samples')}")
        if 'error' in res:
            print(f"    ERROR: {res['error'][:120]}")

    print("\n=== Testing prediction robustness ===")
    for machine_id in MACHINE_IDS:
        bad_params = {
            'vibration_freq': np.nan,
            'inclination_angle': 99999,
            'fan_airflow': -500,
            'amplitude': np.inf,
            'purity': None
        }
        pred = model.predict(machine_id, bad_params)
        print(f"  {machine_id} bad params pred: {pred}")

        good_params = get_current_parameters(machine_id) or {
            'vibration_freq': 50, 'inclination_angle': 3.5,
            'fan_airflow': 1500, 'amplitude': 1.8
        }
        pred2 = model.predict(machine_id, good_params)
        opt = model.optimize_amplitude(machine_id, good_params)
        print(f"  {machine_id} good params pred: {pred2['predicted_purity']}, "
              f"opt_dir={opt['adjustment_direction']}")
