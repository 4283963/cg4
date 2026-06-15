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


global_model = PurityPredictionModel(degree=2)


def get_global_model():
    return global_model


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
