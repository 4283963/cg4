import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from datetime import datetime

from industrial_db import (
    get_current_parameters,
    query_machine_data,
    MACHINE_IDS,
    generate_historical_data,
    PARAM_RANGES,
    get_latest_quality_report,
    clean_sensor_data,
)
from purity_model import (
    get_global_model,
    FEATURE_COLS,
    FALLBACK_PURITY_BASE,
    _safe_float,
    _clip_parameters,
    compute_golden_adjustments,
    PURITY_THRESHOLD,
    FUTURE_MINUTES,
)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

model = get_global_model()

DEGRADED_FLAG_NOTE = (
    '服务正在降级模式运行：因原材料切换导致传感器数据异常，'
    '结果基于鲁棒回归或基线估算，请谨慎参考。'
)

DEFAULT_PARAMS = {
    'vibration_freq': 50.0,
    'inclination_angle': 3.5,
    'fan_airflow': 1500.0,
    'amplitude': 1.8,
}


def _safe_get_current(machine_id):
    try:
        current = get_current_parameters(machine_id)
        if current:
            return current, None
    except Exception as e:
        return None, f'{type(e).__name__}: {str(e)}'

    fallback = {
        'machine_id': machine_id,
        'timestamp': datetime.now().isoformat(),
        **DEFAULT_PARAMS,
        'stone_residual_rate': 0.05,
        'purity': FALLBACK_PURITY_BASE,
        'data_quality': {
            'quality_warning': 'PARAMS_FALLBACK',
            'machine_id': machine_id,
        },
        '_degraded': True,
    }
    return fallback, 'NO_CURRENT_DATA'


def _build_feature_params(source_dict):
    params = {}
    for col in FEATURE_COLS:
        raw = source_dict.get(col) if source_dict else None
        val = _safe_float(raw, default=DEFAULT_PARAMS[col])
        if col in PARAM_RANGES:
            lo, hi = PARAM_RANGES[col]
            val = max(lo, min(hi, val))
        params[col] = val
    return params


def _add_degraded_note(payload, warnings_list):
    if warnings_list:
        payload.setdefault('degraded', True)
        payload.setdefault('warnings', [])
        for w in warnings_list:
            if w and w not in payload['warnings']:
                payload['warnings'].append(w)
        if payload.get('degraded'):
            payload['degraded_reason'] = DEGRADED_FLAG_NOTE
    return payload


@app.errorhandler(Exception)
def handle_unexpected_exception(e):
    if isinstance(e, HTTPException):
        return jsonify({
            'error': f'{e.name}: {e.description}',
            'degraded': False,
        }), e.code
    tb = traceback.format_exc()
    app.logger.error(f'Unhandled exception: {str(e)}\n{tb}')
    return jsonify({
        'error': f'{type(e).__name__}: {str(e)}',
        'degraded': True,
        'warnings': [
            'GLOBAL_EXCEPTION_HANDLER_TRIGGERED',
            DEGRADED_FLAG_NOTE,
        ],
        'degraded_reason': DEGRADED_FLAG_NOTE,
        'timestamp': datetime.now().isoformat(),
    }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    warnings = []
    try:
        trained = list(model.models.keys())
        degraded_count = 0
        for mid in trained:
            info = model.model_info.get(mid, {})
            strat = str(info.get('strategy', ''))
            if 'FALLBACK' in strat or 'CACHED' in strat:
                degraded_count += 1

        status = 'healthy'
        if degraded_count > 0:
            status = 'degraded'
            warnings.append(f'{degraded_count} machine(s) using fallback or cached model')

        payload = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'service': 'rice-stone-separator-analysis',
            'trained_machines': len(trained),
            'degraded_machines': degraded_count,
        }
        return jsonify(_add_degraded_note(payload, warnings))
    except Exception as e:
        return jsonify({
            'status': 'degraded',
            'timestamp': datetime.now().isoformat(),
            'service': 'rice-stone-separator-analysis',
            'error': f'{type(e).__name__}: {str(e)}',
            'degraded': True,
            'warnings': [DEGRADED_FLAG_NOTE],
            'degraded_reason': DEGRADED_FLAG_NOTE,
        }), 200


@app.route('/api/machines', methods=['GET'])
def get_machines():
    warnings = []
    machines_info = []
    for machine_id in MACHINE_IDS:
        current, err = _safe_get_current(machine_id)
        if current:
            if err:
                warnings.append(f'{machine_id}: {err}')
            machines_info.append(current)

    payload = {
        'machines': machines_info,
        'count': len(machines_info),
        'timestamp': datetime.now().isoformat(),
    }
    return jsonify(_add_degraded_note(payload, warnings))


@app.route('/api/machines/<machine_id>', methods=['GET'])
def get_machine_detail(machine_id):
    warnings = []
    if machine_id not in MACHINE_IDS:
        return jsonify({'error': f'Machine {machine_id} not found'}), 404

    try:
        current, err = _safe_get_current(machine_id)
        if not current:
            current, err = _safe_get_current(machine_id)
        if err:
            warnings.append(err)
        payload = dict(current) if current else {}
        return jsonify(_add_degraded_note(payload, warnings))
    except Exception as e:
        fallback, _ = _safe_get_current(machine_id)
        payload = dict(fallback)
        payload['error'] = f'{type(e).__name__}: {str(e)}'
        return jsonify(_add_degraded_note(payload, [f'{type(e).__name__}: {str(e)}']))


@app.route('/api/machines/<machine_id>/history', methods=['GET'])
def get_machine_history(machine_id):
    warnings = []
    if machine_id not in MACHINE_IDS:
        return jsonify({'error': f'Machine {machine_id} not found'}), 404

    hours = request.args.get('hours', 24, type=int)

    try:
        raw_data = query_machine_data(machine_id, hours=hours, clean=False)
        data, qr = clean_sensor_data(raw_data, machine_id=machine_id)
        if qr and qr.get('quality_warning') != 'OK':
            warnings.append(f"Data quality: {qr.get('quality_warning')} "
                            f"(null={qr.get('null_ratio', 0):.0%}, "
                            f"outlier={qr.get('outlier_ratio', 0):.0%})")
        if data.empty:
            data = query_machine_data(machine_id, hours=min(hours * 3, 72), clean=True)
            warnings.append('Using extended 72h fallback window')
    except Exception as e:
        data = query_machine_data(machine_id, hours=72, clean=True)
        warnings.append(f'history_query_error: {type(e).__name__}')

    history = []
    try:
        for _, row in data.iterrows():
            history.append({
                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                'vibration_freq': round(float(row['vibration_freq']), 2),
                'inclination_angle': round(float(row['inclination_angle']), 2),
                'fan_airflow': round(float(row['fan_airflow']), 2),
                'amplitude': round(float(row['amplitude']), 3),
                'stone_residual_rate': round(float(row['stone_residual_rate']), 4),
                'purity': round(float(row['purity']), 3),
            })
    except Exception as e:
        warnings.append(f'history_serialize_error: {type(e).__name__}')

    payload = {
        'machine_id': machine_id,
        'hours': hours,
        'records': len(history),
        'history': history,
        'data_quality': get_latest_quality_report(machine_id, hours),
    }
    return jsonify(_add_degraded_note(payload, warnings))


@app.route('/api/predict', methods=['POST'])
def predict_purity():
    warnings = []
    data = request.get_json(silent=True) or {}

    machine_id = data.get('machine_id')
    if not machine_id:
        return jsonify({'error': 'Missing machine_id in request body'}), 400
    if machine_id not in MACHINE_IDS:
        return jsonify({'error': f'Machine {machine_id} not found'}), 404

    if 'parameters' in data and data['parameters']:
        parameters = _build_feature_params(data['parameters'])
    else:
        current, err = _safe_get_current(machine_id)
        if err:
            warnings.append(err)
        parameters = _build_feature_params(current or {})

    try:
        if machine_id not in model.models:
            model.train(machine_id)

        result = model.predict(machine_id, parameters)

        if result.get('model_warning'):
            warnings.append(result['model_warning'])
        strategy = result.get('model_strategy', '')
        if 'FALLBACK' in strategy or 'CACHED' in strategy:
            warnings.append(f'Model strategy: {strategy}')

        payload = dict(result)
        return jsonify(_add_degraded_note(payload, warnings))
    except Exception as e:
        fallback_pred = _safe_float(parameters.get('purity'), FALLBACK_PURITY_BASE)
        if not (90 < fallback_pred < 100):
            fallback_pred = FALLBACK_PURITY_BASE
        payload = {
            'predicted_purity': round(float(fallback_pred), 4),
            'input_parameters': _clip_parameters(parameters),
            'model_strategy': 'API_FALLBACK',
            'error': f'{type(e).__name__}: {str(e)}',
        }
        return jsonify(_add_degraded_note(payload, [f'{type(e).__name__}: {str(e)}']))


@app.route('/api/machines/<machine_id>/purity-curve', methods=['GET'])
def get_purity_curve(machine_id):
    warnings = []
    if machine_id not in MACHINE_IDS:
        return jsonify({'error': f'Machine {machine_id} not found'}), 404

    current, err = _safe_get_current(machine_id)
    if err:
        warnings.append(err)
    params = _build_feature_params(current or {})

    try:
        if machine_id not in model.models:
            model.train(machine_id)
        curve = model.predict_purity_curve(machine_id, params)
        if curve.get('model_warning'):
            warnings.append(curve['model_warning'])
        return jsonify(_add_degraded_note(dict(curve), warnings))
    except Exception as e:
        import numpy as np
        amps = np.linspace(*PARAM_RANGES['amplitude'], 50).tolist()
        payload = {
            'machine_id': machine_id,
            'amplitudes': amps,
            'predicted_purities': [round(FALLBACK_PURITY_BASE, 4)] * len(amps),
            'current_amplitude': params['amplitude'],
            'current_purity': round(FALLBACK_PURITY_BASE, 4),
            'model_strategy': 'API_CURVE_FALLBACK',
            'error': f'{type(e).__name__}: {str(e)}',
        }
        return jsonify(_add_degraded_note(payload, [f'{type(e).__name__}: {str(e)}']))


@app.route('/api/machines/<machine_id>/optimize', methods=['GET', 'POST'])
def optimize_amplitude(machine_id):
    warnings = []
    if machine_id not in MACHINE_IDS:
        return jsonify({'error': f'Machine {machine_id} not found'}), 404

    body = request.get_json(silent=True) if request.method == 'POST' else None
    body_params = (body or {}).get('parameters', body if body else {})
    has_custom_params = body_params and any(k in FEATURE_COLS for k in body_params.keys()) if isinstance(body_params, dict) else False

    current, err = _safe_get_current(machine_id)
    if err and not has_custom_params:
        warnings.append(err)

    if has_custom_params:
        params = _build_feature_params(body_params)
        warnings.append('OPTIMIZE_USING_CLIENT_PARAMS_INSTEAD_OF_SENSOR')
    else:
        params = _build_feature_params(current or {})

    try:
        if machine_id not in model.models:
            model.train(machine_id)
        opt = model.optimize_amplitude(machine_id, params)
        if opt.get('model_warning'):
            warnings.append(opt['model_warning'])
        strat = opt.get('model_strategy', '')
        if 'FALLBACK' in strat:
            warnings.append(f'Model strategy: {strat}')
        return jsonify(_add_degraded_note(dict(opt), warnings))
    except Exception as e:
        payload = {
            'machine_id': machine_id,
            'current_amplitude': round(float(params['amplitude']), 4),
            'optimal_amplitude': round(float(params['amplitude']), 4),
            'suggested_adjustment': 0.0,
            'current_purity': FALLBACK_PURITY_BASE,
            'predicted_purity_at_optimal': FALLBACK_PURITY_BASE,
            'expected_improvement': 0.0,
            'adjustment_direction': 'maintain',
            'amplitude_range': list(PARAM_RANGES['amplitude']),
            'model_strategy': 'API_OPTIMIZE_FALLBACK',
            'error': f'{type(e).__name__}: {str(e)}',
        }
        return jsonify(_add_degraded_note(payload, [f'{type(e).__name__}: {str(e)}']))


@app.route('/api/machines/<machine_id>/feature-importance', methods=['GET'])
def get_feature_importance(machine_id):
    warnings = []
    if machine_id not in MACHINE_IDS:
        return jsonify({'error': f'Machine {machine_id} not found'}), 404

    try:
        if machine_id not in model.models:
            model.train(machine_id)
        importance = model.get_feature_importance(machine_id)
        if importance.get('warning'):
            warnings.append(importance['warning'])
        return jsonify(_add_degraded_note(dict(importance), warnings))
    except Exception as e:
        payload = {
            'machine_id': machine_id,
            'feature_importance': [],
            'intercept': None,
            'model_strategy': 'API_IMPORTANCE_FALLBACK',
            'error': f'{type(e).__name__}: {str(e)}',
            'warning': 'Feature importance unavailable due to error',
        }
        return jsonify(_add_degraded_note(payload, [f'{type(e).__name__}: {str(e)}']))


@app.route('/api/model/train', methods=['POST'])
def train_model():
    warnings = []
    data = request.get_json(silent=True) or {}
    hours = data.get('hours', 24)
    machine_id = data.get('machine_id')

    try:
        if machine_id:
            if machine_id not in MACHINE_IDS:
                return jsonify({'error': f'Machine {machine_id} not found'}), 404
            results = [model.train(machine_id, hours)]
        else:
            results = model.train_all(hours)

        for res in results:
            err = res.get('error')
            if err:
                warnings.append(f"{res.get('machine_id')}: {err}")
            strat = res.get('strategy', '')
            if 'FALLBACK' in strat or 'CACHED' in strat:
                warnings.append(f"{res.get('machine_id')} degraded strategy: {strat}")

        payload = {
            'status': 'training_completed' if not warnings else 'training_completed_with_warnings',
            'results': results,
            'timestamp': datetime.now().isoformat(),
        }
        return jsonify(_add_degraded_note(payload, warnings))
    except Exception as e:
        payload = {
            'status': 'training_failed',
            'error': f'{type(e).__name__}: {str(e)}',
            'timestamp': datetime.now().isoformat(),
        }
        return jsonify(_add_degraded_note(payload, [f'{type(e).__name__}: {str(e)}']))


@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    warnings = []
    trained = list(model.models.keys())
    model_statuses = {}
    for mid in MACHINE_IDS:
        info = model.model_info.get(mid, {})
        err = model.training_errors.get(mid)
        if err:
            warnings.append(f'{mid}: {err}')
        model_statuses[mid] = {
            'trained': mid in trained,
            'strategy': info.get('strategy', 'NOT_TRAINED'),
            'r2_score': info.get('r2_score'),
            'training_samples': info.get('training_samples'),
            'data_issues': info.get('data_issues', []),
            'stale_model': info.get('stale_model', False),
            'error': err,
        }

    payload = {
        'trained_machines': trained,
        'all_machines': MACHINE_IDS,
        'feature_columns': FEATURE_COLS,
        'polynomial_degree': model.degree,
        'parameter_ranges': PARAM_RANGES,
        'per_machine': model_statuses,
    }
    return jsonify(_add_degraded_note(payload, warnings))


@app.route('/api/data-quality', methods=['GET'])
def data_quality_overview():
    warnings = []
    machine_id = request.args.get('machine_id')
    hours = request.args.get('hours', 24, type=int)

    reports = {}
    if machine_id:
        if machine_id not in MACHINE_IDS:
            return jsonify({'error': f'Machine {machine_id} not found'}), 404
        ids_to_check = [machine_id]
    else:
        ids_to_check = MACHINE_IDS

    for mid in ids_to_check:
        try:
            raw = query_machine_data(mid, hours=hours, clean=False)
            cleaned, report = clean_sensor_data(raw, machine_id=mid)
            reports[mid] = report
            if report.get('quality_warning') != 'OK':
                warnings.append(f'{mid}: {report.get("quality_warning")}')
        except Exception as e:
            reports[mid] = {'error': f'{type(e).__name__}: {str(e)}'}
            warnings.append(f'{mid}: query failed')

    payload = {
        'hours': hours,
        'reports': reports,
        'timestamp': datetime.now().isoformat(),
    }
    return jsonify(_add_degraded_note(payload, warnings))


@app.route('/api/data/refresh', methods=['GET', 'POST'])
def refresh_data():
    warnings = []
    try:
        body = request.get_json(silent=True) or {}

        def _get_float(name, default):
            v = body.get(name)
            if v is None:
                v = request.args.get(name)
            if v is None:
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        dirty_ratio = _get_float('dirty_ratio', 0.0)
        purity_start = _get_float('purity_start_level', None)
        purity_trend = _get_float('purity_trend', 0.0)

        data = generate_historical_data(
            dirty_ratio=dirty_ratio,
            purity_start_level=purity_start,
            purity_trend=purity_trend,
        )

        train_results = model.train_all()
        for res in train_results:
            if res.get('error'):
                warnings.append(f"{res.get('machine_id')}: {res['error']}")

        payload = {
            'status': 'data_refreshed',
            'records_generated': len(data),
            'dirty_ratio': dirty_ratio,
            'purity_start_level': purity_start,
            'purity_trend': purity_trend,
            'training_results': train_results,
            'timestamp': datetime.now().isoformat(),
        }
        return jsonify(_add_degraded_note(payload, warnings))
    except Exception as e:
        payload = {
            'status': 'data_refresh_partial',
            'error': f'{type(e).__name__}: {str(e)}',
            'timestamp': datetime.now().isoformat(),
        }
        return jsonify(_add_degraded_note(payload, [f'{type(e).__name__}: {str(e)}']))


@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    warnings = []
    machines_info = []

    for machine_id in MACHINE_IDS:
        try:
            current, err = _safe_get_current(machine_id)
            if err:
                warnings.append(f'{machine_id}: {err}')
            if not current:
                continue
            params = _build_feature_params(current)

            if machine_id not in model.models:
                model.train(machine_id)

            optimization = model.optimize_amplitude(machine_id, params)
            if optimization.get('model_warning'):
                warnings.append(f'{machine_id}: {optimization["model_warning"]}')

            curve = model.predict_purity_curve(machine_id, params)

            machines_info.append({
                'current': current,
                'optimization': optimization,
                'curve': {
                    'current_amplitude': curve.get('current_amplitude'),
                    'current_purity': curve.get('current_purity'),
                    'amplitudes': curve.get('amplitudes', []),
                    'predicted_purities': curve.get('predicted_purities', []),
                    'model_strategy': curve.get('model_strategy'),
                },
            })
        except Exception as e:
            warnings.append(f'{machine_id}: dashboard build failed ({type(e).__name__})')
            machines_info.append({
                'machine_id': machine_id,
                'error': f'{type(e).__name__}: {str(e)}',
                'degraded': True,
            })

    payload = {
        'machines': machines_info,
        'timestamp': datetime.now().isoformat(),
    }
    return jsonify(_add_degraded_note(payload, warnings))


@app.route('/api/golden-adjustments', methods=['GET'])
def get_golden_adjustments():
    warnings = []
    try:
        hours = request.args.get('hours', 24, type=int)
        target_raw = request.args.get('target_purity')
        try:
            target = float(target_raw) if target_raw is not None else PURITY_THRESHOLD
        except (TypeError, ValueError):
            target = PURITY_THRESHOLD
            warnings.append('INVALID_TARGET_PURITY_FALLBACK')

        result = compute_golden_adjustments(model=model, hours=hours, target_purity=target)

        if result.get('urgent'):
            warnings.append('PURITY_DROP_IMMINENT: Golden adjustment suggestions issued')
        if result.get('error'):
            warnings.append(result['error'])

        return jsonify(_add_degraded_note(result, warnings))
    except Exception as e:
        fallback = {
            'generated_at': datetime.now().isoformat(),
            'target_purity': PURITY_THRESHOLD,
            'future_minutes': FUTURE_MINUTES,
            'urgent': False,
            'machines_at_risk': [],
            'energy_plan': [],
            'speed_plan': [],
            'summary': f'黄金方案生成异常（降级兜底）：{type(e).__name__}',
            'error': f'{type(e).__name__}: {str(e)}',
        }
        return jsonify(_add_degraded_note(fallback, [f'GOLDEN_ADJ_FALLBACK: {type(e).__name__}: {str(e)}']))


if __name__ == '__main__':
    print("Initializing purity prediction models (robust mode)...")
    try:
        init_results = model.train_all(hours=24)
        degraded = sum(1 for r in init_results if 'FALLBACK' in str(r.get('strategy', '')) or r.get('error'))
        print(f"Models trained. {degraded}/{len(init_results)} machine(s) using fallback estimators.")
    except Exception as e:
        print(f"Model init hit error (continuing with lazy loading): {type(e).__name__}: {e}")
    print("Starting server on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
