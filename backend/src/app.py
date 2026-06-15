import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime

from industrial_db import (
    get_current_parameters, 
    query_machine_data, 
    MACHINE_IDS,
    generate_historical_data,
    PARAM_RANGES
)
from purity_model import get_global_model, FEATURE_COLS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

model = get_global_model()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'rice-stone-separator-analysis'
    })

@app.route('/api/machines', methods=['GET'])
def get_machines():
    machines_info = []
    for machine_id in MACHINE_IDS:
        current = get_current_parameters(machine_id)
        if current:
            machines_info.append(current)
    return jsonify({
        'machines': machines_info,
        'count': len(machines_info),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/machines/<machine_id>', methods=['GET'])
def get_machine_detail(machine_id):
    if machine_id not in MACHINE_IDS:
        return jsonify({'error': f'Machine {machine_id} not found'}), 404
    
    current = get_current_parameters(machine_id)
    if not current:
        return jsonify({'error': 'No data available'}), 404
    
    return jsonify(current)

@app.route('/api/machines/<machine_id>/history', methods=['GET'])
def get_machine_history(machine_id):
    if machine_id not in MACHINE_IDS:
        return jsonify({'error': f'Machine {machine_id} not found'}), 404
    
    hours = request.args.get('hours', 24, type=int)
    data = query_machine_data(machine_id, hours=hours)
    
    if data.empty:
        return jsonify({'error': 'No historical data available'}), 404
    
    history = []
    for _, row in data.iterrows():
        history.append({
            'timestamp': row['timestamp'].isoformat(),
            'vibration_freq': round(row['vibration_freq'], 2),
            'inclination_angle': round(row['inclination_angle'], 2),
            'fan_airflow': round(row['fan_airflow'], 2),
            'amplitude': round(row['amplitude'], 3),
            'stone_residual_rate': round(row['stone_residual_rate'], 4),
            'purity': round(row['purity'], 3)
        })
    
    return jsonify({
        'machine_id': machine_id,
        'hours': hours,
        'records': len(history),
        'history': history
    })

@app.route('/api/predict', methods=['POST'])
def predict_purity():
    data = request.get_json()
    
    if not data or 'machine_id' not in data:
        return jsonify({'error': 'Missing machine_id in request body'}), 400
    
    machine_id = data['machine_id']
    if machine_id not in MACHINE_IDS:
        return jsonify({'error': f'Machine {machine_id} not found'}), 404
    
    if 'parameters' not in data:
        params = get_current_parameters(machine_id)
        if params:
            parameters = {k: params[k] for k in FEATURE_COLS}
        else:
            return jsonify({'error': 'No current parameters available'}), 404
    else:
        parameters = data['parameters']
        missing = [col for col in FEATURE_COLS if col not in parameters]
        if missing:
            return jsonify({'error': f'Missing required parameters: {missing}'}), 400
    
    try:
        result = model.predict(machine_id, parameters)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/machines/<machine_id>/purity-curve', methods=['GET'])
def get_purity_curve(machine_id):
    if machine_id not in MACHINE_IDS:
        return jsonify({'error': f'Machine {machine_id} not found'}), 404
    
    current = get_current_parameters(machine_id)
    if not current:
        return jsonify({'error': 'No current parameters available'}), 404
    
    params = {k: current[k] for k in FEATURE_COLS}
    
    try:
        curve = model.predict_purity_curve(machine_id, params)
        return jsonify(curve)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/machines/<machine_id>/optimize', methods=['GET', 'POST'])
def optimize_amplitude(machine_id):
    if machine_id not in MACHINE_IDS:
        return jsonify({'error': f'Machine {machine_id} not found'}), 404
    
    current = get_current_parameters(machine_id)
    if not current:
        return jsonify({'error': 'No current parameters available'}), 404
    
    params = {k: current[k] for k in FEATURE_COLS}
    
    try:
        optimization = model.optimize_amplitude(machine_id, params)
        return jsonify(optimization)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/machines/<machine_id>/feature-importance', methods=['GET'])
def get_feature_importance(machine_id):
    if machine_id not in MACHINE_IDS:
        return jsonify({'error': f'Machine {machine_id} not found'}), 404
    
    try:
        importance = model.get_feature_importance(machine_id)
        return jsonify(importance)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/train', methods=['POST'])
def train_model():
    data = request.get_json() or {}
    hours = data.get('hours', 24)
    machine_id = data.get('machine_id')
    
    try:
        if machine_id:
            if machine_id not in MACHINE_IDS:
                return jsonify({'error': f'Machine {machine_id} not found'}), 404
            results = [model.train(machine_id, hours)]
        else:
            results = model.train_all(hours)
        
        return jsonify({
            'status': 'training_completed',
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    trained = list(model.models.keys())
    return jsonify({
        'trained_machines': trained,
        'all_machines': MACHINE_IDS,
        'feature_columns': FEATURE_COLS,
        'polynomial_degree': model.degree,
        'parameter_ranges': PARAM_RANGES
    })

@app.route('/api/data/refresh', methods=['POST'])
def refresh_data():
    try:
        data = generate_historical_data()
        model.train_all()
        return jsonify({
            'status': 'data_refreshed',
            'records_generated': len(data),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    try:
        machines_info = []
        for machine_id in MACHINE_IDS:
            current = get_current_parameters(machine_id)
            if current:
                params = {k: current[k] for k in FEATURE_COLS}
                optimization = model.optimize_amplitude(machine_id, params)
                curve = model.predict_purity_curve(machine_id, params)
                
                machines_info.append({
                    'current': current,
                    'optimization': optimization,
                    'curve': {
                        'current_amplitude': curve['current_amplitude'],
                        'current_purity': curve['current_purity'],
                        'amplitudes': curve['amplitudes'],
                        'predicted_purities': curve['predicted_purities']
                    }
                })
        
        return jsonify({
            'machines': machines_info,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Initializing purity prediction models...")
    model.train_all(hours=24)
    print("Models trained. Starting server...")
    app.run(host='0.0.0.0', port=5001, debug=True)
