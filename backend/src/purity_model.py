import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from datetime import timedelta
from industrial_db import query_machine_data, MACHINE_IDS, PARAM_RANGES

FEATURE_COLS = ['vibration_freq', 'inclination_angle', 'fan_airflow', 'amplitude']
TARGET_COL = 'purity'

class PurityPredictionModel:
    def __init__(self, degree=2):
        self.degree = degree
        self.models = {}
        self.poly_features = {}
    
    def train(self, machine_id, hours=24):
        data = query_machine_data(machine_id, hours=hours)
        if data.empty:
            raise ValueError(f"No data available for machine {machine_id}")
        
        X = data[FEATURE_COLS].values
        y = data[TARGET_COL].values
        
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        model = Pipeline([
            ('poly_features', poly),
            ('linear_regression', LinearRegression())
        ])
        model.fit(X, y)
        
        self.models[machine_id] = model
        self.poly_features[machine_id] = poly
        
        score = model.score(X, y)
        return {
            'machine_id': machine_id,
            'r2_score': round(score, 4),
            'training_samples': len(data),
            'coefficients': model.named_steps['linear_regression'].coef_.tolist(),
            'intercept': float(model.named_steps['linear_regression'].intercept_)
        }
    
    def train_all(self, hours=24):
        results = []
        for machine_id in MACHINE_IDS:
            try:
                result = self.train(machine_id, hours)
                results.append(result)
            except Exception as e:
                results.append({'machine_id': machine_id, 'error': str(e)})
        return results
    
    def predict(self, machine_id, params):
        if machine_id not in self.models:
            self.train(machine_id)
        
        model = self.models[machine_id]
        X = np.array([[params[col] for col in FEATURE_COLS]])
        predicted_purity = model.predict(X)[0]
        
        return {
            'predicted_purity': round(float(predicted_purity), 4),
            'input_parameters': params
        }
    
    def predict_purity_curve(self, machine_id, current_params, amplitude_range=None, num_points=50):
        if machine_id not in self.models:
            self.train(machine_id)
        
        if amplitude_range is None:
            amplitude_range = PARAM_RANGES['amplitude']
        
        amplitudes = np.linspace(amplitude_range[0], amplitude_range[1], num_points)
        purity_predictions = []
        
        for amp in amplitudes:
            test_params = current_params.copy()
            test_params['amplitude'] = amp
            X = np.array([[test_params[col] for col in FEATURE_COLS]])
            pred = self.models[machine_id].predict(X)[0]
            purity_predictions.append(round(float(pred), 4))
        
        return {
            'machine_id': machine_id,
            'amplitudes': amplitudes.tolist(),
            'predicted_purities': purity_predictions,
            'current_amplitude': current_params['amplitude'],
            'current_purity': self.predict(machine_id, current_params)['predicted_purity']
        }
    
    def optimize_amplitude(self, machine_id, current_params):
        curve = self.predict_purity_curve(machine_id, current_params)
        
        purity_array = np.array(curve['predicted_purities'])
        amplitude_array = np.array(curve['amplitudes'])
        
        max_purity_idx = np.argmax(purity_array)
        optimal_amplitude = amplitude_array[max_purity_idx]
        max_purity = purity_array[max_purity_idx]
        
        current_amp = current_params['amplitude']
        current_purity = self.predict(machine_id, current_params)['predicted_purity']
        
        adjustment = round(optimal_amplitude - current_amp, 4)
        
        expected_improvement = round(max_purity - current_purity, 4)
        
        return {
            'machine_id': machine_id,
            'current_amplitude': current_amp,
            'optimal_amplitude': round(float(optimal_amplitude), 4),
            'suggested_adjustment': adjustment,
            'current_purity': current_purity,
            'predicted_purity_at_optimal': round(float(max_purity), 4),
            'expected_improvement': expected_improvement,
            'adjustment_direction': 'increase' if adjustment > 0 else 'decrease' if adjustment < 0 else 'maintain',
            'amplitude_range': PARAM_RANGES['amplitude']
        }
    
    def get_feature_importance(self, machine_id):
        if machine_id not in self.models:
            self.train(machine_id)
        
        model = self.models[machine_id]
        poly = self.poly_features[machine_id]
        
        coefs = model.named_steps['linear_regression'].coef_
        feature_names = poly.get_feature_names_out(FEATURE_COLS)
        
        importance = []
        for name, coef in zip(feature_names, coefs):
            importance.append({
                'feature': name,
                'coefficient': round(float(coef), 6)
            })
        
        importance.sort(key=lambda x: abs(x['coefficient']), reverse=True)
        
        return {
            'machine_id': machine_id,
            'feature_importance': importance,
            'intercept': round(float(model.named_steps['linear_regression'].intercept_), 4)
        }

global_model = PurityPredictionModel(degree=2)

def get_global_model():
    return global_model

if __name__ == '__main__':
    model = PurityPredictionModel(degree=2)
    
    print("Training models for all machines...")
    results = model.train_all(hours=24)
    for res in results:
        print(res)
    
    print("\n" + "="*50)
    from industrial_db import get_current_parameters
    
    for machine_id in MACHINE_IDS[:1]:
        current = get_current_parameters(machine_id)
        if current:
            print(f"\nMachine {machine_id}:")
            print(f"Current params: {current}")
            
            pred = model.predict(machine_id, current)
            print(f"Predicted purity: {pred['predicted_purity']}")
            
            opt = model.optimize_amplitude(machine_id, current)
            print(f"Optimization result: {opt}")
            
            importance = model.get_feature_importance(machine_id)
            print(f"Top features: {importance['feature_importance'][:5]}")
