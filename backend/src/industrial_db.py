import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

MACHINE_IDS = ['CG-STONE-001', 'CG-STONE-002', 'CG-STONE-003', 'CG-STONE-004']

PARAM_RANGES = {
    'vibration_freq': (45.0, 55.0),
    'inclination_angle': (2.0, 5.0),
    'fan_airflow': (1200.0, 1800.0),
    'amplitude': (1.2, 2.5),
}

def generate_historical_data(hours=24, interval_minutes=5):
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    time_index = pd.date_range(start=start_time, end=end_time, freq=f'{interval_minutes}min')
    
    all_data = []
    
    for machine_id in MACHINE_IDS:
        base_freq = np.random.uniform(47, 53)
        base_angle = np.random.uniform(2.5, 4.5)
        base_airflow = np.random.uniform(1300, 1700)
        base_amplitude = np.random.uniform(1.5, 2.2)
        
        freq_noise = np.random.normal(0, 0.8, len(time_index))
        angle_noise = np.random.normal(0, 0.3, len(time_index))
        airflow_noise = np.random.normal(0, 80, len(time_index))
        amplitude_noise = np.random.normal(0, 0.15, len(time_index))
        
        vibration_freq = np.clip(base_freq + freq_noise, *PARAM_RANGES['vibration_freq'])
        inclination_angle = np.clip(base_angle + angle_noise, *PARAM_RANGES['inclination_angle'])
        fan_airflow = np.clip(base_airflow + airflow_noise, *PARAM_RANGES['fan_airflow'])
        amplitude = np.clip(base_amplitude + amplitude_noise, *PARAM_RANGES['amplitude'])
        
        ideal_amplitude = 1.8 + 0.05 * (vibration_freq - 50) - 0.1 * (inclination_angle - 3.5) + 0.0003 * (fan_airflow - 1500)
        amplitude_deviation = np.abs(amplitude - ideal_amplitude)
        
        base_purity = 99.5
        purity_loss = amplitude_deviation * 1.2 + np.random.normal(0, 0.08, len(time_index))
        purity = np.clip(base_purity - purity_loss, 95.0, 99.95)
        
        stone_count = np.random.poisson(lam=5, size=len(time_index))
        stone_count = stone_count + (amplitude_deviation * 8).astype(int)
        total_count = 10000
        stone_residual_rate = (stone_count / total_count) * 100
        
        machine_df = pd.DataFrame({
            'timestamp': time_index,
            'machine_id': machine_id,
            'vibration_freq': vibration_freq,
            'inclination_angle': inclination_angle,
            'fan_airflow': fan_airflow,
            'amplitude': amplitude,
            'stone_count': stone_count,
            'total_count': total_count,
            'stone_residual_rate': stone_residual_rate,
            'purity': purity
        })
        
        all_data.append(machine_df)
    
    full_data = pd.concat(all_data, ignore_index=True)
    output_path = os.path.join(data_dir, 'historical_data.csv')
    full_data.to_csv(output_path, index=False)
    
    return full_data

def get_db_connection():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_data.csv')
    if not os.path.exists(data_path):
        generate_historical_data()
    return pd.read_csv(data_path, parse_dates=['timestamp'])

def query_machine_data(machine_id=None, hours=24):
    df = get_db_connection()
    end_time = df['timestamp'].max()
    start_time = end_time - timedelta(hours=hours)
    
    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
    if machine_id:
        mask = mask & (df['machine_id'] == machine_id)
    
    return df[mask].copy()

def get_current_parameters(machine_id):
    df = query_machine_data(machine_id, hours=1)
    if df.empty:
        return None
    
    latest = df.iloc[-1]
    return {
        'machine_id': machine_id,
        'timestamp': latest['timestamp'].isoformat(),
        'vibration_freq': round(latest['vibration_freq'], 2),
        'inclination_angle': round(latest['inclination_angle'], 2),
        'fan_airflow': round(latest['fan_airflow'], 2),
        'amplitude': round(latest['amplitude'], 3),
        'stone_residual_rate': round(latest['stone_residual_rate'], 4),
        'purity': round(latest['purity'], 3)
    }

if __name__ == '__main__':
    data = generate_historical_data()
    print(f"Generated {len(data)} records")
    print(f"Columns: {data.columns.tolist()}")
    print("\nSample data:")
    print(data.head())
