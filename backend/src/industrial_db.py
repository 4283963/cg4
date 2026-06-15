import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

MACHINE_IDS = ['CG-STONE-001', 'CG-STONE-002', 'CG-STONE-003', 'CG-STONE-004']

PARAM_RANGES = {
    'vibration_freq': (45.0, 55.0),
    'inclination_angle': (2.0, 5.0),
    'fan_airflow': (1200.0, 1800.0),
    'amplitude': (1.2, 2.5),
}

NUMERIC_COLS = ['vibration_freq', 'inclination_angle', 'fan_airflow',
                'amplitude', 'stone_count', 'total_count',
                'stone_residual_rate', 'purity']

DATA_QUALITY_THRESHOLDS = {
    'max_null_ratio': 0.5,
    'max_outlier_ratio': 0.4,
    'min_valid_samples': 20,
}

def _iqr_outlier_mask(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        return pd.Series([False] * len(series), index=series.index)
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)

def _zscore_outlier_mask(series, threshold=3.0):
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series([False] * len(series), index=series.index)
    z = np.abs((series - mean) / std)
    return z > threshold

def clean_sensor_data(df, machine_id=None):
    if df is None or len(df) == 0:
        return df, {
            'original_count': 0,
            'cleaned_count': 0,
            'null_count': 0,
            'outlier_count': 0,
            'dropped_ratio': 1.0,
            'quality_warning': 'NO_DATA',
            'machine_id': machine_id,
        }

    df = df.copy()
    original_count = len(df)

    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    null_mask = df[NUMERIC_COLS].isnull().any(axis=1)
    null_count = int(null_mask.sum())

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue

        if df[col].isnull().all():
            if col in PARAM_RANGES:
                fill_val = (PARAM_RANGES[col][0] + PARAM_RANGES[col][1]) / 2
            elif col == 'purity':
                fill_val = 99.0
            elif col == 'stone_residual_rate':
                fill_val = 0.05
            elif col == 'stone_count':
                fill_val = 5
            elif col == 'total_count':
                fill_val = 10000
            else:
                fill_val = 0.0
            df[col] = fill_val
            continue

        if 'timestamp' in df.columns:
            try:
                df[col] = df[col].interpolate(method='time', limit_direction='both')
            except Exception:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')

        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill()

        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    outlier_count = 0
    for col in ['vibration_freq', 'inclination_angle', 'fan_airflow', 'amplitude', 'purity']:
        if col not in df.columns:
            continue

        if col in PARAM_RANGES:
            lo, hi = PARAM_RANGES[col]
            range_mask = (df[col] < lo) | (df[col] > hi)
            if col == 'purity':
                df.loc[df[col] < 90, col] = np.nan
                df.loc[df[col] > 99.99, col] = np.nan
            else:
                df.loc[range_mask, col] = np.nan

        iqr_mask = _iqr_outlier_mask(df[col], k=2.0)
        z_mask = _zscore_outlier_mask(df[col], threshold=2.5)
        combined_mask = iqr_mask | z_mask

        outlier_count += int(combined_mask.sum())

        if combined_mask.any():
            median_val = df.loc[~combined_mask, col].median()
            if pd.isna(median_val):
                if col in PARAM_RANGES:
                    median_val = (PARAM_RANGES[col][0] + PARAM_RANGES[col][1]) / 2
                elif col == 'purity':
                    median_val = 99.0
            df.loc[combined_mask, col] = median_val

        if col in PARAM_RANGES:
            lo, hi = PARAM_RANGES[col]
            df[col] = df[col].clip(lower=lo, upper=hi)
        elif col == 'purity':
            df[col] = df[col].clip(lower=90.0, upper=99.99)
        elif col == 'stone_residual_rate':
            df[col] = df[col].clip(lower=0.0, upper=5.0)

    df = df.dropna(subset=NUMERIC_COLS, how='any')
    cleaned_count = len(df)

    null_ratio = null_count / max(original_count, 1)
    outlier_ratio = outlier_count / max(original_count, 1)
    dropped_ratio = 1.0 - (cleaned_count / max(original_count, 1))

    quality_warning = 'OK'
    if cleaned_count < DATA_QUALITY_THRESHOLDS['min_valid_samples']:
        quality_warning = 'INSUFFICIENT_SAMPLES'
    elif null_ratio > DATA_QUALITY_THRESHOLDS['max_null_ratio']:
        quality_warning = 'HIGH_NULL_RATIO'
    elif outlier_ratio > DATA_QUALITY_THRESHOLDS['max_outlier_ratio']:
        quality_warning = 'HIGH_OUTLIER_RATIO'

    quality_report = {
        'original_count': original_count,
        'cleaned_count': cleaned_count,
        'null_count': null_count,
        'outlier_count': outlier_count,
        'null_ratio': round(null_ratio, 4),
        'outlier_ratio': round(outlier_ratio, 4),
        'dropped_ratio': round(dropped_ratio, 4),
        'quality_warning': quality_warning,
        'machine_id': machine_id,
    }

    return df, quality_report

def generate_historical_data(hours=24, interval_minutes=5, dirty_ratio=0.0,
                             purity_start_level=None, purity_trend=0.0):
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    time_index = pd.date_range(start=start_time, end=end_time, freq=f'{interval_minutes}min')
    n = len(time_index)

    all_data = []

    for machine_id in MACHINE_IDS:
        base_freq = np.random.uniform(47, 53)
        base_angle = np.random.uniform(2.5, 4.5)
        base_airflow = np.random.uniform(1300, 1700)
        base_amplitude = np.random.uniform(1.5, 2.2)

        freq_noise = np.random.normal(0, 0.8, n)
        angle_noise = np.random.normal(0, 0.3, n)
        airflow_noise = np.random.normal(0, 80, n)
        amplitude_noise = np.random.normal(0, 0.15, n)

        vibration_freq = np.clip(base_freq + freq_noise, *PARAM_RANGES['vibration_freq'])
        inclination_angle = np.clip(base_angle + angle_noise, *PARAM_RANGES['inclination_angle'])
        fan_airflow = np.clip(base_airflow + airflow_noise, *PARAM_RANGES['fan_airflow'])
        amplitude = np.clip(base_amplitude + amplitude_noise, *PARAM_RANGES['amplitude'])

        ideal_amplitude = 1.8 + 0.05 * (vibration_freq - 50) - 0.1 * (inclination_angle - 3.5) + 0.0003 * (fan_airflow - 1500)
        amplitude_deviation = np.abs(amplitude - ideal_amplitude)

        if purity_start_level is None:
            base_purity = 99.85 + np.random.uniform(-0.05, 0.1)
        else:
            base_purity = float(purity_start_level)

        purity_loss = amplitude_deviation * 1.2 + np.random.normal(0, 0.08, n)

        time_scalar = np.linspace(0.0, 1.0, n)
        trend_component = time_scalar * float(purity_trend)

        purity = np.clip(
            base_purity + trend_component - purity_loss,
            95.0, 99.99
        )

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

        if dirty_ratio > 0:
            n_dirty = int(len(machine_df) * dirty_ratio)
            if n_dirty > 0:
                dirty_idx = np.random.choice(len(machine_df), size=n_dirty, replace=False)
                for idx in dirty_idx:
                    col = np.random.choice(NUMERIC_COLS)
                    if np.random.random() < 0.4:
                        machine_df.loc[idx, col] = np.nan
                    else:
                        if col in PARAM_RANGES:
                            lo, hi = PARAM_RANGES[col]
                            machine_df.loc[idx, col] = np.random.choice([
                                lo * np.random.uniform(0.01, 0.3),
                                hi * np.random.uniform(2.0, 5.0)
                            ])
                        elif col == 'purity':
                            machine_df.loc[idx, col] = np.random.choice([50.0, 150.0, -10.0])
                        else:
                            machine_df.loc[idx, col] = machine_df.loc[idx, col] * 100

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

_quality_reports_cache = {}

def query_machine_data(machine_id=None, hours=24, clean=True):
    df = get_db_connection()
    end_time = df['timestamp'].max()
    start_time = end_time - timedelta(hours=hours)

    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
    if machine_id:
        mask = mask & (df['machine_id'] == machine_id)

    raw_df = df[mask].copy()

    if not clean:
        return raw_df

    cleaned_df, report = clean_sensor_data(raw_df, machine_id=machine_id)

    cache_key = f"{machine_id or 'all'}_{hours}"
    _quality_reports_cache[cache_key] = report

    if len(cleaned_df) < DATA_QUALITY_THRESHOLDS['min_valid_samples']:
        fallback_hours = min(hours * 3, 72)
        fallback_mask = (df['timestamp'] >= end_time - timedelta(hours=fallback_hours))
        if machine_id:
            fallback_mask = fallback_mask & (df['machine_id'] == machine_id)
        fallback_raw = df[fallback_mask].copy()
        fallback_cleaned, fallback_report = clean_sensor_data(fallback_raw, machine_id=machine_id)
        fallback_report['fallback_from_hours'] = fallback_hours
        _quality_reports_cache[cache_key] = fallback_report
        return fallback_cleaned

    return cleaned_df

def get_latest_quality_report(machine_id=None, hours=24):
    cache_key = f"{machine_id or 'all'}_{hours}"
    return _quality_reports_cache.get(cache_key)

def get_current_parameters(machine_id):
    df = query_machine_data(machine_id, hours=1, clean=True)
    if df.empty:
        df = query_machine_data(machine_id, hours=6, clean=True)
    if df.empty:
        df = query_machine_data(machine_id, hours=24, clean=True)
    if df.empty:
        return None

    recent = df.tail(min(5, len(df)))

    latest = df.iloc[-1]
    return {
        'machine_id': machine_id,
        'timestamp': latest['timestamp'].isoformat() if hasattr(latest['timestamp'], 'isoformat') else str(latest['timestamp']),
        'vibration_freq': round(float(recent['vibration_freq'].median()), 2),
        'inclination_angle': round(float(recent['inclination_angle'].median()), 2),
        'fan_airflow': round(float(recent['fan_airflow'].median()), 2),
        'amplitude': round(float(recent['amplitude'].median()), 3),
        'stone_residual_rate': round(float(recent['stone_residual_rate'].median()), 4),
        'purity': round(float(recent['purity'].median()), 3),
        'data_quality': get_latest_quality_report(machine_id, hours=1) or get_latest_quality_report(machine_id, hours=24)
    }

if __name__ == '__main__':
    print("=== Testing with dirty data (30% corrupt) ===")
    data = generate_historical_data(dirty_ratio=0.3)
    print(f"Generated {len(data)} records with 30% dirty data")

    for mid in MACHINE_IDS[:2]:
        raw = query_machine_data(mid, hours=24, clean=False)
        cleaned, report = clean_sensor_data(raw, machine_id=mid)
        print(f"\nMachine {mid}:")
        print(f"  Quality report: {report}")
        print(f"  Raw shape: {raw.shape}, Cleaned shape: {cleaned.shape}")
        if not cleaned.empty:
            print(f"  Cleaned purity range: [{cleaned['purity'].min():.3f}, {cleaned['purity'].max():.3f}]")
