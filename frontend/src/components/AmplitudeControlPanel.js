import React, { useState, useEffect } from 'react';

const AmplitudeControlPanel = ({ 
  machineId, 
  currentParams, 
  onPredict, 
  optimization 
}) => {
  const [params, setParams] = useState({
    vibration_freq: 50,
    inclination_angle: 3.5,
    fan_airflow: 1500,
    amplitude: 1.8,
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);

  useEffect(() => {
    if (currentParams) {
      setParams({
        vibration_freq: currentParams.vibration_freq,
        inclination_angle: currentParams.inclination_angle,
        fan_airflow: currentParams.fan_airflow,
        amplitude: currentParams.amplitude,
      });
    }
  }, [currentParams]);

  const handleParamChange = (param, value) => {
    setParams(prev => ({
      ...prev,
      [param]: parseFloat(value),
    }));
    setPredictionResult(null);
  };

  const handlePredict = async () => {
    setIsPredicting(true);
    try {
      const result = await onPredict(machineId, params);
      setPredictionResult(result);
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setIsPredicting(false);
    }
  };

  const applyOptimal = () => {
    if (optimization) {
      setParams(prev => ({
        ...prev,
        amplitude: optimization.optimal_amplitude,
      }));
      setPredictionResult(null);
    }
  };

  const paramConfig = {
    vibration_freq: { label: '振动频率', min: 45, max: 55, step: 0.1, unit: 'Hz' },
    inclination_angle: { label: '倾角', min: 2, max: 5, step: 0.1, unit: '°' },
    fan_airflow: { label: '风机风量', min: 1200, max: 1800, step: 10, unit: 'm³/h' },
    amplitude: { label: '振幅', min: 1.2, max: 2.5, step: 0.001, unit: 'mm' },
  };

  return (
    <div className="control-panel">
      <h3>工艺参数微调 & 纯度预测</h3>

      {optimization && (
        <div style={{ marginBottom: '20px', padding: '12px', background: '#ebf8ff', borderRadius: '8px', border: '1px solid #90cdf4' }}>
          <div style={{ fontSize: '13px', color: '#2c5282', marginBottom: '8px' }}>
            💡 系统建议将振幅 <strong>{optimization.adjustment_direction === 'increase' ? '调增' : optimization.adjustment_direction === 'decrease' ? '调减' : '保持'}</strong> 
            <span style={{ fontWeight: '700', color: '#2b6cb0' }}> {Math.abs(optimization.suggested_adjustment).toFixed(4)} mm</span>
          </div>
          <button 
            className="action-btn"
            onClick={applyOptimal}
            style={{ background: '#4299e1', color: 'white' }}
          >
            应用最优振幅 ({optimization.optimal_amplitude.toFixed(4)} mm)
          </button>
        </div>
      )}

      {Object.entries(paramConfig).map(([key, config]) => (
        <div key={key} className="slider-container">
          <label>{config.label}</label>
          <input
            type="range"
            min={config.min}
            max={config.max}
            step={config.step}
            value={params[key]}
            onChange={(e) => handleParamChange(key, e.target.value)}
          />
          <span className="slider-value">
            {params[key].toFixed(config.step < 1 ? (config.step < 0.01 ? 3 : 1) : 0)} {config.unit}
          </span>
        </div>
      ))}

      <button 
        className="predict-btn" 
        onClick={handlePredict}
        disabled={isPredicting}
      >
        {isPredicting ? '预测中...' : '🔮 预测纯度'}
      </button>

      {predictionResult && (
        <div className="prediction-result">
          <div className="result-label">预测成品纯度</div>
          <div className="result-value">
            {predictionResult.predicted_purity.toFixed(3)}%
            {predictionResult.predicted_purity >= 99.5 && <span style={{ marginLeft: '8px', fontSize: '14px' }}>✨</span>}
          </div>
          <div style={{ fontSize: '12px', color: '#2c7a7b', marginTop: '4px' }}>
            基于 {machineId} 的多元线性回归模型预测
          </div>
        </div>
      )}
    </div>
  );
};

export default AmplitudeControlPanel;
