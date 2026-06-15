import React, { useState, useEffect } from 'react';
import MachineList from './components/MachineList';
import PurityCurveChart from './components/PurityCurveChart';
import HistoryChart from './components/HistoryChart';
import AmplitudeControlPanel from './components/AmplitudeControlPanel';
import {
  getAllMachines,
  getMachineDetail,
  getMachineHistory,
  getPurityCurve,
  getOptimization,
  getFeatureImportance,
  predictPurity,
  trainModel,
  refreshData,
  healthCheck
} from './services/api';

function App() {
  const [machines, setMachines] = useState([]);
  const [selectedMachineId, setSelectedMachineId] = useState(null);
  const [currentParams, setCurrentParams] = useState(null);
  const [historyData, setHistoryData] = useState([]);
  const [curveData, setCurveData] = useState(null);
  const [optimization, setOptimization] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [systemStatus, setSystemStatus] = useState('loading');

  useEffect(() => {
    loadMachines();
    checkHealth();
    const interval = setInterval(loadMachines, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (selectedMachineId) {
      loadMachineData(selectedMachineId);
    }
  }, [selectedMachineId]);

  const checkHealth = async () => {
    try {
      const result = await healthCheck();
      setSystemStatus(result.status);
    } catch (err) {
      setSystemStatus('error');
    }
  };

  const loadMachines = async () => {
    try {
      setError(null);
      const result = await getAllMachines();
      setMachines(result.machines);
      setLastUpdate(new Date(result.timestamp));
      
      if (!selectedMachineId && result.machines.length > 0) {
        setSelectedMachineId(result.machines[0].machine_id);
      }
    } catch (err) {
      setError('加载去石机列表失败，请检查后端服务是否正常');
      console.error('Load machines error:', err);
    }
  };

  const loadMachineData = async (machineId) => {
    setLoading(true);
    setError(null);
    try {
      const [detail, history, curve, opt, importance] = await Promise.all([
        getMachineDetail(machineId),
        getMachineHistory(machineId, 24),
        getPurityCurve(machineId),
        getOptimization(machineId),
        getFeatureImportance(machineId),
      ]);
      
      setCurrentParams(detail);
      setHistoryData(history.history || []);
      setCurveData(curve);
      setOptimization(opt);
      setFeatureImportance(importance);
    } catch (err) {
      setError(`加载 ${machineId} 数据失败: ${err.message}`);
      console.error('Load machine data error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectMachine = (machineId) => {
    setSelectedMachineId(machineId);
  };

  const handlePredict = async (machineId, params) => {
    const result = await predictPurity(machineId, params);
    return result;
  };

  const handleTrainModel = async () => {
    try {
      setError(null);
      await trainModel(24);
      if (selectedMachineId) {
        loadMachineData(selectedMachineId);
      }
      alert('模型训练完成！');
    } catch (err) {
      setError('模型训练失败');
      console.error('Train model error:', err);
    }
  };

  const handleRefreshData = async () => {
    try {
      setError(null);
      await refreshData();
      loadMachines();
      if (selectedMachineId) {
        loadMachineData(selectedMachineId);
      }
      alert('数据刷新完成！');
    } catch (err) {
      setError('数据刷新失败');
      console.error('Refresh data error:', err);
    }
  };

  const formatTime = (date) => {
    if (!date) return '--';
    return date.toLocaleString('zh-CN', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const getAdjustmentClass = (direction) => {
    if (direction === 'increase') return 'adjustment-increase';
    if (direction === 'decrease') return 'adjustment-decrease';
    return 'adjustment-maintain';
  };

  const getAdjustmentText = (direction) => {
    if (direction === 'increase') return '↑ 调增';
    if (direction === 'decrease') return '↓ 调减';
    return '→ 保持';
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🌾 大米重力去石机群 - 质量工艺分析台</h1>
        <p className="subtitle">振幅微调与加工纯度预测系统 · 基于多元线性回归的智能分析</p>
        
        <div className="status-bar">
          <div className="status-item">
            <span className="status-dot" style={{ 
              background: systemStatus === 'healthy' ? '#48bb78' : 
                         systemStatus === 'loading' ? '#ecc94b' : '#f56565' 
            }}></span>
            <span>后端服务: {systemStatus === 'healthy' ? '正常' : 
                           systemStatus === 'loading' ? '连接中...' : '异常'}</span>
          </div>
          <div className="status-item">
            <span>更新时间: {formatTime(lastUpdate)}</span>
          </div>
          <div className="status-item">
            <span>去石机数量: {machines.length} 台</span>
          </div>
          
          <div className="action-buttons" style={{ marginLeft: 'auto' }}>
            <button className="action-btn" onClick={handleTrainModel}>
              📊 重新训练模型
            </button>
            <button className="action-btn" onClick={handleRefreshData}>
              🔄 刷新模拟数据
            </button>
            <button className="action-btn" onClick={loadMachines}>
              ↻ 刷新
            </button>
          </div>
        </div>
      </header>

      {error && <div className="error-message">⚠️ {error}</div>}

      <div className="main-grid">
        <MachineList
          machines={machines}
          selectedMachineId={selectedMachineId}
          onSelectMachine={handleSelectMachine}
        />

        <div className="detail-panel">
          {loading ? (
            <div className="loading">
              <div className="spinner"></div>
            </div>
          ) : selectedMachineId && currentParams ? (
            <>
              <div className="detail-header">
                <div>
                  <h2>{selectedMachineId} · 工艺参数与纯度分析</h2>
                  <p className="update-time">
                    数据更新于 {new Date(currentParams.timestamp).toLocaleString('zh-CN')}
                  </p>
                </div>
              </div>

              <div className="current-params">
                <div className="param-card">
                  <div className="param-label">振幅</div>
                  <div className="param-value">
                    {currentParams.amplitude?.toFixed(3)}
                    <span className="param-unit">mm</span>
                  </div>
                </div>
                <div className="param-card">
                  <div className="param-label">振动频率</div>
                  <div className="param-value">
                    {currentParams.vibration_freq?.toFixed(1)}
                    <span className="param-unit">Hz</span>
                  </div>
                </div>
                <div className="param-card">
                  <div className="param-label">倾角</div>
                  <div className="param-value">
                    {currentParams.inclination_angle?.toFixed(2)}
                    <span className="param-unit">°</span>
                  </div>
                </div>
                <div className="param-card">
                  <div className="param-label">风机风量</div>
                  <div className="param-value">
                    {currentParams.fan_airflow?.toFixed(0)}
                    <span className="param-unit">m³/h</span>
                  </div>
                </div>
              </div>

              <div className="current-params">
                <div className="param-card">
                  <div className="param-label">当前纯度</div>
                  <div className="param-value" style={{ 
                    color: currentParams.purity >= 99 ? '#276749' : 
                           currentParams.purity >= 98 ? '#9c4221' : '#742a2a' 
                  }}>
                    {currentParams.purity?.toFixed(3)}
                    <span className="param-unit">%</span>
                  </div>
                </div>
                <div className="param-card">
                  <div className="param-label">砂石残留率</div>
                  <div className="param-value">
                    {currentParams.stone_residual_rate?.toFixed(4)}
                    <span className="param-unit">%</span>
                  </div>
                </div>
                {optimization && (
                  <>
                    <div className="param-card">
                      <div className="param-label">预测最优纯度</div>
                      <div className="param-value" style={{ color: '#276749' }}>
                        {optimization.predicted_purity_at_optimal?.toFixed(3)}
                        <span className="param-unit">%</span>
                      </div>
                    </div>
                    <div className="param-card">
                      <div className="param-label">预期提升</div>
                      <div className="param-value" style={{ color: '#2b6cb0' }}>
                        +{optimization.expected_improvement?.toFixed(3)}
                        <span className="param-unit">%</span>
                      </div>
                    </div>
                  </>
                )}
              </div>

              {optimization && (
                <div className="optimization-panel">
                  <h3>🎯 振幅优化建议</h3>
                  <div className="optimization-content">
                    <div className="optimization-item">
                      <div className="label">当前振幅</div>
                      <div className="value">
                        {optimization.current_amplitude?.toFixed(4)}
                        <span className="unit">mm</span>
                      </div>
                    </div>
                    <div className="optimization-item">
                      <div className="label">建议调整</div>
                      <div>
                        <div className="value">
                          {Math.abs(optimization.suggested_adjustment)?.toFixed(4)}
                          <span className="unit">mm</span>
                        </div>
                        <span className={`adjustment-badge ${getAdjustmentClass(optimization.adjustment_direction)}`}>
                          {getAdjustmentText(optimization.adjustment_direction)}
                        </span>
                      </div>
                    </div>
                    <div className="optimization-item">
                      <div className="label">最优振幅</div>
                      <div className="value">
                        {optimization.optimal_amplitude?.toFixed(4)}
                        <span className="unit">mm</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="charts-section">
                <div className="chart-card">
                  <h3>📈 振幅-纯度预测曲线</h3>
                  <PurityCurveChart curveData={curveData} optimization={optimization} />
                </div>
                <div className="chart-card">
                  <h3>📊 24小时纯度与振幅趋势</h3>
                  <HistoryChart historyData={historyData} />
                </div>
              </div>

              <AmplitudeControlPanel
                machineId={selectedMachineId}
                currentParams={currentParams}
                onPredict={handlePredict}
                optimization={optimization}
              />

              {featureImportance && (
                <div className="feature-importance">
                  <h3>🔬 特征重要性分析 (多项式回归系数)</h3>
                  <div className="feature-list">
                    {featureImportance.feature_importance?.slice(0, 8).map((item, idx) => (
                      <div key={idx} className="feature-item">
                        <div className="feature-name">{item.feature}</div>
                        <div className="feature-coef">系数: {item.coefficient}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ marginTop: '12px', fontSize: '12px', color: '#718096' }}>
                    截距: {featureImportance.intercept}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="loading">
              <span>请从左侧选择一台去石机查看详细分析</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
