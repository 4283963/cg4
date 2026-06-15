import React, { useState, useEffect } from 'react';
import { getGoldenAdjustments, refreshData } from '../services/api';

const PARAM_DISPLAY = {
  vibration_freq: { name: '振动频率', unit: 'Hz', emoji: '📳' },
  inclination_angle: { name: '倾角', unit: '°', emoji: '📐' },
  fan_airflow: { name: '风机风量', unit: 'm³/h', emoji: '🌀' },
  amplitude: { name: '振幅', unit: 'mm', emoji: '📏' },
};

const formatAdj = (param, value) => {
  const d = PARAM_DISPLAY[param] || { name: param, unit: '' };
  const sign = value > 0 ? '+' : '';
  return `${d.emoji} ${d.name} ${sign}${value.toFixed(
    param === 'inclination_angle' ? 2 : param === 'fan_airflow' ? 0 : param === 'amplitude' ? 3 : 1
  )}${d.unit}`;
};

function GoldenAdvicePanel() {
  const [advice, setAdvice] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activePlan, setActivePlan] = useState('energy');
  const [dismissed, setDismissed] = useState(false);

  const fetchAdvice = async () => {
    try {
      const data = await getGoldenAdjustments();
      setAdvice(data);
      if (data && data.urgent) {
        setDismissed(false);
      }
    } catch (err) {
      console.error('Golden adjustments fetch error:', err);
      setAdvice({
        summary: '黄金微调建议加载失败，正在使用上一次缓存数据',
        urgent: false,
        machines_at_risk: [],
        energy_plan: [],
        speed_plan: [],
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAdvice();
    const timer = setInterval(fetchAdvice, 15000);
    return () => clearInterval(timer);
  }, []);

  const triggerEmergency = async () => {
    setLoading(true);
    try {
      await refreshData();
      await fetchAdvice();
    } catch (err) {
      console.error(err);
    }
  };

  if (loading) {
    return (
      <aside className="golden-panel golden-panel--loading">
        <div className="golden-panel__header">
          <h3>⚡ 新手排障 · 黄金微调组合</h3>
        </div>
        <div className="loading">
          <div className="spinner spinner--small"></div>
          <span>正在分析纯度趋势...</span>
        </div>
      </aside>
    );
  }

  const showAlert = advice && advice.urgent && !dismissed;

  return (
    <aside className={`golden-panel ${showAlert ? 'golden-panel--alert' : ''}`}>
      <div className="golden-panel__header">
        <h3>⚡ 新手排障 · 黄金微调组合</h3>
        <div className="golden-panel__actions">
          <button
            className="golden-mini-btn"
            title="模拟紧急状态（纯度跌破阈值）"
            onClick={triggerEmergency}
          >
            🔔 模拟危机
          </button>
          {showAlert && (
            <button
              className="golden-mini-btn"
              onClick={() => setDismissed(true)}
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {advice && advice.summary && (
        <div className={`golden-summary ${showAlert ? 'golden-summary--urgent' : ''}`}>
          {showAlert && <span className="blink-dot">🆘</span>}
          <p>{advice.summary}</p>
          <div className="golden-meta">
            <span>目标纯度 ≥ {advice.target_purity}%</span>
            <span>预测窗口 {advice.future_minutes} 分钟</span>
          </div>
        </div>
      )}

      {advice && advice.urgent && advice.machines_at_risk && advice.machines_at_risk.length > 0 && (
        <div className="golden-risk-list">
          <div className="golden-risk-title">⚠️ 风险机器清单：</div>
          {advice.machines_at_risk.map((m) => (
            <div key={m.machine_id} className="golden-risk-item">
              <div className="golden-risk-id">{m.machine_id}</div>
              <div className="golden-risk-values">
                <span>当前纯度 <b>{m.current_purity?.toFixed?.(3) ?? m.current_purity}%</b></span>
                <span className="golden-risk-future">
                  → 30分钟后 <b style={{ color: '#742a2a' }}>{m.predicted_30min_purity?.toFixed?.(3) ?? m.predicted_30min_purity}%</b>
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {advice && advice.urgent && (
        <div className="golden-tabs">
          <button
            className={`golden-tab ${activePlan === 'energy' ? 'golden-tab--active' : ''}`}
            onClick={() => setActivePlan('energy')}
          >
            🔋 最省电方案
          </button>
          <button
            className={`golden-tab ${activePlan === 'speed' ? 'golden-tab--active' : ''}`}
            onClick={() => setActivePlan('speed')}
          >
            ⚡ 最省时间方案
          </button>
        </div>
      )}

      {advice && advice.urgent && (
        <div className="golden-plan-list">
          {(activePlan === 'energy' ? advice.energy_plan : advice.speed_plan)?.length > 0 ? (
            (activePlan === 'energy' ? advice.energy_plan : advice.speed_plan).map((plan) => (
              <div key={plan.machine_id} className="golden-card">
                <div className="golden-card__header">
                  <span className="golden-card__badge">🎯 {plan.machine_id}</span>
                  <span className="golden-card__gain">
                    预期提升 +{plan.total_expected_improvement?.toFixed?.(3) ?? 0}%
                  </span>
                </div>
                <div className="golden-card__hint">
                  当前 {plan.current_purity?.toFixed?.(3) ?? plan.current_purity}%
                  &nbsp;→&nbsp; 目标 ≥ {plan.target_purity}%
                </div>
                <ol className="golden-steps">
                  {plan.actions?.map((a, idx) => (
                    <li key={idx} className="golden-step">
                      <span className="golden-step__no">{idx + 1}</span>
                      <span className="golden-step__action">
                        {formatAdj(a.parameter, a.adjustment)}
                      </span>
                      <span className="golden-step__detail">
                        {a.from_value?.toFixed?.(2) ?? a.from_value}
                        {' → '}
                        <b>{a.to_value?.toFixed?.(2) ?? a.to_value}</b>
                        {' '}
                        <span className="golden-step__unit">
                          {PARAM_DISPLAY[a.parameter]?.unit || ''}
                        </span>
                      </span>
                      <span className="golden-step__gain">
                        +{a.expected_purity_improvement?.toFixed?.(4) ?? 0}%
                      </span>
                    </li>
                  ))}
                </ol>
                <div className="golden-card__footer">
                  🔧 工人操作提示：按步骤顺序调节相应物理阀门即可
                </div>
              </div>
            ))
          ) : (
            <div className="golden-empty">
              暂无可用的微调方案（参数已达物理极限或模型精度不足），
              建议人工检查进料稻谷质量并暂时降低产量。
            </div>
          )}
        </div>
      )}

      {!advice?.urgent && (
        <div className="golden-allgood">
          <div className="golden-allgood__emoji">✅</div>
          <div className="golden-allgood__text">
            所有去石机运行平稳，
            <br />
            预测未来 {advice?.future_minutes || 30} 分钟内纯度均保持在 {advice?.target_purity || 99.9}% 以上
          </div>
          <div className="golden-allgood__hint">
            本面板每 15 秒自动刷新 · 出现危机时会弹出高亮绿色卡片
          </div>
        </div>
      )}
    </aside>
  );
}

export default GoldenAdvicePanel;
