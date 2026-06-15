import React from 'react';

const MachineList = ({ machines, selectedMachineId, onSelectMachine }) => {
  const getPurityClass = (purity) => {
    if (purity >= 99) return 'purity-high';
    if (purity >= 98) return 'purity-medium';
    return 'purity-low';
  };

  const getPurityLabel = (purity) => {
    if (purity >= 99) return '优';
    if (purity >= 98) return '良';
    return '待优化';
  };

  if (!machines || machines.length === 0) {
    return (
      <div className="machine-list">
        <h2>去石机群</h2>
        <div className="loading">
          <div className="spinner"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="machine-list">
      <h2>去石机群</h2>
      {machines.map((machine) => (
        <div
          key={machine.machine_id}
          className={`machine-card ${selectedMachineId === machine.machine_id ? 'active' : ''}`}
          onClick={() => onSelectMachine(machine.machine_id)}
        >
          <div className="machine-id">{machine.machine_id}</div>
          <div className="machine-stats">
            <div className="stat-item">
              <span className="stat-label">振幅</span>
              <span className="stat-value">{machine.amplitude?.toFixed(3)} mm</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">频率</span>
              <span className="stat-value">{machine.vibration_freq?.toFixed(1)} Hz</span>
            </div>
          </div>
          <span className={`purity-badge ${getPurityClass(machine.purity)}`}>
            纯度 {machine.purity?.toFixed(2)}% · {getPurityLabel(machine.purity)}
          </span>
        </div>
      ))}
    </div>
  );
};

export default MachineList;
