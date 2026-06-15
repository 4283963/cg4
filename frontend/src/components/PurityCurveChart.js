import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const PurityCurveChart = ({ curveData, optimization }) => {
  if (!curveData || !curveData.amplitudes) {
    return (
      <div className="loading">
        <div className="spinner"></div>
      </div>
    );
  }

  const currentIdx = curveData.amplitudes.findIndex(
    amp => Math.abs(amp - curveData.current_amplitude) < 0.001
  );

  const optimalIdx = optimization
    ? curveData.amplitudes.findIndex(
        amp => Math.abs(amp - optimization.optimal_amplitude) < 0.01
      )
    : -1;

  const pointBackgroundColors = curveData.amplitudes.map((amp, idx) => {
    if (idx === currentIdx) return '#e53e3e';
    if (idx === optimalIdx) return '#38a169';
    return 'transparent';
  });

  const pointRadius = curveData.amplitudes.map((amp, idx) => {
    if (idx === currentIdx || idx === optimalIdx) return 6;
    return 0;
  });

  const data = {
    labels: curveData.amplitudes.map(a => a.toFixed(3)),
    datasets: [
      {
        label: '预测纯度 (%)',
        data: curveData.predicted_purities,
        borderColor: '#667eea',
        backgroundColor: 'rgba(102, 126, 234, 0.1)',
        fill: true,
        tension: 0.4,
        pointBackgroundColor: pointBackgroundColors,
        pointBorderColor: pointBackgroundColors.map(c => c === 'transparent' ? c : '#fff'),
        pointBorderWidth: 2,
        pointRadius: pointRadius,
        pointHoverRadius: 4,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: { size: 12 },
          usePointStyle: true,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(26, 54, 93, 0.9)',
        titleFont: { size: 13 },
        bodyFont: { size: 12 },
        padding: 12,
        cornerRadius: 8,
        callbacks: {
          title: (items) => `振幅: ${items[0].label} mm`,
          label: (item) => `预测纯度: ${item.parsed.y.toFixed(3)}%`,
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: '振幅 (mm)',
          font: { size: 13, weight: '600' },
          color: '#4a5568',
        },
        ticks: {
          maxTicksLimit: 10,
          font: { size: 11 },
          color: '#718096',
        },
        grid: {
          color: 'rgba(226, 232, 240, 0.5)',
        },
      },
      y: {
        title: {
          display: true,
          text: '预测纯度 (%)',
          font: { size: 13, weight: '600' },
          color: '#4a5568',
        },
        ticks: {
          font: { size: 11 },
          color: '#718096',
          callback: (value) => value.toFixed(2),
        },
        grid: {
          color: 'rgba(226, 232, 240, 0.5)',
        },
      },
    },
  };

  return (
    <div>
      <div style={{ height: '300px' }}>
        <Line data={data} options={options} />
      </div>
      <div style={{ display: 'flex', gap: '16px', marginTop: '12px', fontSize: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#e53e3e' }}></div>
          <span style={{ color: '#4a5568' }}>当前振幅</span>
        </div>
        {optimization && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#38a169' }}></div>
            <span style={{ color: '#4a5568' }}>最优振幅</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default PurityCurveChart;
