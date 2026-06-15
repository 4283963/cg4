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
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const HistoryChart = ({ historyData }) => {
  if (!historyData || historyData.length === 0) {
    return (
      <div className="loading">
        <div className="spinner"></div>
      </div>
    );
  }

  const labels = historyData.map(h => {
    const date = new Date(h.timestamp);
    return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
  });

  const data = {
    labels,
    datasets: [
      {
        label: '实际纯度 (%)',
        data: historyData.map(h => h.purity),
        borderColor: '#f56565',
        backgroundColor: 'rgba(245, 101, 101, 0.1)',
        fill: false,
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 4,
        yAxisID: 'y',
      },
      {
        label: '振幅 (mm)',
        data: historyData.map(h => h.amplitude),
        borderColor: '#4299e1',
        backgroundColor: 'rgba(66, 153, 225, 0.1)',
        fill: false,
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 4,
        yAxisID: 'y1',
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
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
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: '时间',
          font: { size: 13, weight: '600' },
          color: '#4a5568',
        },
        ticks: {
          maxTicksLimit: 12,
          font: { size: 11 },
          color: '#718096',
        },
        grid: {
          color: 'rgba(226, 232, 240, 0.5)',
        },
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: '纯度 (%)',
          font: { size: 13, weight: '600' },
          color: '#e53e3e',
        },
        ticks: {
          font: { size: 11 },
          color: '#e53e3e',
        },
        grid: {
          color: 'rgba(226, 232, 240, 0.5)',
        },
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: '振幅 (mm)',
          font: { size: 13, weight: '600' },
          color: '#3182ce',
        },
        ticks: {
          font: { size: 11 },
          color: '#3182ce',
        },
        grid: {
          drawOnChartArea: false,
        },
      },
    },
  };

  return (
    <div style={{ height: '300px' }}>
      <Line data={data} options={options} />
    </div>
  );
};

export default HistoryChart;
