import axios from 'axios';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const getAllMachines = async () => {
  const response = await api.get('/machines');
  return response.data;
};

export const getMachineDetail = async (machineId) => {
  const response = await api.get(`/machines/${machineId}`);
  return response.data;
};

export const getMachineHistory = async (machineId, hours = 24) => {
  const response = await api.get(`/machines/${machineId}/history?hours=${hours}`);
  return response.data;
};

export const getPurityCurve = async (machineId) => {
  const response = await api.get(`/machines/${machineId}/purity-curve`);
  return response.data;
};

export const getOptimization = async (machineId) => {
  const response = await api.get(`/machines/${machineId}/optimize`);
  return response.data;
};

export const getFeatureImportance = async (machineId) => {
  const response = await api.get(`/machines/${machineId}/feature-importance`);
  return response.data;
};

export const predictPurity = async (machineId, parameters) => {
  const response = await api.post('/predict', {
    machine_id: machineId,
    parameters,
  });
  return response.data;
};

export const trainModel = async (hours = 24, machineId = null) => {
  const response = await api.post('/model/train', {
    hours,
    machine_id: machineId,
  });
  return response.data;
};

export const getModelStatus = async () => {
  const response = await api.get('/model/status');
  return response.data;
};

export const refreshData = async () => {
  const response = await api.post('/data/refresh');
  return response.data;
};

export const getDashboardData = async () => {
  const response = await api.get('/dashboard');
  return response.data;
};

export default api;
