#!/bin/bash
cd "$(dirname "$0")"

echo "=========================================="
echo "大米去石机分析系统 - 后端服务启动"
echo "=========================================="

if [ ! -d "venv" ]; then
    echo "创建 Python 虚拟环境..."
    python3 -m venv venv
fi

echo "激活虚拟环境..."
source venv/bin/activate

echo "安装依赖..."
pip install -r requirements.txt

echo "生成历史数据..."
python3 -c "from src.industrial_db import generate_historical_data; generate_historical_data()"

echo "启动 Flask 服务器 (端口 5001)..."
python3 src/app.py
