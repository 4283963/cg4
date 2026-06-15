#!/bin/bash
cd "$(dirname "$0")"

echo "=========================================="
echo "大米去石机分析系统 - 前端服务启动"
echo "=========================================="

echo "检查 Node.js 版本..."
node --version

if [ ! -d "node_modules" ]; then
    echo "安装 npm 依赖..."
    npm install
fi

echo "启动 Webpack 开发服务器 (端口 3001)..."
npm run dev
