#!/bin/bash
cd "$(dirname "$0")"

echo "=========================================="
echo "大米重力去石机群 - 质量工艺分析台"
echo "完整系统启动脚本"
echo "=========================================="

echo ""
echo "[1/3] 启动后端服务 (端口 5001)..."
cd backend && chmod +x start.sh && nohup ./start.sh > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

echo "等待后端服务启动..."
sleep 10

echo ""
echo "[2/3] 检查后端服务状态..."
for i in {1..10}; do
    if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
        echo "✅ 后端服务启动成功 (PID: $BACKEND_PID)"
        break
    fi
    echo "等待中... ($i/10)"
    sleep 2
done

echo ""
echo "[3/3] 启动前端服务 (端口 3001)..."
cd frontend && chmod +x start.sh && nohup ./start.sh > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo ""
echo "=========================================="
echo "系统启动完成！"
echo "=========================================="
echo "后端服务: http://localhost:5001"
echo "前端界面: http://localhost:3001"
echo "后端PID: $BACKEND_PID"
echo "前端PID: $FRONTEND_PID"
echo ""
echo "停止服务命令:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo "查看日志:"
echo "  后端: tail -f backend.log"
echo "  前端: tail -f frontend.log"
echo "=========================================="
