#!/bin/bash
# =============================================================================
# 一键启动全部服务 (Python 后端 + Go 前端网关)
# 用法: ./scripts/start-all.sh
# 停止: Ctrl+C 或 ./scripts/stop-all.sh
# =============================================================================
source "$(dirname "$0")/common.sh"

load_env
activate_conda

PY_BIN=$(get_python)
PORT="${PORT:-1235}"
WEB_PORT="${WEB_PORT:-8080}"
BACKEND_URL="${BACKEND_URL:-http://localhost:1235}"

print_banner
log_step "一键启动全部服务"

# 清理端口
kill_port "$PORT"
kill_port "$WEB_PORT"

# 1. 后台启动 Python 后端
log_step "启动 Python LLM 后端 (端口 $PORT)..."
cd "$PROJECT_ROOT"
$PY_BIN app.py > /tmp/aicode-backend.log 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > /tmp/aicode-backend.pid
log_info "后端 PID: $BACKEND_PID"

# 等待后端就绪
wait_for_service "http://localhost:$PORT/health" "Python 后端" 60 || {
    log_error "后端启动失败，查看日志: cat /tmp/aicode-backend.log"
    kill "$BACKEND_PID" 2>/dev/null || true
    exit 1
}

# 2. 后台启动 Go 前端网关
log_step "启动 Web 前端网关 (端口 $WEB_PORT)..."
cd "$PROJECT_ROOT/webapp"
if ! [ -f webapp-gateway ]; then
    log_step "编译 Go 网关..."
    go build -o webapp-gateway .
fi
export WEB_PORT BACKEND_URL
./webapp-gateway > /tmp/aicode-frontend.log 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > /tmp/aicode-frontend.pid
log_info "前端 PID: $FRONTEND_PID"

wait_for_service "http://localhost:$WEB_PORT/" "Web 前端" 10 || true

echo ""
echo "=========================================="
echo -e "  ${GREEN}全部服务已启动${NC}"
echo "=========================================="
echo "  🐍 Python 后端 : http://localhost:$PORT      (API + 推理)"
echo "  🌐 Web 前端    : http://localhost:$WEB_PORT   (聊天界面)"
echo "  📖 API 文档    : http://localhost:$PORT/docs"
echo "  💊 健康检查    : http://localhost:$PORT/health"
echo "------------------------------------------"
echo "  日志: tail -f /tmp/aicode-backend.log"
echo "  停止: ./scripts/stop-all.sh 或 Ctrl+C"
echo "=========================================="

# 捕获退出信号，清理子进程
cleanup() {
    echo ""
    log_step "正在停止所有服务..."
    kill "$BACKEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    rm -f /tmp/aicode-backend.pid /tmp/aicode-frontend.pid
    log_info "所有服务已停止"
    exit 0
}
trap cleanup SIGINT SIGTERM

# 前台等待
wait
