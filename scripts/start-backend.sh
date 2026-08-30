#!/bin/bash
# =============================================================================
# 启动 Python LLM 后端服务
# 用法: ./scripts/start-backend.sh
# =============================================================================
source "$(dirname "$0")/common.sh"

load_env
activate_conda

PY_BIN=$(get_python)
PORT="${PORT:-1235}"

print_banner
log_step "启动 Python LLM 后端"

# 检查配置文件
if [ ! -f "$PROJECT_ROOT/config.yaml" ]; then
    log_error "未找到 config.yaml，请先创建配置文件"
    exit 1
fi

# 清理端口
kill_port "$PORT"

echo "  API 地址 : http://localhost:$PORT"
echo "  API 文档 : http://localhost:$PORT/docs"
echo "  健康检查 : http://localhost:$PORT/health"
echo "=========================================="

cd "$PROJECT_ROOT"
exec $PY_BIN app.py
