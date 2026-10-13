#!/bin/bash
# =============================================================================
# 一键启动 opencode Code Agent 服务
# - 自动安装 opencode (若缺失)
# - 自动拉起 Python LLM 后端 (若未运行)，使用模型 qwen3.8-flash-next-oq4e-mtp-128k
# - 生成 agent/opencode.json 并以 `opencode serve` 对外提供 HTTP code agent 服务
# 用法: ./scripts/start-agent.sh
# =============================================================================
source "$(dirname "$0")/common.sh"

load_env
activate_conda

PY_BIN=$(get_python)
PORT="${PORT:-1235}"
BACKEND_URL="${BACKEND_URL:-http://localhost:$PORT}"
MODEL="${DEFAULT_MODEL:-qwen3.8-flash-next-oq4e-mtp-128k}"
OPENCODE_HOST="${OPENCODE_HOST:-0.0.0.0}"
OPENCODE_PORT="${OPENCODE_PORT:-4096}"
export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"

print_banner
log_step "启动 opencode Code Agent 服务"

# 1. 确保 opencode 已安装
if ! command -v opencode &> /dev/null; then
    log_warn "未检测到 opencode，开始自动安装..."
    "$PROJECT_ROOT/scripts/install-opencode.sh" || exit 1
fi
log_info "opencode: $(command -v opencode) ($(opencode --version 2>/dev/null || echo unknown))"

# 2. 确保 Python LLM 后端就绪
BACKEND_STARTED_BY_US=false
if curl -s -o /dev/null "$BACKEND_URL/health" 2>/dev/null; then
    log_info "LLM 后端已在运行: $BACKEND_URL"
else
    log_step "启动 Python LLM 后端 (端口 $PORT)..."
    kill_port "$PORT"
    cd "$PROJECT_ROOT"
    $PY_BIN app.py > /tmp/aicode-backend.log 2>&1 &
    BACKEND_PID=$!
    echo "$BACKEND_PID" > /tmp/aicode-backend.pid
    BACKEND_STARTED_BY_US=true
    wait_for_service "$BACKEND_URL/health" "Python 后端" 120 || {
        log_error "后端启动失败，查看日志: cat /tmp/aicode-backend.log"
        kill "$BACKEND_PID" 2>/dev/null || true
        exit 1
    }
fi

# 3. 生成 opencode 配置 (注入后端地址与模型名)
log_step "生成 opencode 配置: agent/opencode.json"
cd "$PROJECT_ROOT"
mkdir -p agent
sed -e "s|__BASE_URL__|${BACKEND_URL}/v1|g" \
    -e "s|__MODEL__|${MODEL}|g" \
    agent/opencode.json.template > agent/opencode.json
log_info "LLM 提供方: ${BACKEND_URL}/v1  模型: ${MODEL}"

# 4. 启动 opencode serve (对外提供 code agent HTTP 服务)
cleanup() {
    echo ""
    log_step "正在停止服务..."
    kill "$OPENCODE_PID" 2>/dev/null || true
    if [ "$BACKEND_STARTED_BY_US" = true ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
        rm -f /tmp/aicode-backend.pid
    fi
    log_info "已停止"
    exit 0
}
trap cleanup SIGINT SIGTERM

cd "$PROJECT_ROOT/agent"

# 服务安全: 监听非回环地址时强制启用 Basic Auth 密码 (可用 OPENCODE_SERVER_PASSWORD 覆盖)
if [ -z "${OPENCODE_SERVER_PASSWORD:-}" ]; then
    if [ ! -f "$PROJECT_ROOT/agent/.opencode-password" ]; then
        LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 24 > "$PROJECT_ROOT/agent/.opencode-password"
    fi
    export OPENCODE_SERVER_PASSWORD="$(cat "$PROJECT_ROOT/agent/.opencode-password")"
    if [ "$OPENCODE_HOST" != "127.0.0.1" ] && [ "$OPENCODE_HOST" != "localhost" ]; then
        log_warn "服务监听 $OPENCODE_HOST (对外暴露)，已启用密码保护，密码存于 agent/.opencode-password"
    fi
fi

log_step "启动 opencode serve (端口 $OPENCODE_PORT)..."
opencode serve --hostname "$OPENCODE_HOST" --port "$OPENCODE_PORT" > /tmp/aicode-agent.log 2>&1 &
OPENCODE_PID=$!
echo "$OPENCODE_PID" > /tmp/aicode-agent.pid

wait_for_service "http://localhost:$OPENCODE_PORT/global/health" "opencode agent" 30 || {
    log_error "opencode serve 启动失败，查看日志: cat /tmp/aicode-agent.log"
    cleanup
}

echo ""
echo "=========================================="
echo -e "  ${GREEN}opencode Code Agent 已就绪${NC}"
echo "=========================================="
echo "  🤖 Code Agent API : http://localhost:$OPENCODE_PORT"
echo "  🧠 LLM 后端       : $BACKEND_URL  (模型: $MODEL)"
echo "  ⚙️  Agent 配置     : $PROJECT_ROOT/agent/opencode.json"
echo "  📜 交互 CLI       : cd agent && opencode"
echo "------------------------------------------"
echo "  日志: tail -f /tmp/aicode-agent.log"
echo "  停止: Ctrl+C"
echo "=========================================="

wait
