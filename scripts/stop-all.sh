#!/bin/bash
# =============================================================================
# 停止所有运行中的服务
# 用法: ./scripts/stop-all.sh
# =============================================================================
source "$(dirname "$0")/common.sh"

load_env

PORT="${PORT:-1235}"
WEB_PORT="${WEB_PORT:-8080}"

print_banner
log_step "停止所有服务"

stopped=0

# 通过 PID 文件停止
for pidfile in /tmp/aicode-backend.pid /tmp/aicode-frontend.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            log_info "已停止进程 PID: $pid ($pidfile)"
            stopped=1
        fi
        rm -f "$pidfile"
    fi
done

# 通过端口停止 (兜底)
for port in "$PORT" "$WEB_PORT"; do
    pid=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        log_info "已停止端口 $port 的进程 (PID: $pid)"
        stopped=1
    fi
done

if [ $stopped -eq 0 ]; then
    log_info "没有正在运行的服务"
else
    log_info "所有服务已停止"
fi
