#!/bin/bash
# =============================================================================
# 启动 Go Web 前端网关 (生产模式) 或 Vite Dev Server (开发模式)
# 用法:
#   ./scripts/start-frontend.sh          # 生产模式: Vite build + Go 网关
#   ./scripts/start-frontend.sh --dev    # 开发模式: Vite dev server (HMR)
# =============================================================================
source "$(dirname "$0")/common.sh"

load_env

WEB_PORT="${WEB_PORT:-8080}"
BACKEND_URL="${BACKEND_URL:-http://localhost:1235}"
WEBAPP_DIR="$PROJECT_ROOT/webapp"

export WEB_PORT BACKEND_URL

# ---------- 开发模式 ----------
if [ "$1" = "--dev" ]; then
    print_banner
    log_step "启动 Vite Dev Server (HMR)"

    if ! command -v node &> /dev/null; then
        log_error "未找到 Node.js，请先安装: https://nodejs.org/"
        exit 1
    fi

    cd "$WEBAPP_DIR/frontend"
    if [ ! -d "node_modules" ]; then
        log_step "安装前端依赖..."
        npm install
    fi

    echo "  Vite 地址 : http://localhost:5173"
    echo "  后端代理  : $BACKEND_URL"
    echo "=========================================="

    exec npm run dev
fi

# ---------- 生产模式 ----------
print_banner
log_step "启动 Web 前端网关 (生产模式)"

# 检查 Go
if ! command -v go &> /dev/null; then
    log_error "未找到 Go 编译器，请先安装: https://go.dev/dl/"
    exit 1
fi

# 检查 Node
if ! command -v node &> /dev/null; then
    log_error "未找到 Node.js，请先安装: https://nodejs.org/"
    exit 1
fi

# 清理端口
kill_port "$WEB_PORT"

# Vite build
log_step "Vite 构建前端..."
cd "$WEBAPP_DIR/frontend"
if [ ! -d "node_modules" ]; then
    log_step "安装前端依赖..."
    npm install
fi
npm run build
log_info "前端构建完成 → dist/"

# Go build
log_step "编译 Go 网关..."
cd "$WEBAPP_DIR"
go build -o webapp-gateway .
log_info "编译完成"

echo "  前端地址 : http://localhost:$WEB_PORT"
echo "  后端代理 : $BACKEND_URL"
echo "=========================================="

exec "$WEBAPP_DIR/webapp-gateway"
