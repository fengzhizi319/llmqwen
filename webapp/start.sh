#!/bin/bash
# =============================================================================
# AI Code Service — Web UI 启动脚本
# 用法:
#   ./start.sh          # 生产模式: Vite build + Go 网关
#   ./start.sh --dev    # 开发模式: Vite dev server (HMR)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载 .env (如果存在)
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a
    . "$SCRIPT_DIR/../.env"
    set +a
fi

# 默认配置
export WEB_PORT="${WEB_PORT:-8080}"
export BACKEND_URL="${BACKEND_URL:-http://localhost:1235}"

# ---------- 开发模式 ----------
if [ "$1" = "--dev" ]; then
    echo "=========================================="
    echo "  🛠️  AI Code Service — Dev Mode (Vite)"
    echo "=========================================="
    echo "  Vite 地址 : http://localhost:5173"
    echo "  后端代理  : ${BACKEND_URL}"
    echo "  (需要 Python 后端运行在 1235 端口)"
    echo "=========================================="

    cd frontend
    if [ ! -d "node_modules" ]; then
        echo "📦 安装前端依赖..."
        npm install
    fi
    exec npm run dev
fi

# ---------- 生产模式 ----------
echo "=========================================="
echo "  🚀 AI Code Service — Web UI"
echo "=========================================="
echo "  前端地址  : http://localhost:${WEB_PORT}"
echo "  后端地址  : ${BACKEND_URL}"
echo "=========================================="

# 检查 Go
if ! command -v go &> /dev/null; then
    echo "❌ 错误: 未找到 Go 编译器，请先安装: https://go.dev/dl/"
    exit 1
fi

# 检查 Node (用于 Vite build)
if ! command -v node &> /dev/null; then
    echo "❌ 错误: 未找到 Node.js，请先安装: https://nodejs.org/"
    exit 1
fi

# 1. Vite build
echo "🔨 Vite 构建前端..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "📦 安装前端依赖..."
    npm install
fi
npm run build
cd ..
echo "✓ 前端构建完成 → dist/"

# 2. Go build
echo "🔨 编译 Go 网关..."
go build -o webapp-gateway .

echo "🚀 启动 Web 网关..."
exec ./webapp-gateway
