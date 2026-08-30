#!/bin/bash
# =============================================================================
# AI Code Service — 公共函数库
# 所有 scripts/*.sh 共享的基础逻辑，通过 source 引入
# =============================================================================

set -e

# 项目根目录 (scripts/ 的上一级)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="llmqwen"

# ---- 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}✓${NC} $1"; }
log_warn()  { echo -e "${YELLOW}⚠${NC}  $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_step()  { echo -e "${CYAN}▶${NC} ${BOLD}$1${NC}"; }

# ---- 加载 .env ----
load_env() {
    if [ -f "$PROJECT_ROOT/.env" ]; then
        set -a
        . "$PROJECT_ROOT/.env"
        set +a
        log_info "已加载 .env 配置"
    fi
}

# ---- 激活 Conda 环境 ----
activate_conda() {
    if [ "$CONDA_DEFAULT_ENV" = "$CONDA_ENV" ]; then
        return
    fi
    if ! command -v conda &> /dev/null; then
        log_warn "未找到 conda，使用系统 Python"
        return
    fi
    CONDA_BASE=$(conda info --base 2>/dev/null || true)
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        if conda env list | grep -q -E "(^|[ /])${CONDA_ENV}($|[ /])"; then
            conda activate "$CONDA_ENV"
            log_info "已激活 Conda 环境: $CONDA_ENV"
        else
            log_warn "Conda 环境 $CONDA_ENV 不存在，使用当前环境"
        fi
    fi
}

# ---- 获取 Python 路径 ----
get_python() {
    # 优先使用 conda 环境中的 Python
    local conda_python="/opt/homebrew/Caskroom/miniforge/base/envs/${CONDA_ENV}/bin/python3"
    if [ -x "$conda_python" ]; then
        echo "$conda_python"
    elif command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        echo "python"
    else
        log_error "未找到 Python，请先安装: conda env create -f environment.yml"
        exit 1
    fi
}

# ---- 清理端口 ----
kill_port() {
    local port=$1
    local pid
    pid=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        log_info "清理端口 $port 残留进程 (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 1
    fi
}

# ---- 检查服务是否运行 ----
wait_for_service() {
    local url=$1
    local name=$2
    local max_wait=${3:-30}
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s -o /dev/null -w '' "$url" 2>/dev/null; then
            log_info "$name 已就绪 ($url)"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    log_error "$name 启动超时 (${max_wait}s)"
    return 1
}

# ---- 打印 Banner ----
print_banner() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║       🚀 AI Code Service                 ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
    echo ""
}
