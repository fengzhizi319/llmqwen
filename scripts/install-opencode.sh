#!/bin/bash
# =============================================================================
# 下载并安装 opencode (AI code agent CLI)
# 用法: ./scripts/install-opencode.sh
# 安装优先级: 已安装跳过 -> npm -> 官方脚本 -> Homebrew
# =============================================================================
source "$(dirname "$0")/common.sh"

print_banner
log_step "安装 opencode code agent"

export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"

if command -v opencode &> /dev/null; then
    log_info "opencode 已安装: $(command -v opencode)"
    log_info "版本: $(opencode --version 2>/dev/null || echo unknown)"
    exit 0
fi

install_via_npm() {
    command -v npm &> /dev/null || return 1
    log_step "尝试通过 npm 安装 opencode-ai..."
    npm install -g opencode-ai --registry="${NPM_REGISTRY:-https://registry.npmjs.org}" && return 0
    log_warn "默认源失败，尝试 npmmirror 镜像源..."
    npm install -g opencode-ai --registry=https://registry.npmmirror.com && return 0
    return 1
}

install_via_official() {
    command -v curl &> /dev/null || return 1
    log_step "尝试通过官方脚本安装 (https://opencode.ai/install)..."
    curl -fsSL https://opencode.ai/install | bash && return 0
    return 1
}

install_via_brew() {
    command -v brew &> /dev/null || return 1
    log_step "尝试通过 Homebrew 安装..."
    brew install anomalyco/tap/opencode && return 0
    return 1
}

if install_via_npm || install_via_official || install_via_brew; then
    hash -r
    if command -v opencode &> /dev/null; then
        log_info "opencode 安装成功: $(command -v opencode)"
        log_info "版本: $(opencode --version 2>/dev/null || echo unknown)"
        exit 0
    fi
fi

log_error "opencode 安装失败，请检查网络后手动执行: npm i -g opencode-ai"
exit 1
