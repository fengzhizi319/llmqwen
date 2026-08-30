#!/bin/bash
# =============================================================================
# 运行全部测试 (Mock 单元测试 + 真实模型集成测试)
# 用法: ./scripts/test-all.sh
# =============================================================================
source "$(dirname "$0")/common.sh"

load_env
activate_conda

PY_BIN=$(get_python)

print_banner
log_step "运行全部测试套件"

cd "$PROJECT_ROOT"

export RUN_REAL_MODEL_TESTS=1

echo "  测试范围: 全部测试 (Mock + 真实模型)"
echo "  Python:   $($PY_BIN --version 2>&1)"
echo "=========================================="
echo ""

$PY_BIN -m pytest tests/ -v -s "$@"
