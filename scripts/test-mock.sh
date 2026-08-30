#!/bin/bash
# =============================================================================
# 运行 Mock 单元测试 (不需要真实模型)
# 用法: ./scripts/test-mock.sh
# =============================================================================
source "$(dirname "$0")/common.sh"

load_env
activate_conda

PY_BIN=$(get_python)

print_banner
log_step "运行 Mock 单元测试"

cd "$PROJECT_ROOT"

# 确保不加载真实模型
unset RUN_REAL_MODEL_TESTS

echo "  测试范围: 除 real_model 标记外的所有测试"
echo "  Python: $($PY_BIN --version 2>&1)"
echo "=========================================="
echo ""

$PY_BIN -m pytest tests/ -v --ignore=tests/test_real_model_integration.py "$@"
