#!/bin/bash
# =============================================================================
# 运行真实模型集成测试 (需要本地模型文件)
# 用法: ./scripts/test-real.sh [测试函数名]
# 示例:
#   ./scripts/test-real.sh                          # 运行所有集成测试
#   ./scripts/test-real.sh test_engine_chat         # 运行单个测试
#   ./scripts/test-real.sh -k "stream or thinking"  # 按关键字筛选
# =============================================================================
source "$(dirname "$0")/common.sh"

load_env
activate_conda

PY_BIN=$(get_python)

print_banner
log_step "运行真实模型集成测试"

cd "$PROJECT_ROOT"

export RUN_REAL_MODEL_TESTS=1

echo "  测试文件: tests/test_real_model_integration.py"
echo "  Python:   $($PY_BIN --version 2>&1)"
echo "  模型:     $(grep 'default_model' config.yaml | head -1 | awk '{print $2}' | tr -d '\"')"
echo "=========================================="
echo ""

$PY_BIN -m pytest tests/test_real_model_integration.py -v -s "$@"
