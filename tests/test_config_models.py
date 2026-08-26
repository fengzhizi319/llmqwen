"""
config.yaml 中模型注册的本地单元测试。

这些测试只验证配置和本地模型目录，不加载大型模型权重，也不访问网络。
添加新模型时只需修改 config.yaml，这些测试会自动适配。
"""

import json
from pathlib import Path

import pytest

from config import ModelSpec, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
VALID_ENGINE_TYPES = {"auto", "mlx_lm", "mlx_vlm", "mock"}


@pytest.fixture(scope="module")
def project_config():
    """加载项目实际配置，而不是测试用的 Mock 配置。"""
    return load_config(str(CONFIG_PATH))


def test_default_model_is_registered(project_config):
    """默认模型必须能在 models 配置中找到。"""
    assert project_config.default_model in project_config.models


def test_all_configured_models_have_valid_specs(project_config):
    """每个模型都应具备有效路径、引擎类型和上下文长度。"""
    assert project_config.models

    for model_name, spec in project_config.models.items():
        assert isinstance(spec, ModelSpec), model_name
        assert spec.path.strip(), model_name
        assert spec.engine_type in VALID_ENGINE_TYPES, model_name
        assert spec.context_length > 0, model_name


def test_local_model_directories_are_available(project_config):
    """所有 ./models 下的本地模型都必须包含模型配置文件。"""
    local_models = {
        name: spec
        for name, spec in project_config.models.items()
        if spec.path.startswith("./models/")
    }

    assert local_models
    for model_name, spec in local_models.items():
        model_dir = PROJECT_ROOT / spec.path.removeprefix("./")
        assert model_dir.is_dir(), f"{model_name}: missing {model_dir}"
        assert (model_dir / "config.json").is_file(), (
            f"{model_name}: missing config.json"
        )


def test_default_model_has_valid_local_directory(project_config):
    """默认模型必须拥有有效的本地目录、引擎类型与模型配置。"""
    default_name = project_config.default_model
    spec = project_config.models[default_name]
    model_dir = PROJECT_ROOT / spec.path.removeprefix("./")

    # 基本属性校验
    assert spec.engine_type in VALID_ENGINE_TYPES
    assert spec.context_length > 0
    assert spec.path.strip()

    # 本地目录校验（仅对本地路径模型）
    if spec.path.startswith("./models/"):
        assert model_dir.is_dir(), f"默认模型目录不存在: {model_dir}"
        config_json_path = model_dir / "config.json"
        assert config_json_path.is_file(), f"默认模型缺少 config.json: {config_json_path}"

        with config_json_path.open(encoding="utf-8") as config_file:
            model_config = json.load(config_file)

        # 验证模型配置文件包含必要字段
        assert "model_type" in model_config
        assert "architectures" in model_config

