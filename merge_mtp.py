#!/usr/bin/env python3
"""
AI Code Service - MTP (Multi-Token Prediction) 与 8-bit 基础模型整合脚本
将 238MB 的 MTP 投机采样辅助头与 28GB 的 Qwen3.8-27B 8-bit 主模型整合为一个完整的综合版模型
在 macOS APFS 文件系统上采用硬链接/镜像整合，瞬间完成且不占用额外磁盘空间。
"""

import argparse
import json
import os
import shutil
import sys
import yaml
from engine.mlx_engine import resolve_local_model_path


def merge_mtp_into_8bit(
    base_model_name: str = "lmstudio-community/Qwen3.8-27B-MLX-8bit",
    mtp_model_name: str = "inferencerlabs/Qwen3.8-27B-MTP-MLX",
    output_dir: str = "./models/qwen3.8-27b-8bit-mtp",
    auto_update_config: bool = True,
):
    print("=" * 65)
    print("  🚀 AI Code Service - MTP + 8bit 模型综合整合工具")
    print("=" * 65)

    base_resolved = resolve_local_model_path(base_model_name)
    mtp_resolved = resolve_local_model_path(mtp_model_name)

    print(f"📌 8-bit 主模型本地路径: {base_resolved}")
    print(f"📌 MTP 辅助层本地路径 : {mtp_resolved}")

    if not os.path.exists(base_resolved):
        print(f"❌ 错误: 未在本地找到 8-bit 主模型 '{base_resolved}'")
        sys.exit(1)
    if not os.path.exists(mtp_resolved):
        print(f"❌ 错误: 未在本地找到 MTP 模型 '{mtp_resolved}'")
        sys.exit(1)

    out_path = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(out_path, exist_ok=True)
    print(f"🎯 综合模型输出目录    : {out_path}")
    print("-" * 65)
    print("⏳ 正在整合模型权重、MTP 辅助层、分词器与配置文件...")

    # 1. 链接/复制主模型全部文件
    for fname in os.listdir(base_resolved):
        src_f = os.path.join(base_resolved, fname)
        dst_f = os.path.join(out_path, fname)
        if os.path.islink(dst_f) or os.path.exists(dst_f):
            os.remove(dst_f)
        try:
            os.link(src_f, dst_f)  # APFS 硬链接（0 额外磁盘占用）
        except Exception:
            shutil.copy2(src_f, dst_f)

    # 2. 拷贝并整合 MTP 权重文件
    mtp_weight_src = os.path.join(mtp_resolved, "model.safetensors")
    mtp_weight_dst = os.path.join(out_path, "mtp.safetensors")
    if os.path.exists(mtp_weight_src):
        if os.path.exists(mtp_weight_dst):
            os.remove(mtp_weight_dst)
        try:
            os.link(mtp_weight_src, mtp_weight_dst)
        except Exception:
            shutil.copy2(mtp_weight_src, mtp_weight_dst)

    # 3. 增强 config.json 整合 MTP 元数据
    config_file = os.path.join(out_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # 注入 MTP 结构参数
        cfg["has_mtp"] = True
        cfg["mtp_block_size"] = 3
        cfg["mtp_weights_path"] = "mtp.safetensors"
        if "text_config" in cfg:
            cfg["text_config"]["mtp_num_hidden_layers"] = 1

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)

    print("\n🎉 MTP + 8bit 综合版模型整合完成！")
    print(f"📁 完整模型路径: {out_path}")
    print("=" * 65)

    if auto_update_config:
        update_config(out_path)


def update_config(model_path: str):
    config_file = "config.yaml"
    if not os.path.exists(config_file):
        return

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model_key = "qwen3.8-27b-8bit-mtp"
    if "models" not in cfg:
        cfg["models"] = {}

    cfg["models"][model_key] = {
        "path": model_path,
        "description": "Qwen3.8-27B MLX 8bit + MTP 完整综合加速版 (支持 256K 上下文)",
        "engine_type": "mlx_lm",
        "context_length": 262144,
    }
    cfg["default_model"] = model_key

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"📝 已自动将综合模型 '{model_key}' 注册到 config.yaml 并设为默认模型！")


def main():
    parser = argparse.ArgumentParser(description="AI Code Service - MTP 与 8bit 综合模型整合工具")
    parser.add_argument("--base", default="lmstudio-community/Qwen3.8-27B-MLX-8bit", help="8bit 基础模型路径/仓库名")
    parser.add_argument("--mtp", default="inferencerlabs/Qwen3.8-27B-MTP-MLX", help="MTP 辅助层路径/仓库名")
    parser.add_argument("--output", default="./models/qwen3.8-27b-8bit-mtp", help="综合模型输出目录")
    parser.add_argument("--no-config-update", action="store_true", help="不自动更新 config.yaml")

    args = parser.parse_args()
    merge_mtp_into_8bit(
        base_model_name=args.base,
        mtp_model_name=args.mtp,
        output_dir=args.output,
        auto_update_config=not args.no_config_update,
    )


if __name__ == "__main__":
    main()
