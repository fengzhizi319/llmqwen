#!/usr/bin/env python3
"""
AI Code Service - LM Studio 零冗余模型双向共享与软链接工具
将本项目与 ModelScope 的本地模型以符号链接 (Symbolic Link) 方式无缝共享至 LM Studio
实现 0 字节额外磁盘占用，让 LM Studio 与 AI Code Service 共享同一套模型资产
"""

import argparse
import os
import sys
from typing import List, Tuple
from engine.mlx_engine import resolve_local_model_path

LM_STUDIO_MODELS_DIR = os.path.expanduser("~/.cache/lm-studio/models")


def get_models_to_link() -> List[Tuple[str, str, str]]:
    """
    定义待链接到 LM Studio 的模型映射表
    格式: (源模型绝对路径/解析名, LM Studio 发布者分类, LM Studio 模型目录名)
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    models = [
        # 1. 本项目 4bit MLX 原生极速量化版模型
        (
            os.path.join(project_root, "models", "qwen3.8-27b-mlx-4bit"),
            "local",
            "qwen3.8-27b-mlx-4bit",
        ),
        # 2. 本项目 8bit + MTP 综合版模型
        (
            os.path.join(project_root, "models", "qwen3.8-27b-8bit-mtp"),
            "local",
            "qwen3.8-27b-8bit-mtp",
        ),
        # 3. ModelScope 下载的 8bit 独立量化版模型
        (
            "lmstudio-community/Qwen3.8-27B-MLX-8bit",
            "lmstudio-community",
            "Qwen3.8-27B-MLX-8bit",
        ),
        # 4. ModelScope 下载的 27B 原始全精度模型
        (
            "Qwen/Qwen3.8-27B",
            "Qwen",
            "Qwen3.8-27B",
        ),
    ]
    return models


def create_lmstudio_symlinks():
    print("=" * 75)
    print("  🚀 AI Code Service <-> LM Studio 零冗余模型互通工具")
    print(f"  📁 LM Studio 模型根目录: {LM_STUDIO_MODELS_DIR}")
    print("=" * 75)

    if not os.path.exists(LM_STUDIO_MODELS_DIR):
        print(f"📂 创建 LM Studio 模型目录: {LM_STUDIO_MODELS_DIR}")
        os.makedirs(LM_STUDIO_MODELS_DIR, exist_ok=True)

    mappings = get_models_to_link()
    success_count = 0

    for src_ident, publisher, model_name in mappings:
        # 智能解析本地源路径
        src_path = resolve_local_model_path(src_ident)
        if not os.path.exists(src_path):
            print(f"⚠️ 跳过: 本地未找到源模型 '{src_ident}' (解析路径: {src_path})")
            continue

        pub_dir = os.path.join(LM_STUDIO_MODELS_DIR, publisher)
        os.makedirs(pub_dir, exist_ok=True)
        dest_link = os.path.join(pub_dir, model_name)

        if os.path.islink(dest_link):
            # 检查软链接目标是否一致
            current_target = os.readlink(dest_link)
            if os.path.abspath(current_target) == os.path.abspath(src_path):
                print(f"✅ 已存在软链接: [{publisher}/{model_name}] -> {src_path}")
                success_count += 1
                continue
            else:
                os.remove(dest_link)
        elif os.path.exists(dest_link):
            print(f"ℹ️ 目标路径已存在实体目录: {dest_link} (无需创建)")
            success_count += 1
            continue

        try:
            os.symlink(src_path, dest_link)
            print(f"🔗 成功创建软链接: [{publisher}/{model_name}]")
            print(f"   源路径: {src_path}")
            print(f"   目标  : {dest_link}")
            print(f"   💾 额外磁盘占用: 0 KB (macOS APFS 软链接)")
            success_count += 1
        except Exception as e:
            print(f"❌ 创建软链接失败 [{publisher}/{model_name}]: {e}")

    # 扫描 LM Studio 中所有已就绪的模型
    scan_existing_lmstudio_models()

    print("\n" + "=" * 75)
    print(f"🎉 操作完成！已成功将 {success_count} 个模型打通至 LM Studio。")
    print("👉 打开 LM Studio 客户端 -> 在 Local Models 列表中点击刷新即可直接加载！")
    print("=" * 75)


def scan_existing_lmstudio_models():
    """扫描 LM Studio 目录中所有模型并展示在控制台"""
    print("\n🔍 正在扫描 LM Studio 中所有可用模型资产:")
    print("-" * 75)
    found = 0
    if not os.path.exists(LM_STUDIO_MODELS_DIR):
        return

    for root, dirs, files in os.walk(LM_STUDIO_MODELS_DIR, followlinks=True):
        if any(f in files for f in ("config.json", "params.json", "model.safetensors.index.json")):
            rel_path = os.path.relpath(root, LM_STUDIO_MODELS_DIR)
            # 计算是否为软链接
            is_link = os.path.islink(os.path.join(LM_STUDIO_MODELS_DIR, rel_path))
            link_tag = "🔗 [软链接/零占用]" if is_link else "📁 [本地实体]"
            print(f"  • {rel_path:<40} {link_tag}")
            found += 1

    if found == 0:
        print("  (暂未扫描到模型)")


def main():
    parser = argparse.ArgumentParser(description="AI Code Service 与 LM Studio 模型软链接互通工具")
    parser.parse_args()
    create_lmstudio_symlinks()


if __name__ == "__main__":
    main()
