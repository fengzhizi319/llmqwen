#!/usr/bin/env python3
"""
AI Code Service - 多模型性能基准测试与对比工具 (Benchmarking Suite)
针对配置的不同模型（8bit+MTP、8bit、16bit 等）进行 TTFT 首字延迟、生成 TPS、显存峰值与端到端耗时的全方位基准压测
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional
import httpx

BASE_URL = "http://localhost:1235"
TIMEOUT = 180.0

TEST_SCENARIOS = [
    {
        "name": "⚡ 短文本对话 (Short Prompt)",
        "prompt": "请用一句话解释 Python 字典的底层哈希表原理",
        "max_tokens": 50,
    },
    {
        "name": "🛠️ 代码重构 (Medium Code Refactor)",
        "prompt": """请重构以下 Python 代码并添加异常处理与类型注解：
def process_data(items):
    res = []
    for item in items:
        if item.get('valid') == True:
            res.append(item['val'] * 10)
    return res""",
        "max_tokens": 120,
    },
    {
        "name": "📜 长文本理解 (Long Context Prefill)",
        "prompt": "假设这是一个包含多模块依赖的后端系统：" + ("\n# 模块逻辑定义: 包含数据清洗、校验、持久化与异步通知管道" * 40) + "\n请根据以上上下文总结系统的核心数据流向：",
        "max_tokens": 80,
    },
]


def check_service() -> bool:
    """检查服务是否正常启动"""
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def get_available_models() -> List[str]:
    """获取所有已注册的模型列表"""
    try:
        resp = httpx.get(f"{BASE_URL}/v1/models", timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        pass
    return ["qwen3.8-27b-8bit-mtp"]


def get_server_metrics() -> Dict[str, Any]:
    """获取服务端实时显存与性能指标"""
    try:
        resp = httpx.get(f"{BASE_URL}/metrics", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def benchmark_single_stream(model: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    """执行单次流式请求并精确统计 TTFT、TPS 与耗时"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.2,
    }

    t_start = time.time()
    t_first_token: Optional[float] = None
    generated_tokens_count = 0
    full_text = ""

    with httpx.stream("POST", f"{BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.read().decode('utf-8')}")

        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    if t_first_token is None:
                        t_first_token = time.time()
                    full_text += content
                    generated_tokens_count += 1
            except Exception:
                pass

    t_end = time.time()
    total_time = max(0.001, t_end - t_start)
    ttft_ms = ((t_first_token - t_start) * 1000) if t_first_token else (total_time * 1000)
    decode_time = max(0.001, t_end - (t_first_token or t_start))
    gen_tps = round(generated_tokens_count / decode_time, 2) if generated_tokens_count > 0 else 0.0

    return {
        "total_time_s": round(total_time, 3),
        "ttft_ms": round(ttft_ms, 1),
        "tokens_out": generated_tokens_count,
        "generation_tps": gen_tps,
        "output_sample": full_text[:60].replace("\n", " ") + "..." if len(full_text) > 60 else full_text,
    }


def run_benchmark(models: List[str], rounds: int = 2) -> Dict[str, Any]:
    print("=" * 80)
    print("  🚀 AI Code Service - 多模型性能基准测试矩阵")
    print(f"  📌 待测模型列表: {models}")
    print(f"  📌 每个场景测试轮数: {rounds}")
    print("=" * 80)

    results: Dict[str, Any] = {}

    for model in models:
        print(f"\n👉 [模型压测中] {model}")
        print("-" * 80)
        results[model] = {"scenarios": {}, "summary": {}}

        total_gen_tps_list = []
        total_ttft_list = []

        for sc in TEST_SCENARIOS:
            sc_name = sc["name"]
            print(f"  🧪 场景: {sc_name} (预计输出 {sc['max_tokens']} tokens)")
            scenario_stats = []

            for r in range(rounds):
                print(f"     第 {r+1}/{rounds} 轮...", end="", flush=True)
                try:
                    res = benchmark_single_stream(model, sc["prompt"], sc["max_tokens"])
                    scenario_stats.append(res)
                    print(f" 完成 -> TTFT: {res['ttft_ms']}ms | TPS: {res['generation_tps']} tok/s | 耗时: {res['total_time_s']}s")
                except Exception as e:
                    print(f" ❌ 失败: {e}")

            if scenario_stats:
                avg_ttft = round(sum(s["ttft_ms"] for s in scenario_stats) / len(scenario_stats), 1)
                avg_tps = round(sum(s["generation_tps"] for s in scenario_stats) / len(scenario_stats), 2)
                avg_time = round(sum(s["total_time_s"] for s in scenario_stats) / len(scenario_stats), 2)

                results[model]["scenarios"][sc_name] = {
                    "avg_ttft_ms": avg_ttft,
                    "avg_generation_tps": avg_tps,
                    "avg_total_time_s": avg_time,
                    "sample": scenario_stats[-1]["output_sample"],
                }
                total_ttft_list.append(avg_ttft)
                total_gen_tps_list.append(avg_tps)

        # 记录显存状态与综合指标
        server_metrics = get_server_metrics()
        engine_stats = server_metrics.get("performance", {}).get("engines", {}).get(model, {})
        memory_stats = server_metrics.get("performance", {})

        results[model]["summary"] = {
            "avg_overall_tps": round(sum(total_gen_tps_list) / len(total_gen_tps_list), 2) if total_gen_tps_list else 0.0,
            "avg_overall_ttft_ms": round(sum(total_ttft_list) / len(total_ttft_list), 1) if total_ttft_list else 0.0,
            "active_memory_mb": engine_stats.get("active_memory_mb") or memory_stats.get("active_memory_mb", "N/A"),
            "peak_memory_mb": engine_stats.get("peak_memory_mb") or memory_stats.get("peak_memory_mb", "N/A"),
        }

    return results


def print_comparison_table(results: Dict[str, Any]):
    print("\n" + "=" * 90)
    print(" 📊 多模型综合性能对比结果 (Performance Benchmark Summary)")
    print("=" * 90)
    print(f"{'模型代号 (Model ID)':<26} | {'生成吞吐 (TPS)':<15} | {'首字延迟 (TTFT)':<15} | {'活跃显存 (RAM)':<15} | {'峰值显存 (Peak)':<12}")
    print("-" * 90)

    for model, data in results.items():
        summary = data.get("summary", {})
        tps = f"{summary.get('avg_overall_tps', 0)} tok/s"
        ttft = f"{summary.get('avg_overall_ttft_ms', 0)} ms"
        act_mem = f"{round(float(summary['active_memory_mb'])/1024, 2)} GB" if isinstance(summary.get("active_memory_mb"), (int, float)) else str(summary.get("active_memory_mb"))
        pk_mem = f"{round(float(summary['peak_memory_mb'])/1024, 2)} GB" if isinstance(summary.get("peak_memory_mb"), (int, float)) else str(summary.get("peak_memory_mb"))
        print(f"{model:<26} | {tps:<15} | {ttft:<15} | {act_mem:<15} | {pk_mem:<12}")

    print("=" * 90)


def save_results(results: Dict[str, Any], output_path: str = "benchmark_results.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 详细基准测试数据已导出至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AI Code Service - 多模型性能基准压测工具")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="指定要测试的模型列表 (默认自动读取已注册的所有可用模型)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="每个测试场景的重复轮数 (默认: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="JSON 测试结果保存路径",
    )
    args = parser.parse_args()

    if not check_service():
        print(f"❌ 错误: 无法连接到服务 ({BASE_URL})，请先启动服务 (运行 ./start.sh 或 python app.py)")
        sys.exit(1)

    models_to_test = args.models or get_available_models()
    # 过滤掉仅为辅助层的模型（如单独的 238MB MTP 层不能单独对话）
    models_to_test = [m for m in models_to_test if m != "qwen3.8-27b-mlx"]

    results = run_benchmark(models_to_test, rounds=args.rounds)
    print_comparison_table(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
