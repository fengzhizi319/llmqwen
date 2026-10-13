#!/usr/bin/env python3
"""Load real Qwen4Exp weights and verify cache/MTP against greedy decoding.

Run with the project's MLX-compatible interpreter. No response cache or mock
engine is involved. Each comparison uses the same loaded weights.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from engine import ModelManager
from engine.qwen4_exp_engine import Qwen4ExpEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--filler-lines", type=int, default=48)
    args = parser.parse_args()
    config = load_config()
    config.use_mock = False
    manager = ModelManager(config)
    engine = manager.get_engine()
    if not isinstance(engine, Qwen4ExpEngine):
        raise RuntimeError("This benchmark requires a real Qwen4ExpEngine")
    started = time.perf_counter()
    engine.load_model()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": engine.model_name,
        "model_path": engine.resolved_path,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ["mlx", "mlx-lm", "transformers"]},
        "load_seconds": time.perf_counter() - started,
        "max_output_tokens": args.max_tokens,
        "prefill_step_size": engine.prefill_step_size,
        "cache_limits": engine._prefix_cache.stats(),
        "runs": [],
    }

    def prompt(messages, system):
        return manager.build_chat_prompt(
            [{"role": "system", "content": system}] + messages,
            enable_thinking=False,
        )

    system = "You are a concise Python coding assistant. Follow the user's instruction.\n"
    system += "".join(f"# reference_{i}: preserve existing public interfaces and validate inputs.\n" for i in range(args.filler_lines))
    user = {"role": "user", "content": "Write a Python function add(a, b). Only output code."}
    first = prompt([user], system)

    def run(name, text, cache, mtp=False):
        engine.enable_prompt_cache = cache
        engine.enable_mtp = mtp
        started = time.perf_counter()
        first_text = None
        chunks = []
        for chunk in engine.stream_generate(text, max_tokens=args.max_tokens, temperature=0.0):
            if chunk and first_text is None:
                first_text = time.perf_counter() - started
            chunks.append(chunk)
        output = "".join(chunks)
        row = {
            "name": name, "cache_enabled": cache, "mtp_enabled": mtp,
            "prompt_tokens": engine.count_tokens(text),
            "output": output,
            "completion_tokens": engine.count_tokens(output),
            "ttft_seconds": first_text,
            "total_seconds": time.perf_counter() - started,
            "reused_tokens": engine._last_reused_tokens,
            "prefill_tokens": engine._last_prefill_tokens,
            "prefill_seconds": engine._last_prefill_seconds,
            "cache": engine._prefix_cache.stats(),
            "memory": engine.get_memory_stats(),
            "mtp_drafted_tokens": engine._mtp_drafted_tokens,
            "mtp_accepted_tokens": engine._mtp_accepted_tokens,
        }
        report["runs"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return output

    try:
        # Initialize kernels before recording the cache comparisons.
        run("kernel_warmup", prompt([user], "You are a Python coding assistant."), False)
        engine._prefix_cache.clear()
        baseline = run("repeat_baseline", first, False)
        cold = run("repeat_cache_cold", first, True)
        warm = run("repeat_cache_warm", first, True)
        assert baseline == cold == warm, "Repeated request output changed with caching"
        assert engine._last_reused_tokens > 0, "Repeated request did not reuse state"

        branch = prompt([{**user, "content": "Write a Python function subtract(a, b). Only output code."}], system)
        branch_baseline = run("branch_baseline", branch, False)
        branch_cached = run("branch_cache", branch, True)
        assert branch_baseline == branch_cached, "Branched request output changed"
        assert engine._last_reused_tokens > 0, "Shared system prefix was not reused"

        extended = prompt([user, {"role": "assistant", "content": baseline},
                           {"role": "user", "content": "Now write multiply(a, b). Only output code."}], system)
        extended_baseline = run("multiturn_baseline", extended, False)
        extended_cached = run("multiturn_cache", extended, True)
        assert extended_baseline == extended_cached, "Multi-turn output changed"

        short = prompt([user], "You are a concise coding assistant.")
        short_baseline = run("short_baseline", short, False)
        short_cached = run("short_cache", short, True)
        assert short_baseline == short_cached, "Shortened/changed-system output changed"
        assert engine._last_reused_tokens == 0, "Unrelated system prompt reused state"

        # Cache construction happens before the first output; aborting decoding
        # must not mutate the stored prefill checkpoint.
        stream = engine.stream_generate(first, max_tokens=args.max_tokens, temperature=0)
        next(stream)
        stream.close()
        after_abort = run("after_abort_cache", first, True)
        assert after_abort == baseline, "Cancelled generation polluted prefix state"

        mtp_cold = run("mtp_cached_prefix_first", first, True, True)
        mtp_warm = run("mtp_cached_prefix_repeat", first, True, True)
        assert mtp_cold == mtp_warm == baseline, "MTP differs from authoritative greedy output"
        assert engine._last_reused_tokens > 0
        report["validation"] = "passed"
    except BaseException as exc:
        report["validation"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        report["final_stats"] = engine.get_stats()
        engine.unload_model()
        report["post_unload_cache"] = engine._prefix_cache.stats()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
