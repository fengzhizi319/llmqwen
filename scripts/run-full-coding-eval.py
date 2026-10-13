#!/usr/bin/env python3
"""Run all 164 HumanEval + 974 MBPP tasks against one isolated real server.

Infrastructure interruptions resume without resampling completed code failures.
Historical results are preserved. Run with the llmqwen Python interpreter.
"""

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def configuration(port):
    from config import load_config
    config = load_config()
    config.use_mock = False
    config.server.host = "127.0.0.1"
    config.server.port = port
    config.server.api_key = None
    config.server.reload = False
    config.models = {config.default_model: config.models[config.default_model]}
    config.performance.enable_cache = False
    config.performance.max_concurrency = 1
    config.performance.metal_cache_limit_mb = 2048
    config.performance.code_agent_mode = True
    config.performance.default_temperature = 0.0
    return config


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=1247)
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = configuration(args.port)
    if args.serve:
        import app as service
        import uvicorn
        service.load_config = lambda path="config.yaml": config
        uvicorn.run(service.create_app(), host="127.0.0.1", port=args.port, log_level="warning")
        return 0

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Do not stop or reuse another service: this supervisor owns its child only.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config.model_dump(exclude={"server"}),
        "response_cache": "disabled", "temperature": 0.0,
        "max_tokens": 2048, "test_timeout_seconds": 15,
        "expected": {"humaneval": 164, "mbpp": 974},
        "backend_restarts": 0,
    }
    diff = subprocess.run(["git", "diff"], cwd=ROOT, capture_output=True).stdout
    metadata["tracked_worktree_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    source_files = ["engine/qwen4_exp_engine.py", "engine/prefix_cache.py", "agent/eval_humaneval.py"]
    metadata["source_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_files}
    backend = runner = None
    base_url = f"http://127.0.0.1:{args.port}"
    backend_log = (output / "backend.log").open("a", encoding="utf-8")
    try:
        for recovery in range(6):
            metadata["backend_restarts"] = recovery
            backend = subprocess.Popen(
                [sys.executable, "-u", __file__, "--serve", "--port", str(args.port), "--output-dir", str(output)],
                cwd=ROOT, stdout=backend_log, stderr=subprocess.STDOUT,
            )
            ready = False
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline and backend.poll() is None:
                try:
                    with urllib.request.urlopen(base_url + "/metrics", timeout=5) as response:
                        metrics = json.load(response)
                    stats = metrics["performance"]["engine_stats"].get(config.default_model, {})
                    ready = stats.get("loaded") is True and stats.get("engine_type") == "qwen4_exp"
                    if ready:
                        metadata["model_path"] = stats["resolved_path"]
                        metadata["initial_memory"] = metrics["performance"]["metal_memory"]
                        model_path = Path(stats["resolved_path"])
                        metadata["model_metadata_sha256"] = {
                            path: hashlib.sha256((model_path / path).read_bytes()).hexdigest()
                            for path in ["config.json", "model.safetensors.index.json", "omlx_support/qwen4_exp.py"]
                            if (model_path / path).exists()
                        }
                        break
                except Exception:
                    pass
                time.sleep(1)
            if not ready:
                stop(backend)
                print(f"Backend not ready; recovery {recovery + 1}/6", flush=True)
                continue

            (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            command = [sys.executable, "-u", str(ROOT / "agent/eval_humaneval.py"),
                       "--dataset", "all", "--mbpp-tier-size", "0", "--temperature", "0",
                       "--max-tokens", "2048", "--base-url", base_url + "/v1",
                       "--model", config.default_model, "--output-dir", str(output),
                       "--timeout", "300", "--request-retries", "1"]
            if (output / "manifest.json").exists():
                command.append("--resume")
            runner = subprocess.Popen(command, cwd=ROOT)
            code = runner.wait()
            try:
                with urllib.request.urlopen(base_url + "/metrics", timeout=5) as response:
                    metadata["final_metrics"] = json.load(response)
            except Exception:
                pass
            stop(backend)
            if code == 0:
                summaries = {name: json.loads((output / f"{name}_summary.json").read_text())
                             for name in ["humaneval", "mbpp"]}
                if all(s["complete"] and s["expected_total"] == metadata["expected"][name]
                       for name, s in summaries.items()):
                    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
                    metadata["status"] = "complete"
                    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    print("All 1138 tasks completed generation and testing.", flush=True)
                    return 0
            print(f"Evaluation interrupted (exit {code}); resuming after backend restart.", flush=True)
        raise RuntimeError("Evaluation remains incomplete after six backend recovery attempts")
    finally:
        stop(runner)
        stop(backend)
        backend_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
