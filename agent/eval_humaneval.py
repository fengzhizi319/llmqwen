"""
编程能力分层评测: HumanEval (164) + MBPP (974, 按 task_id 区间分三层)

流程: 加载本地数据集 → 调用后端 /chat/completions 生成代码 →
独立子进程运行官方断言 → 按层级统计 pass@1。仅标准库。

用法 (需先启动后端):
    python3 agent/eval_humaneval.py --dataset humaneval
    python3 agent/eval_humaneval.py --dataset mbpp --mbpp-tier-size 100
    python3 agent/eval_humaneval.py --dataset all --mbpp-tier-size 100
    # 中断续跑:
    python3 agent/eval_humaneval.py --dataset humaneval --resume

结果逐题追加写入 agent/eval_results/<dataset>.jsonl，
汇总输出 agent/eval_results/<dataset>_summary.json。
"""

import argparse
import hashlib
import importlib.metadata
import gzip
import json
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "eval_results"

DEFAULT_BASE_URL = "http://localhost:1235/v1"
DEFAULT_MODEL = "qwen3.8-flash-next-oq4e-mtp-128k"

HE_SYSTEM = (
    "你是一位严谨的 Python 工程师。根据题目给定的函数签名与 docstring，"
    "输出完整的函数实现代码（包含函数签名、必要的导入，不要重复 docstring 之外的说明文字），"
    "只输出代码，不要输出任何解释。"
)
MB_SYSTEM = (
    "你是一位严谨的 Python 工程师。根据题目描述实现所需的 Python 函数，"
    "函数名必须与测试中调用的一致。只输出完整代码（含必要导入），"
    "不要输出任何解释文字。"
)

# HumanEval 知识域分类 (关键词按序首个匹配生效)
HE_CATEGORIES = [
    ("字符串", ["string", "str", "paren", "reverse", "palindrom", "anagram", "letters", "words", "smallest_change"]),
    ("数值与数学", ["num", "prime", "fib", "square", "triangular", "gcd", "mod", "power", "digit", "prod", "mean", "median", "mode", "temperature", "monoton"]),
    ("列表与集合", ["list", "array", "intersperse", "sum_squares", "rolling", "concatenate", "unique", "remove", "shuffle", "filter", "extract", "nested", "tuple", "skew", "symmetric", "below_threshold"]),
    ("算法与逻辑", ["search", "sort", "select", "advance", "move", "decode", "encode", "cycle", "parse", "separate", "count", "max", "min", "is", "can", "has", "check", "gen"]),
]


def he_category(task_id: str, entry_point: str) -> str:
    name = entry_point.lower()
    for cat, kws in HE_CATEGORIES:
        if any(k in name for k in kws):
            return cat
    return "其他"


def mbpp_tier(task_id: int) -> str:
    if task_id <= 300:
        return "基础(id1-300)"
    if task_id <= 600:
        return "中级(id301-600)"
    return "综合(id601-974)"


def load_humaneval() -> list[dict]:
    path = DATA_DIR / "HumanEval.jsonl"
    opener = gzip.open if path.suffix == ".gz" else open
    if not path.exists():
        path = DATA_DIR / "HumanEval.jsonl.gz"
        if not path.exists():
            raise SystemExit(f"缺少数据集: {DATA_DIR}/HumanEval.jsonl(.gz)")
    with opener(path, "rt") as f:
        tasks = [json.loads(line) for line in f]
    for t in tasks:
        t["uid"] = t["task_id"].split("/")[-1]
        t["tier"] = he_category(t["task_id"], t["entry_point"])
    return tasks


def load_mbpp(tier_size: int) -> list[dict]:
    path = DATA_DIR / "mbpp.jsonl"
    if not path.exists():
        raise SystemExit(f"缺少数据集: {path}")
    tasks = []
    for line in open(path, encoding="utf-8"):
        m = json.loads(line)
        first = m["test_list"][0] if m["test_list"] else ""
        mt = re.search(r"assert\s+(?:\(\s*)?(\w+)\s*\(", first)
        entry = mt.group(1) if mt else f"task_{m['task_id']}_fn"
        tasks.append({
            "uid": f"mbpp_{m['task_id']}",
            "raw": m,
            "entry_point": entry,
            "tier": mbpp_tier(m["task_id"]),
        })
    if tier_size > 0:
        rng = random.Random(42)
        by_tier: dict[str, list] = {}
        for t in tasks:
            by_tier.setdefault(t["tier"], []).append(t)
        tasks = [x for g in by_tier.values() for x in (rng.sample(g, min(tier_size, len(g))) if len(g) > tier_size else g)]
        tasks.sort(key=lambda t: t["uid"])
    return tasks


def build_prompt(task: dict, dataset: str) -> tuple[str, str]:
    if dataset == "humaneval":
        user = (
            "请实现以下 Python 函数，输出完整代码：\n\n"
            + task["prompt"]
        )
        return HE_SYSTEM, user
    m = task["raw"]
    tests = "\n".join(m["test_list"])
    user = (
        f"{m['text']}\n\n你的实现应通过以下测试:\n{tests}"
    )
    return MB_SYSTEM, user


def call_chat(base_url: str, model: str, system: str, user: str, timeout: int, temperature: float, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # 随机 seed: 避免命中后端响应缓存，保证计时真实
        "seed": random.randint(1, 2**31 - 1),
        "stream": False,
        "enable_thinking": False,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    elapsed = time.perf_counter() - t0
    return {
        "text": data["choices"][0]["message"]["content"] or "",
        "elapsed": elapsed,
        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
        "finish_reason": data["choices"][0].get("finish_reason"),
    }


def strip_fences(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def assemble_humaneval(task: dict, raw: str) -> str:
    """模型应输出完整函数；兜底: 若只给了函数体，则与官方 prompt 拼接。"""
    code = strip_fences(raw)
    header = "\n".join(
        l for l in task["prompt"].splitlines() if re.match(r"^(import |from )", l)
    )
    if re.search(rf"^(\s*)def\s+{re.escape(task['entry_point'])}\s*\(", code, re.M):
        return (header + "\n" if header else "") + code
    # 兜底续写模式: 将正文统一缩进到 prompt 签名之下
    body = code.splitlines()
    indented = "\n".join(l if (not l or l[0].isspace()) else "    " + l for l in body)
    return (header + "\n" if header else "") + task["prompt"] + indented


def assemble_mbpp(task: dict, raw: str) -> str:
    return strip_fences(raw)


def run_test(module_code: str, task: dict, dataset: str, timeout: int) -> tuple[bool, str]:
    if dataset == "humaneval":
        test_code = task["test"] + f"\ncheck(\n    {task['entry_point']}\n)\nprint('ALL_PASS')\n"
    else:
        m = task["raw"]
        test_code = m["test_setup_code"] + "\n" + "\n".join(m["test_list"]) + "\nprint('ALL_PASS')\n"
    script = module_code + "\n" + test_code
    try:
        with tempfile.TemporaryDirectory(prefix="llmqwen-eval-") as workdir:
            r = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=timeout, cwd=workdir)
        ok = r.returncode == 0 and "ALL_PASS" in r.stdout
        err = (r.stderr or r.stdout or "").strip()
        last = err.splitlines()[-1] if err else ""
        return ok, last[:300]
    except subprocess.TimeoutExpired:
        return False, "timeout"


def done_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            latest[record["uid"]] = record
        except Exception:
            continue
    # An infrastructure failure is not a completed model sample. Code/test
    # failures ARE completed samples and must never be resampled for pass@1.
    return {uid for uid, record in latest.items()
            if record.get("status") != "request_error"
            and not record.get("err", "").startswith("request:")}


def evaluate(dataset: str, tasks: list[dict], args) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset}.jsonl"
    done = done_uids(out_path) if args.resume else set()
    if not done:
        out_path.write_text("", encoding="utf-8")

    f = out_path.open("a", encoding="utf-8")
    n = 0
    for task in tasks:
        if task["uid"] in done:
            continue
        system, user = build_prompt(task, dataset)
        rec = {"uid": task["uid"], "tier": task["tier"], "entry_point": task["entry_point"]}
        try:
            for attempt in range(args.request_retries + 1):
                try:
                    gen = call_chat(args.base_url, args.model, system, user, args.timeout, args.temperature, args.max_tokens)
                    break
                except Exception:
                    if attempt == args.request_retries:
                        raise
                    time.sleep(min(2 ** attempt, 10))
            code = assemble_humaneval(task, gen["text"]) if dataset == "humaneval" else assemble_mbpp(task, gen["text"])
            ok, err = run_test(code, task, dataset, args.test_timeout)
            rec.update({
                "pass": ok,
                "elapsed": round(gen["elapsed"], 2),
                "completion_tokens": gen["completion_tokens"],
                "tps": round(gen["completion_tokens"] / gen["elapsed"], 1) if gen["elapsed"] > 0.05 else 0.0,
                "err": "" if ok else err,
                "status": "passed" if ok else "test_timeout" if err == "timeout" else "test_failed",
                "request_attempts": attempt + 1,
                "model_output": gen["text"],
                "code": code,
                "finish_reason": gen["finish_reason"],
                "output_tokens_at_limit": gen["completion_tokens"] >= args.max_tokens,
            })
            if args.dump_code and not ok:
                rec["code"] = code
        except Exception as e:
            rec.update({"pass": False, "status": "request_error", "err": f"request: {e.__class__.__name__}: {e}"})
        rec["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        mark = "PASS" if rec["pass"] else "FAIL"
        print(f"[{mark}] {task['uid']:18s} tier={task['tier']:10s} {rec.get('elapsed', 0):6.1f}s {rec.get('completion_tokens', 0):5d} tok  {rec.get('tps', 0):5.1f} tok/s" + ("" if rec["pass"] else f"  | {rec.get('err', '')[:80]}"), flush=True)
        n += 1
        if rec["status"] == "request_error":
            print("Infrastructure error: stopping so the backend can recover; resume will retry this task.", flush=True)
            break
        if args.limit and n >= args.limit:
            break
    f.close()

    records = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # 去重(同 uid 取最后一次)
    latest: dict[str, dict] = {}
    for r in records:
        latest[r["uid"]] = r
    expected_uids = {task["uid"] for task in tasks}
    records = [r for uid, r in latest.items() if uid in expected_uids]
    infrastructure_errors = sum(r.get("status") == "request_error" or r.get("err", "").startswith("request:") for r in records)
    completed = len(records) - infrastructure_errors

    tiers: dict[str, dict] = {}
    for r in records:
        t = tiers.setdefault(r["tier"], {"total": 0, "pass": 0, "elapsed": 0.0, "tokens": 0})
        t["total"] += 1
        t["pass"] += int(r["pass"])
        t["elapsed"] += r.get("elapsed", 0)
        t["tokens"] += r.get("completion_tokens", 0)
    for t in tiers.values():
        t["pass_rate"] = round(t["pass"] / t["total"], 4)
        t["tps"] = round(t["tokens"] / t["elapsed"], 1) if t["elapsed"] > 0 else 0.0

    total = len(records)
    passed = sum(int(r["pass"]) for r in records)
    elapsed = sum(r.get("elapsed", 0) for r in records)
    tokens = sum(r.get("completion_tokens", 0) for r in records)
    summary = {
        "dataset": dataset,
        "pass_at_1": round(passed / total, 4) if total else 0.0,
        "passed": passed,
        "total": total,
        "expected_total": len(tasks),
        "completed_generation_and_test": completed,
        "infrastructure_errors": infrastructure_errors,
        "missing": len(tasks) - total,
        "complete": completed == len(tasks),
        "test_timeouts": sum(r.get("status") == "test_timeout" for r in records),
        "avg_elapsed_s": round(elapsed / total, 2) if total else 0.0,
        "avg_tps": round(tokens / elapsed, 1) if elapsed else 0.0,
        "by_tier": tiers,
        "failures": [{"uid": r["uid"], "tier": r["tier"], "err": r.get("err", "")[:200]} for r in records if not r["pass"]],
    }
    (output_dir / f"{dataset}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def print_summary(s: dict) -> None:
    print(f"\n===== {s['dataset']}: pass@1 {s['passed']}/{s['total']} = {s['pass_at_1']*100:.1f}% | 平均 {s['avg_tps']} tok/s | 平均耗时 {s['avg_elapsed_s']}s =====")
    for tier, t in sorted(s["by_tier"].items()):
        print(f"  [{tier:12s}] {t['pass']:4d}/{t['total']:<4d} = {t['pass_rate']*100:5.1f}%   {t['tps']:5.1f} tok/s")


def main() -> int:
    ap = argparse.ArgumentParser(description="HumanEval/MBPP 分层评测")
    ap.add_argument("--dataset", choices=["humaneval", "mbpp", "all"], default="all")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mbpp-tier-size", type=int, default=100, help="MBPP 每层随机抽样题数, 0=全量974")
    ap.add_argument("--limit", type=int, default=0, help="每个数据集本次最多新跑题数(调试用), 0=不限")
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定 uid (如 HumanEval/3 的 3 或 mbpp_12)")
    ap.add_argument("--timeout", type=int, default=600, help="单题生成超时(s)")
    ap.add_argument("--test-timeout", type=int, default=15, help="单题测试超时(s)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--resume", action="store_true", help="跳过结果文件中已完成的题目")
    ap.add_argument("--dump-code", action="store_true", help="失败时把生成代码写入结果记录")
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR, help="独立评测目录，避免覆盖历史结果")
    ap.add_argument("--request-retries", type=int, default=2, help="只重试请求失败，不重采样测试失败")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                if key not in {"resume", "dump_code"}}
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["settings"] != settings:
            raise SystemExit("Cannot resume with different evaluation settings")
    else:
        manifest = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "settings": settings, "python": sys.version,
            "datasets_sha256": {name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
                                for name in ["HumanEval.jsonl", "mbpp.jsonl"] if (DATA_DIR / name).exists()},
            "versions": {},
        }
        for name in ["mlx", "mlx-lm", "transformers"]:
            try:
                manifest["versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    datasets = ["humaneval", "mbpp"] if args.dataset == "all" else [args.dataset]
    exit_code = 0
    for ds in datasets:
        if ds == "humaneval":
            tasks = load_humaneval()
        else:
            tasks = load_mbpp(0 if args.only else args.mbpp_tier_size)
        if args.only:
            wanted = {str(o).split("/")[-1] if ds == "humaneval" else str(o) for o in args.only}
            tasks = [t for t in tasks if t["uid"] in wanted or t["uid"].split("_")[-1] in wanted]
        print(f"===== {ds}: {len(tasks)} 题 =====", flush=True)
        summary = evaluate(ds, tasks, args)
        print_summary(summary)
        if not summary["complete"]:
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
