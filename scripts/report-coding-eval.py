#!/usr/bin/env python3
"""Audit a complete local evaluation and write a reproducible capability report."""

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return 0
    position = (len(values) - 1) * fraction
    left, right = math.floor(position), math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.resolve()
    metadata = json.loads((results / "run_metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise SystemExit("Run is not marked complete; refusing to publish partial scores")
    expected = {
        "humaneval": {str(json.loads(row)["task_id"].split("/")[-1])
                      for row in (ROOT / "agent/data/HumanEval.jsonl").read_text().splitlines()},
        "mbpp": {f"mbpp_{json.loads(row)['task_id']}"
                 for row in (ROOT / "agent/data/mbpp.jsonl").read_text().splitlines()},
    }
    summaries, records = {}, {}
    for dataset, uids in expected.items():
        latest = {row["uid"]: row for row in
                  (json.loads(line) for line in (results / f"{dataset}.jsonl").read_text().splitlines())}
        assert set(latest) == uids, f"{dataset}: task coverage differs from dataset"
        assert all(row.get("status") in {"passed", "test_failed", "test_timeout"}
                   and "code" in row and "model_output" in row for row in latest.values()), dataset
        summary = json.loads((results / f"{dataset}_summary.json").read_text())
        assert summary["complete"] and summary["infrastructure_errors"] == summary["missing"] == 0
        assert summary["completed_generation_and_test"] == len(uids)
        assert summary["passed"] == sum(bool(row["pass"]) for row in latest.values())
        summaries[dataset], records[dataset] = summary, list(latest.values())

    config = metadata["configuration"]
    performance = config["performance"]
    text = [
        "# 全量编程能力评测报告", "",
        f"运行开始：{metadata['started_utc']}；结束：{metadata['completed_utc']}。", "",
        "## 1. 结论与覆盖", "",
        "本轮实际完成 HumanEval 164 题和 MBPP 974 题，共 1,138 题的生成与测试。"
        "报告生成前逐一核对任务 ID、原始输出、组装代码、测试终态与汇总，无遗漏或未完成的请求错误。", "",
        "| 数据集 | 完成/预期 | 一次通过 | 一次通过率 | 基础设施未完成 | 测试超时 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, summary in summaries.items():
        text.append(f"| {dataset} | {summary['completed_generation_and_test']}/{summary['expected_total']} | "
                    f"{summary['passed']}/{summary['total']} | {summary['passed']/summary['total']:.2%} | "
                    f"{summary['infrastructure_errors']} | {summary['test_timeouts']} |")
    text += ["", "这是本项目协议下的单次样本通过率。MBPP 的公开测试包含在模型输入中；"
             "HumanEval 使用本地原始测试，没有 HumanEval+/MBPP+ 隐藏增强测试。"
             "因此不能把分数直接当作官方排行榜成绩或跨文件工程任务成功率。", "",
             "## 2. 模型与评测配置", "",
             f"- 模型：`{config['default_model']}`；真实模型路径：`{metadata['model_path']}`。",
             f"- temperature=0，输出上限 {metadata['max_tokens']} tokens，thinking 关闭，完整响应缓存禁用。",
             f"- MTP={performance['enable_mtp']}，深度={performance['mtp_depth']}；"
             f"前缀缓存={performance['enable_prompt_cache']}，最多 {performance['prompt_cache_max_entries']} 条、"
             f"每条 {performance['prompt_cache_max_tokens']} tokens。",
             f"- 串行生成；独立本地评测后端，Metal 空闲内存池上限 {performance['metal_cache_limit_mb']} MiB。",
             f"- 测试子进程超时 {metadata['test_timeout_seconds']} 秒，临时工作目录、Python `-I`；不是 OS 安全沙箱。",
             f"- 后端恢复重启次数：{metadata['backend_restarts']}。请求失败可以恢复，已生成代码的测试失败不重新采样。", "",
             "详细数据集/代码/模型元数据哈希与依赖版本见结果目录中的 manifest 和 run_metadata。", "",
             "## 3. 分类结果", ""]
    for dataset, summary in summaries.items():
        text += [f"### {dataset}", "", "| 本地分类 | 通过/完成 | 通过率 |", "| --- | ---: | ---: |"]
        for tier, values in sorted(summary["by_tier"].items()):
            text.append(f"| {tier} | {values['pass']}/{values['total']} | {values['pass']/values['total']:.2%} |")
        text.append("")
    text += ["HumanEval 按函数名关键词分类；MBPP 按任务 ID 区间分组，不是经过标定的官方难度层级。", "",
             "## 4. 耗时与输出", "", "| 数据集 | 平均请求耗时 | p50 | p95 | 总生成 tokens | 请求口径 TPS |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for dataset, rows in records.items():
        durations = [row["elapsed"] for row in rows]
        seconds = sum(durations)
        tokens = sum(row["completion_tokens"] for row in rows)
        text.append(f"| {dataset} | {seconds/len(rows):.2f} s | {percentile(durations,.5):.2f} s | "
                    f"{percentile(durations,.95):.2f} s | {tokens} | {tokens/seconds:.2f} tok/s |")
    text += ["", "耗时为 HTTP 非流式生成请求耗时，包含预填充与生成；首次 kernel warmup 未剔除。"
             "TPS 是总输出 token / 总请求时间，不是纯解码吞吐。测试时间没有逐题单独记录，"
             "整个运行的墙钟时间还包含加载、测试与调度。", "", "## 5. 失败样本", ""]
    for dataset, rows in records.items():
        failures = [row for row in rows if not row["pass"]]
        categories = Counter(row["err"].split(":",1)[0] for row in failures)
        text += [f"### {dataset}：{len(failures)} 个失败", "", f"错误分类：`{dict(categories)}`。", ""]
        if failures:
            text += ["| 题目 ID | 状态 | 测试错误 |", "| --- | --- | --- |"]
            for row in failures:
                error = row["err"].replace("|", "\\|").replace("\n", " ")
                text.append(f"| {row['uid']} | {row['status']} | {error} |")
        text.append("")
    review_path = results / "failure_review.json"
    if review_path.exists():
        reviews = json.loads(review_path.read_text())
        text += ["### 已人工复核的失败原因", "", "以下解释来自原始生成代码与原题测试，未修改答案或重新采样。", "",
                 "| 题目 | 诊断 |", "| --- | --- |"]
        for uid, review in reviews.items():
            diagnosis = review["diagnosis"].replace("|", "\\|").replace("\n", " ")
            text.append(f"| {uid} | {diagnosis} |")
        text += ["", "部分 MBPP 题文存在歧义，或测试对集合输出顺序有固定约定；仍按原断言计分。"
                 "这类失败与明确的越界、参数错位、算法条件遗漏应分开解读。", ""]
    text += ["错误分类只描述测试终态，根因应同时查看逐题的 raw 输出、code 和数据集测试。"
             "报告不通过重试失败答案提高成绩。", "", "## 6. 编程能力判断与限制", "",
             "- 函数级能力：以完整一次通过率和失败类型为依据，判断 Python 算法、字符串、数值和集合类任务的可靠程度。",
             "- 服务可用性：全量覆盖与未完成请求数独立于代码正确率；历史连接失败造成的低分不混入本轮。",
             "- 工具能力：此前真实集成测试验证了工具调用和结果续接，本轮函数评测不执行 OpenCode 工具循环。",
             "- 工程能力：未评测真实多文件 Bug 修复、依赖迁移或大型架构重构，不能将本轮分数解释为独立完成仓库任务的概率。",
             "- 数据限制：公开题目可能存在训练数据污染；提供 MBPP 测试、代码组装、固定温度与原始测试集都会影响可比性。", "",
             "建议用于有明确输入输出与可执行测试的开发任务；复杂修改应结合仓库任务验收与人工审查。", "",
             "## 7. 可复现记录", ""]
    relative = Path(os.path.relpath(results, args.output.resolve().parent))
    for filename in ["manifest.json", "run_metadata.json", "humaneval_summary.json", "mbpp_summary.json", "humaneval.jsonl", "mbpp.jsonl", "failure_review.json"]:
        if not (results / filename).exists():
            continue
        text.append(f"- [{filename}]({relative.as_posix()}/{filename})")
    text += ["", "```bash", "conda run -n llmqwen python -u scripts/run-full-coding-eval.py \\",
             "  --output-dir agent/eval_results/full-new-run --port 1247",
             "conda run -n llmqwen python scripts/report-coding-eval.py \\",
             "  --results agent/eval_results/full-new-run --output docs/CODING_EVAL_REPORT_LOCAL.md", "```", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({name: {"passed": summary["passed"], "total": summary["total"], "complete": summary["complete"]}
                      for name, summary in summaries.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
