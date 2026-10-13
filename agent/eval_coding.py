"""
Code Agent 编程能力评测脚本 (HumanEval-lite)

向本地 LLM 服务发送若干道 Python 函数实现题，提取生成代码并在独立子进程中
运行断言测试，统计 pass@1 与耗时。仅使用标准库。

用法 (需先启动后端):
    python3 agent/eval_coding.py
    python3 agent/eval_coding.py --base-url http://localhost:1235/v1 --model qwen3.8-flash-next-oq4e-mtp-128k
"""

import argparse
import json
import random
import re
import subprocess
import sys
import time
import urllib.request

DEFAULT_BASE_URL = "http://localhost:1235/v1"
DEFAULT_MODEL = "qwen3.8-flash-next-oq4e-mtp-128k"

SYSTEM_PROMPT = (
    "你是一位严谨的 Python 工程师。根据题目要求实现指定函数，"
    "只输出完整的 Python 代码（包含函数定义与必要导入），不要输出任何解释文字或 markdown 围栏之外的内容。"
)

PROBLEMS = [
    {
        "id": "two_sum",
        "signature": "def two_sum(nums: list[int], target: int) -> list[int]:",
        "doc": '返回列表中和为 target 的两个数的下标（恰好一组解，同一元素不可复用）。\n>>> two_sum([2,7,11,15], 9)\n[0, 1]',
        "tests": [
            "assert two_sum([2,7,11,15], 9) == [0,1]",
            "assert two_sum([3,2,4], 6) == [1,2]",
            "r = two_sum([1,5,8,3], 11); assert sorted(r) == [2,3]",
        ],
    },
    {
        "id": "max_subarray",
        "signature": "def max_subarray_sum(nums: list[int]) -> int:",
        "doc": "返回连续子数组的最大和（Kadane 算法，非空数组）。\n>>> max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4])\n6",
        "tests": [
            "assert max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]) == 6",
            "assert max_subarray_sum([1]) == 1",
            "assert max_subarray_sum([-3,-1,-2]) == -1",
        ],
    },
    {
        "id": "reverse_words",
        "signature": "def reverse_words(s: str) -> str:",
        "doc": "反转句子中单词顺序，多余空格压缩为单个空格。\n>>> reverse_words('  the sky  is blue ')\n'blue is sky the'",
        "tests": [
            "assert reverse_words('  the sky  is blue ') == 'blue is sky the'",
            "assert reverse_words('hello') == 'hello'",
            "assert reverse_words('a b') == 'b a'",
        ],
    },
    {
        "id": "binary_search",
        "signature": "def binary_search(arr: list[int], x: int) -> int:",
        "doc": "在升序数组中查找 x，返回下标，不存在返回 -1（迭代实现，不许用 in/index）。\n>>> binary_search([1,3,5,7], 5)\n2",
        "tests": [
            "assert binary_search([1,3,5,7,9], 7) == 3",
            "assert binary_search([1,3,5], 2) == -1",
            "assert binary_search([], 1) == -1",
            "assert binary_search([2], 2) == 0",
        ],
    },
    {
        "id": "is_palindrome",
        "signature": "def is_palindrome(s: str) -> bool:",
        "doc": "忽略大小写与非字母数字字符，判断回文。\n>>> is_palindrome('A man, a plan, a canal: Panama')\nTrue",
        "tests": [
            "assert is_palindrome('A man, a plan, a canal: Panama') is True",
            "assert is_palindrome('race a car') is False",
            "assert is_palindrome('') is True",
        ],
    },
    {
        "id": "group_anagrams",
        "signature": "def group_anagrams(words: list[str]) -> list[list[str]]:",
        "doc": "将互为字母重排的单词分组，组内顺序任意，返回各分组的列表。\n>>> sorted(map(sorted, group_anagrams(['eat','tea','tan','ate','nat','bat'])))\n[['ate','eat','tea'], ['bat'], ['nat','tan']]",
        "tests": [
            "res = sorted([sorted(g) for g in group_anagrams(['eat','tea','tan','ate','nat','bat'])]); assert res == [['ate','eat','tea'], ['bat'], ['nat','tan']]",
            "assert group_anagrams(['a']) == [['a']]",
        ],
    },
    {
        "id": "roman_to_int",
        "signature": "def roman_to_int(s: str) -> int:",
        "doc": "罗马数字转整数，正确处理减法规则（IV=4, IX=9, XL=40）。\n>>> roman_to_int('MCMXCIV')\n1994",
        "tests": [
            "assert roman_to_int('III') == 3",
            "assert roman_to_int('LVIII') == 58",
            "assert roman_to_int('MCMXCIV') == 1994",
        ],
    },
    {
        "id": "merge_intervals",
        "signature": "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:",
        "doc": "合并所有重叠区间（含端点相接），返回升序结果。\n>>> merge_intervals([[1,3],[2,6],[8,10],[15,18]])\n[[1,6],[8,10],[15,18]]",
        "tests": [
            "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
            "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]",
            "assert merge_intervals([[5,5]]) == [[5,5]]",
        ],
    },
    {
        "id": "lru_cache_class",
        "signature": "class LRUCache:",
        "doc": "实现 LRU 缓存：__init__(capacity:int)、get(key)->int（未命中返回 -1）、put(key,value)（超容量淘汰最久未用）。要求 O(1)。\n>>> c = LRUCache(2); c.put(1,1); c.put(2,2); c.get(1); c.put(3,3); c.get(2)\n-1",
        "tests": [
            "c = LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1) == 1; c.put(3,3); assert c.get(2) == -1; c.put(4,4); assert c.get(1) == -1; assert c.get(3) == 3; assert c.get(4) == 4",
        ],
    },
    {
        "id": "valid_parentheses",
        "signature": "def valid_parentheses(s: str) -> bool:",
        "doc": "仅含 ()[]{} 的字符串是否括号匹配有效。\n>>> valid_parentheses('([]{})')\nTrue",
        "tests": [
            "assert valid_parentheses('([]{})') is True",
            "assert valid_parentheses('(]') is False",
            "assert valid_parentheses('') is True",
            "assert valid_parentheses('((') is False",
        ],
    },
]


def build_user_prompt(p: dict) -> str:
    return (
        f"请实现以下 Python 函数/类（含 docstring 描述的功能），只输出代码：\n\n"
        f"{p['signature']}\n"
        f'    """{p["doc"]}"""'
    )


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def call_chat(base_url: str, model: str, prompt: str, timeout: int) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 3000,
        "temperature": 0.2,
        # 随机 seed: 避免命中后端响应缓存，保证计时真实
        "seed": random.randint(1, 2**31 - 1),
        "stream": False,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    elapsed = time.time() - t0
    choice = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage", {})
    return {"text": choice, "elapsed": elapsed, "usage": usage}


def run_tests(code: str, tests: list[str], timeout: int) -> tuple[bool, str]:
    script = code + "\n\n" + "\n".join(tests) + "\nprint('ALL_PASS')\n"
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        ok = r.returncode == 0 and "ALL_PASS" in r.stdout
        return ok, (r.stderr or r.stdout)[-400:]
    except subprocess.TimeoutExpired:
        return False, "timeout"


def main() -> int:
    ap = argparse.ArgumentParser(description="Code Agent 编程能力评测")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json", action="store_true", help="输出 JSON 结果")
    ap.add_argument("--only", nargs="*", default=None, help="只测指定题目 id")
    ap.add_argument("--dump-code", action="store_true", help="打印每题生成的代码")
    args = ap.parse_args()

    problems = [p for p in PROBLEMS if args.only is None or p["id"] in args.only]

    results = []
    passed = 0
    for p in problems:
        try:
            gen = call_chat(args.base_url, args.model, build_user_prompt(p), args.timeout)
            code = extract_code(gen["text"])
            ok, err = run_tests(code, p["tests"], timeout=15)
            completion_tokens = gen["usage"].get("completion_tokens", 0)
            tps = completion_tokens / gen["elapsed"] if gen["elapsed"] > 0 else 0.0
            results.append({"id": p["id"], "pass": ok, "elapsed": round(gen["elapsed"], 2),
                            "completion_tokens": completion_tokens, "tps": round(tps, 1), "err": "" if ok else err})
            passed += int(ok)
            mark = "PASS" if ok else "FAIL"
            if not args.json:
                print(f"[{mark}] {p['id']:22s} {gen['elapsed']:6.1f}s  {completion_tokens:4d} tok  {tps:5.1f} tok/s")
                if args.dump_code:
                    print("  --- generated code ---")
                    for line in code.splitlines():
                        print("  |", line)
                    print("  ----------------------")
                if not ok:
                    print("       err:", err.strip().splitlines()[-1] if err.strip() else "(无输出)")
        except Exception as e:
            results.append({"id": p["id"], "pass": False, "err": str(e)})
            if not args.json:
                print(f"[ERR ] {p['id']}: {e}")

    total = len(PROBLEMS)
    if args.json:
        print(json.dumps({"passed": passed, "total": total, "results": results}, ensure_ascii=False, indent=2))
    else:
        print(f"\n===== pass@1: {passed}/{total} ({passed/total*100:.0f}%) =====")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
