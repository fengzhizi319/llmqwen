"""
AI Code Service - OpenAI tools 协议与模型原生 XML 工具调用格式双向转换

模型原生格式与 chat_template.jinja 约定一致，标签名通过字符串拼接构造，
避免在本文件中出现保留控制词被其它处理流误解析:
  TC_OPEN / TC_CLOSE   单次工具调用包裹标签
  FN_OPEN(name)        函数名开始标签
  FN_CLOSE             函数名结束标签
  PARAM_OPEN(key)      参数开始标签
  PARAM_CLOSE          参数结束标签
  TOOL_RESULT_OPEN     工具执行结果注入 user 消息的开始标签
  TOOL_RESULT_CLOSE    工具执行结果注入的结束标签
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

_LT = "<"
_GT = ">"

TC_OPEN = _LT + "tool_call" + _GT
TC_CLOSE = _LT + "/tool_call" + _GT
FN_CLOSE = _LT + "/function" + _GT
PARAM_CLOSE = _LT + "/parameter" + _GT
TOOL_RESULT_OPEN = _LT + "tool_r" + "esult" + _GT
TOOL_RESULT_CLOSE = _LT + "/tool_r" + "esult" + _GT
TOOL_RESPONSE_OPEN = _LT + "tool_response" + _GT
TOOL_RESPONSE_CLOSE = _LT + "/tool_response" + _GT
THINK_OPEN = _LT + "think" + _GT
# 思考结束标签为分词器单一特殊 token (id 248069)
THINK_CLOSE = _LT + "/th" + "ink" + _GT

_THINK_BLOCK_RE = re.compile(
    re.escape(THINK_OPEN) + r".*?(?:" + re.escape(THINK_CLOSE) + r"|$\Z)",
    re.DOTALL,
)


def clean_think_tags(text: str) -> str:
    """移除文本中的思考块，返回可见内容

    兼容思考开标签已被 prompt 消费的场景: 若仍存在思考结束标签，
    则其之前的全部内容视为思考文本一并剥离。
    """
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    pos = cleaned.find(THINK_CLOSE)
    if pos != -1:
        cleaned = cleaned[pos + len(THINK_CLOSE):]
    return cleaned.strip()


def fn_open(name: str) -> str:
    return _LT + "function=" + name + _GT


def param_open(key: str) -> str:
    return _LT + "parameter=" + key + _GT


# ── System Prompt 工具声明块 (与 chat_template.jinja 文本一致) ──

TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n" + _LT + "tools" + _GT
TOOLS_FOOTER = _LT + "/tools" + _GT

TOOLS_INSTRUCTION = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    + TC_OPEN + "\n"
    + _LT + "function=example_function_name" + _GT + "\n"
    + _LT + "parameter=example_parameter_1" + _GT + "\nvalue_1\n" + PARAM_CLOSE + "\n"
    + _LT + "parameter=example_parameter_2" + _GT + "\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" + PARAM_CLOSE + "\n"
    + FN_CLOSE + "\n" + TC_CLOSE + "\n\n"
    + _LT + "IMPORTANT" + _GT + "\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner " + _LT + "function=..." + _GT + FN_CLOSE + " block must be nested within " + TC_OPEN + TC_CLOSE + " XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
    + _LT + "/IMPORTANT" + _GT
)

AGENT_MODE_NOTE = (
    "\n\nAgent 模式补充约束: 调用任何工具前，必须先用思考结束标签 ("
    + THINK_CLOSE
    + ") 关闭当前思考块，然后直接输出工具调用块；调用块之后不要再输出任何内容。"
)


def render_tools_block(tools: List[Dict[str, Any]]) -> str:
    """将 OpenAI 格式 tools 列表渲染为模型原生工具声明文本"""
    lines = [TOOLS_HEADER]
    for tool in tools:
        lines.append("\n" + json.dumps(tool, ensure_ascii=False))
    lines.append("\n" + TOOLS_FOOTER)
    lines.append(TOOLS_INSTRUCTION)
    lines.append(AGENT_MODE_NOTE)
    return "".join(lines)


# ── 模型原生输出 → OpenAI tool_calls 解析 ──

_BLOCK_RE = re.compile(
    r"(?:" + re.escape(TC_OPEN) + r"\s*)?" + _LT + r"function=([\w.\-]+)\s*" + _GT + r"(.*?)(?:" + re.escape(FN_CLOSE) + r"(?:\s*" + re.escape(TC_CLOSE) + r")?|" + re.escape(TC_CLOSE) + r")",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    _LT + r"parameter=([\w.\-]+)" + _GT + r"(?:\r?\n)?(.*?)" + re.escape(PARAM_CLOSE),
    re.DOTALL,
)
_TC_BLOCK_RE = re.compile(
    re.escape(TC_OPEN) + r"\s*(.*?)\s*" + re.escape(TC_CLOSE),
    re.DOTALL,
)


def _normalize_param_value(raw: str) -> Any:
    """归一化参数值：严格保留代码缩进与内部格式，仅去除首尾紧邻闭合标签的单一换行，安全反序列化 JSON"""
    val = raw
    if val.startswith("\r\n"):
        val = val[2:]
    elif val.startswith("\n"):
        val = val[1:]
    if val.endswith("\r\n"):
        val = val[:-2]
    elif val.endswith("\n"):
        val = val[:-1]

    stripped = val.strip()
    if stripped:
        # 仅当明显是 JSON 结构时尝试 loads (对象、数组、布尔值、数字、带引号字符串)
        if (
            (stripped.startswith("{") and stripped.endswith("}"))
            or (stripped.startswith("[") and stripped.endswith("]"))
            or stripped in ("true", "false", "null")
            or re.match(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$", stripped)
            or (stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2)
        ):
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                pass
    return val


def _extract_json_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """从 <tool_call>... 中解析 JSON 格式工具调用"""
    tool_calls: List[Dict[str, Any]] = []

    def repl(m: re.Match) -> str:
        content = m.group(1).strip()
        fence_m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", content, re.DOTALL)
        if fence_m:
            content = fence_m.group(1).strip()
        try:
            parsed = json.loads(content)
            items = parsed if isinstance(parsed, list) else [parsed]
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "function" and "function" in item:
                    fn_info = item["function"]
                    name = fn_info.get("name")
                    args = fn_info.get("arguments", {})
                elif "name" in item:
                    name = item.get("name")
                    args = item.get("arguments") or item.get("parameters") or {}
                else:
                    continue
                if not name:
                    continue
                args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": str(name),
                        "arguments": args_str,
                    },
                })
            return ""
        except (json.JSONDecodeError, ValueError):
            return m.group(0)

    remaining = _TC_BLOCK_RE.sub(repl, text)
    return remaining, tool_calls


def parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """从模型输出解析工具调用，返回 (去除调用块后的可见文本, OpenAI tool_calls 列表)"""
    tool_calls: List[Dict[str, Any]] = []

    # 1. 优先解析原生 XML 格式 <function=...>
    for match in _BLOCK_RE.finditer(text):
        name = match.group(1)
        params: Dict[str, Any] = {}
        for p in _PARAM_RE.finditer(match.group(2)):
            key = p.group(1)
            params[key] = _normalize_param_value(p.group(2))
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(params, ensure_ascii=False),
            },
        })

    visible = _BLOCK_RE.sub("", text)

    # 2. 若未解析到 XML 格式或仍残留 <tool_call> 标签，尝试解析 JSON 格式
    if not tool_calls or TC_OPEN in visible:
        visible, json_calls = _extract_json_tool_calls(visible)
        tool_calls.extend(json_calls)

    # 3. 清理可能残留的包裹标签
    if tool_calls:
        visible = visible.replace(TC_OPEN, "").replace(TC_CLOSE, "").replace(FN_CLOSE, "")

    return visible.strip(), tool_calls


# ── OpenAI 消息历史 → 模型原生格式 ──

def serialize_assistant_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    """将 OpenAI tool_calls 序列化回模型原生调用格式 (用于多轮历史)

    多个调用渲染在同一包裹标签内 (与 chat_template.jinja 原生约定一致)。
    """
    if not tool_calls:
        return ""
    blocks = []
    for call in tool_calls:
        fn = call.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        parts = [TC_OPEN, "\n", fn_open(name), "\n"]
        for key, value in args.items():
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            parts.append(param_open(key) + "\n" + rendered + "\n" + PARAM_CLOSE + "\n")
        parts.append(FN_CLOSE + "\n" + TC_CLOSE)
        blocks.append("".join(parts))
    return "\n".join(blocks)


def render_tool_message(content: str) -> str:
    """工具执行结果消息包装为模型原生格式 (采用 Qwen 官方 <tool_response>)"""
    return TOOL_RESPONSE_OPEN + "\n" + content + "\n" + TOOL_RESPONSE_CLOSE


FN_OPEN_PREFIX = _LT + "function="
FN_PARTIAL_TAIL = _LT + "function"


class StreamRegionFilter:
    """流式输出区域过滤器: 透传可见文本，抑制思考块与工具调用区域。

    工具开式支持完整包裹标签或裸 <function=NAME> (模型常省略外围调用包裹标签)。
    一旦进入工具区域，后续文本全部吞掉 (结束标签可缺失)，可见正文只出现在调用之前。
    """

    def __init__(self) -> None:
        self._buf = ""
        self._state = "text"  # text | think | tool

    @property
    def in_thinking(self) -> bool:
        return self._state == "think"

    def feed(self, text: str) -> str:
        self._buf += text
        return self._consume(final=False)

    def flush(self) -> str:
        return self._consume(final=True)

    def has_tool_region(self) -> bool:
        return self._state == "tool"

    def _consume(self, final: bool) -> str:
        out: List[str] = []
        while True:
            if self._state == "tool":
                self._buf = ""
                break
            if self._state == "think":
                pos, end_len = self._find_think_close()
                if pos == -1:
                    if final:
                        self._buf = ""
                        self._state = "text"
                    else:
                        keep = min(len(self._buf), len(THINK_CLOSE) - 1)
                        self._buf = self._buf[-keep:] if keep else ""
                    break
                self._buf = self._buf[pos + end_len:]
                self._state = "text"
                continue
            idx, kind = self._first_marker()
            if idx is None:
                keep = 0 if final else self._partial_hold()
                cut = len(self._buf) - keep
                out.append(self._buf[:cut])
                self._buf = self._buf[cut:]
                break
            out.append(self._buf[:idx])
            if kind == "think":
                self._buf = self._buf[idx + len(THINK_OPEN):]
                self._state = "think"
            elif kind == "tc":
                self._buf = self._buf[idx + len(TC_OPEN):]
                self._state = "tool"
            else:  # 裸 <function=NAME> 开式
                gt = self._buf.find(_GT, idx)
                if gt == -1:
                    self._buf = self._buf[idx:]
                    break
                self._buf = self._buf[gt + 1:]
                self._state = "tool"
        if final and self._state == "text":
            out.append(self._buf)
            self._buf = ""
        return "".join(out)

    def _first_marker(self) -> Tuple[Optional[int], str]:
        best: Tuple[Optional[int], str] = (None, "")
        for tag, kind in ((THINK_OPEN, "think"), (TC_OPEN, "tc"), (FN_OPEN_PREFIX, "fn")):
            i = self._buf.find(tag)
            if i != -1 and (best[0] is None or i < best[0]):
                best = (i, kind)
        return best

    def _find_think_close(self) -> Tuple[int, int]:
        """返回 (思考结束标签位置, 匹配长度)，未找到返回 (-1, 0)"""
        pos = self._buf.find(THINK_CLOSE)
        return (pos, len(THINK_CLOSE)) if pos != -1 else (-1, 0)

    def _partial_hold(self) -> int:
        hold = 0
        candidates = [THINK_OPEN, TC_OPEN, FN_OPEN_PREFIX, FN_PARTIAL_TAIL]
        for tag in candidates:
            for k in range(min(len(self._buf), len(tag) - 1), 0, -1):
                if self._buf.endswith(tag[:k]):
                    hold = max(hold, k)
                    break
        return hold
