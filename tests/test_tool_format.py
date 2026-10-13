"""
engine/tool_format.py 单元测试: 工具调用解析/序列化/流式过滤器
标签全部通过模块常量拼接构造，避免源码中出现保留控制词
"""

import json

import pytest

from engine import tool_format as tf


def _call_block(name: str, params: dict) -> str:
    parts = [tf.TC_OPEN, "\n", tf.fn_open(name), "\n"]
    for key, value in params.items():
        parts.append(tf.param_open(key) + "\n" + value + "\n" + tf.PARAM_CLOSE + "\n")
    parts.append(tf.FN_CLOSE + "\n" + tf.TC_CLOSE)
    return "".join(parts)


class TestParseToolCalls:
    def test_single_call(self):
        block = _call_block("write", {"path": "a.txt", "content": "hello"})
        visible, calls = tf.parse_tool_calls("前言\n" + block + "\n后记")
        assert len(calls) == 1
        fn = calls[0]["function"]
        assert fn["name"] == "write"
        assert json.loads(fn["arguments"]) == {"path": "a.txt", "content": "hello"}
        assert calls[0]["id"].startswith("call_")
        assert "前言" in visible and "后记" in visible
        assert tf.TC_OPEN not in visible

    def test_multiple_calls(self):
        text = _call_block("read", {"p": "1"}) + "\n" + _call_block("bash", {"c": "ls"})
        _, calls = tf.parse_tool_calls(text)
        assert [c["function"]["name"] for c in calls] == ["read", "bash"]

    def test_no_call(self):
        visible, calls = tf.parse_tool_calls("普通回答，没有调用")
        assert calls == []
        assert visible == "普通回答，没有调用"

    def test_parse_bare_call_without_wrapper(self):
        text = "结果\n" + tf.fn_open("write") + "\n" + tf.param_open("content") + "\nhi\n" + tf.PARAM_CLOSE + tf.FN_CLOSE
        visible, calls = tf.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "write"
        assert json.loads(calls[0]["function"]["arguments"]) == {"content": "hi"}
        assert tf.fn_open("write") not in visible

    def test_parameter_code_indentation_preserved(self):
        # 严格保留 Python 代码多行缩进与内部格式
        code = "    def calculate(x: int) -> int:\n        return x * 2"
        text = tf.TC_OPEN + "\n" + tf.fn_open("write_file") + "\n" + tf.param_open("code") + "\n" + code + "\n" + tf.PARAM_CLOSE + "\n" + tf.FN_CLOSE + "\n" + tf.TC_CLOSE
        _, calls = tf.parse_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["code"] == code

    def test_json_tool_call_inside_tag(self):
        # 支持模型输出 JSON 格式的工具调用块
        text = "分析中...\n" + tf.TC_OPEN + '\n{"name": "bash", "arguments": {"cmd": "ls -la"}}\n' + tf.TC_CLOSE + "\n完成"
        visible, calls = tf.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "bash"
        assert json.loads(calls[0]["function"]["arguments"]) == {"cmd": "ls -la"}
        assert "分析中..." in visible and "完成" in visible

    def test_json_tool_call_markdown_fenced(self):
        # 支持 Markdown 围栏内嵌的 JSON 工具调用
        text = tf.TC_OPEN + '\n```json\n{"name": "read_file", "parameters": {"path": "a.py"}}\n```\n' + tf.TC_CLOSE
        _, calls = tf.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "read_file"
        assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py"}

    def test_function_closed_by_tool_call_directly(self):
        # 容错：模型省略 </function> 直接输出 </tool_call>
        text = tf.TC_OPEN + "\n" + tf.fn_open("pwd") + "\n" + tf.param_open("env") + "\nprod\n" + tf.PARAM_CLOSE + "\n" + tf.TC_CLOSE
        _, calls = tf.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "pwd"
        assert json.loads(calls[0]["function"]["arguments"]) == {"env": "prod"}


class TestSerialize:
    def test_roundtrip(self):
        block = _call_block("exec", {"cmd": "echo hi"})
        _, calls = tf.parse_tool_calls(block)
        serialized = tf.serialize_assistant_tool_calls(calls)
        _, calls2 = tf.parse_tool_calls(serialized)
        assert len(calls2) == 1
        assert calls2[0]["function"]["name"] == "exec"
        assert json.loads(calls2[0]["function"]["arguments"]) == {"cmd": "echo hi"}


class TestRenderToolsBlock:
    def test_contains_schema_and_instruction(self):
        tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]
        text = tf.render_tools_block(tools)
        assert "# Tools" in text
        assert "bash" in text
        assert tf.TC_OPEN in text  # 指令示例中包含调用格式


class TestStreamRegionFilter:
    def test_plain_text(self):
        f = tf.StreamRegionFilter()
        out = f.feed("hello ") + f.feed("world") + f.flush()
        assert out == "hello world"

    def test_suppress_think_then_tool_region(self):
        # 工具区域一旦进入，其后文本全部吞掉 (模板约定调用后不应再有正文)
        stream = "看这里" + tf.THINK_OPEN + "内部推理" + tf.THINK_CLOSE + " 可见A " + _call_block("bash", {"c": "ls"}) + " 吞掉B"
        f = tf.StreamRegionFilter()
        collected = "".join(f.feed(ch) for ch in stream) + f.flush()
        assert collected == "看这里 可见A "

    def test_bare_function_open_swallows_tail(self):
        stream = "答案: 42" + tf.fn_open("calc") + "\n" + tf.param_open("x") + "\n1\n" + tf.PARAM_CLOSE + tf.FN_CLOSE
        f = tf.StreamRegionFilter()
        out = f.feed(stream) + f.flush()
        assert out == "答案: 42"

    def test_partial_opener_not_leaked(self):
        f = tf.StreamRegionFilter()
        partial = f.feed("text <too")
        rest = f.feed("l_call>" + tf.fn_open("x") + tf.PARAM_CLOSE + tf.FN_CLOSE + tf.TC_CLOSE + "end")
        final = partial + rest + f.flush()
        assert final == "text "

    def test_marker_reassembly_no_false_positive(self):
        # 普通尖括号文本不应被误吞
        f = tf.StreamRegionFilter()
        out = f.feed("a < b && c <> d") + f.flush()
        assert out == "a < b && c <> d"


class TestRenderToolMessage:
    def test_tool_response_tags(self):
        msg = tf.render_tool_message("file content here")
        assert tf.TOOL_RESPONSE_OPEN in msg
        assert tf.TOOL_RESPONSE_CLOSE in msg
        assert "file content here" in msg
