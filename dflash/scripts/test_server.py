import os
import struct
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from _prefill_hook import PrefillConfig
from server import (
    build_app, MODEL_NAME,
    parse_tool_calls, parse_reasoning,
    normalize_stop, first_stop_match,
)


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1]
    tokenizer.decode.return_value = "hello"
    tokenizer.apply_chat_template.return_value = "prompt"
    return tokenizer


@pytest.fixture
def app(mock_tokenizer):
    """Build a FastAPI app with mocked daemon."""
    with patch("server.subprocess.Popen") as mock_popen:
        mock_popen.return_value.poll.return_value = None  # daemon alive
        a = build_app(
            target=Path("target.gguf"),
            draft=Path("draft.safetensors"),
            bin_path=Path("test_dflash"),
            budget=22,
            max_ctx=131072,
            tokenizer=mock_tokenizer,
            stop_ids={2},
        )
    return a


@pytest.fixture
def app_with_prefill(mock_tokenizer):
    """Build a FastAPI app with compression enabled."""
    drafter_tokenizer = MagicMock()
    drafter_tokenizer.return_value = {"input_ids": [1, 2, 3]}
    drafter_tokenizer.decode.return_value = "compressed prompt"
    with patch("server.subprocess.Popen") as mock_popen:
        mock_popen.return_value.poll.return_value = None  # daemon alive
        a = build_app(
            target=Path("target.gguf"),
            draft=Path("draft.safetensors"),
            bin_path=Path("test_dflash"),
            budget=22,
            max_ctx=131072,
            tokenizer=mock_tokenizer,
            stop_ids={2},
            prefill_cache_slots=0,
            prefill_cfg=PrefillConfig(
                mode="always",
                threshold=1,
                keep_ratio=0.5,
                drafter_gguf=Path("drafter.gguf"),
                drafter_tokenizer_id="mock-tokenizer",
            ),
            drafter_tokenizer=drafter_tokenizer,
        )
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


def _build_app_with_process(mock_tokenizer, process, *, enable_prefill: bool = False):
    kwargs = {}
    if enable_prefill:
        drafter_tokenizer = MagicMock()
        drafter_tokenizer.return_value = {"input_ids": [1, 2, 3]}
        drafter_tokenizer.decode.return_value = "compressed prompt"
        kwargs.update(
            prefill_cache_slots=0,
            prefill_cfg=PrefillConfig(
                mode="always",
                threshold=1,
                keep_ratio=0.5,
                drafter_gguf=Path("drafter.gguf"),
                drafter_tokenizer_id="mock-tokenizer",
            ),
            drafter_tokenizer=drafter_tokenizer,
        )
    with patch("server.subprocess.Popen") as mock_popen:
        mock_popen.return_value = process
        return build_app(
            target=Path("target.gguf"),
            draft=Path("draft.safetensors"),
            bin_path=Path("test_dflash"),
            budget=22,
            max_ctx=131072,
            tokenizer=mock_tokenizer,
            stop_ids={2},
            **kwargs,
        )


def _chat_sse_chunks(text: str) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in text.strip().split("\n\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _responses_sse_events(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_line = next((line for line in block.splitlines() if line.startswith("event: ")), None)
        data_line = next((line for line in block.splitlines() if line.startswith("data: ")), None)
        if event_line and data_line:
            events.append((event_line[7:], json.loads(data_line[6:])))
    return events


def _chat_stream_assistant_message(chunks: list[dict]) -> dict:
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            text = delta.get("content")
            if isinstance(text, str):
                content_parts.append(text)
            for tc_delta in delta.get("tool_calls", []):
                index = tc_delta["index"]
                state = tool_calls.setdefault(index, {
                    "id": tc_delta.get("id"),
                    "type": tc_delta.get("type", "function"),
                    "function": {"name": None, "arguments": ""},
                })
                if tc_delta.get("id"):
                    state["id"] = tc_delta["id"]
                fn_delta = tc_delta.get("function", {})
                if fn_delta.get("name"):
                    state["function"]["name"] = fn_delta["name"]
                if fn_delta.get("arguments"):
                    state["function"]["arguments"] += fn_delta["arguments"]
    msg = {"role": "assistant"}
    content = "".join(content_parts)
    msg["content"] = content or None
    if tool_calls:
        msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return msg


# ─── parse_reasoning ───────────────────────────────────────────────

class TestParseReasoning:
    def test_full_think_tags(self):
        cleaned, reasoning = parse_reasoning("<think>my reasoning</think>\n\nthe answer")
        assert cleaned == "the answer"
        assert reasoning == "my reasoning"

    def test_headless_think(self):
        """Model started in thinking — output has no <think>, just body+</think>."""
        cleaned, reasoning = parse_reasoning("chain of thought</think>\n\nvisible")
        assert cleaned == "visible"
        assert reasoning == "chain of thought"

    def test_no_think_tags_thinking_enabled(self):
        """With thinking enabled but no tags and not started_in_thinking, text is content."""
        cleaned, reasoning = parse_reasoning("all content", thinking_enabled=True)
        assert cleaned == "all content"
        assert reasoning is None

    def test_no_think_tags_thinking_disabled(self):
        """With thinking disabled, plain text passes through as content."""
        cleaned, reasoning = parse_reasoning("plain answer", thinking_enabled=False)
        assert cleaned == "plain answer"
        assert reasoning is None

    def test_started_in_thinking_no_close_tag(self):
        """Truncated reasoning when prompt started in thinking mode."""
        cleaned, reasoning = parse_reasoning(
            "unfinished thought", started_in_thinking=True)
        assert cleaned == ""
        assert reasoning == "unfinished thought"

    def test_started_in_thinking_with_close_tag(self):
        """Full reasoning block when prompt started in thinking mode."""
        cleaned, reasoning = parse_reasoning(
            "thought body</think>the answer", started_in_thinking=True)
        assert cleaned == "the answer"
        assert reasoning == "thought body"

    def test_empty_think_block(self):
        cleaned, reasoning = parse_reasoning("<think></think>answer")
        assert cleaned == "answer"
        assert reasoning is None  # empty reasoning stripped to None

    def test_multiline_reasoning(self):
        text = "<think>line1\nline2\nline3</think>result"
        cleaned, reasoning = parse_reasoning(text)
        assert cleaned == "result"
        assert "line1" in reasoning and "line3" in reasoning

    def test_repeated_leading_think_closers_are_stripped(self):
        cleaned, reasoning = parse_reasoning("</think>\n</think>\n8")
        assert cleaned == "8"
        assert reasoning is None


# ─── parse_tool_calls ─────────────────────────────────────────────

class TestParseToolCalls:
    def test_single_tool_call(self):
        text = (
            'Sure!\n<tool_call>'
            '<function=read_file><parameter=path>test.py</parameter></function>'
            '</tool_call>'
        )
        cleaned, calls = parse_tool_calls(text, tools=None)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "read_file"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["path"] == "test.py"
        assert cleaned.strip() == "Sure!"

    def test_no_tool_tags(self):
        text = "Just a plain answer."
        cleaned, calls = parse_tool_calls(text, tools=None)
        assert calls == []
        assert cleaned == text

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>'
            '<function=read_file><parameter=path>a.py</parameter></function>'
            '</tool_call>'
            '<tool_call>'
            '<function=read_file><parameter=path>b.py</parameter></function>'
            '</tool_call>'
        )
        cleaned, calls = parse_tool_calls(text, tools=None)
        assert len(calls) == 2
        assert json.loads(calls[0]["function"]["arguments"])["path"] == "a.py"
        assert json.loads(calls[1]["function"]["arguments"])["path"] == "b.py"

    def test_multiple_parameters(self):
        text = (
            '<tool_call>'
            '<function=write_file>'
            '<parameter=path>out.txt</parameter>'
            '<parameter=content>hello world</parameter>'
            '</function></tool_call>'
        )
        cleaned, calls = parse_tool_calls(text, tools=None)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["path"] == "out.txt"
        assert args["content"] == "hello world"

    def test_bare_qwen_xml_function_call(self):
        text = (
            '<function=read>\n'
            '<parameter=path>\n'
            '~/.npm-global/lib/node_modules/openclaw/skills/weather/SKILL.md\n'
            '</parameter>\n'
            '</function>\n'
            '</tool_call>'
        )
        cleaned, calls = parse_tool_calls(text, tools=[{
            "type": "function",
            "function": {"name": "read", "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            }},
        }])
        assert cleaned == ""
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "read"
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "path": "~/.npm-global/lib/node_modules/openclaw/skills/weather/SKILL.md"
        }

    def test_tool_call_id_format(self):
        text = "<tool_call><function=f><parameter=x>1</parameter></function></tool_call>"
        _, calls = parse_tool_calls(text, tools=None)
        assert calls[0]["id"].startswith("call_")
        assert calls[0]["type"] == "function"

    def test_text_before_and_after_tool_call(self):
        text = "Before\n<tool_call><function=f><parameter=x>1</parameter></function></tool_call>\nAfter"
        cleaned, calls = parse_tool_calls(text, tools=None)
        assert len(calls) == 1
        assert "Before" in cleaned
        assert "After" in cleaned

    def test_function_signature_tool_call(self):
        text = '<function=web_search(query="Open Claw docs documentation")</function>'
        cleaned, calls = parse_tool_calls(text, tools=[{
            "type": "function",
            "function": {"name": "web_search", "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            }},
        }])
        assert cleaned == ""
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "query": "Open Claw docs documentation"
        }

    @pytest.mark.parametrize("text", [
        '{"name":"web_search","arguments":{"query":"OpenAI docs"}}',
        '<tool_code>{"name":"web_search","arguments":{"query":"OpenAI docs"}}</tool_code>',
    ])
    def test_json_tool_call_shapes(self, text):
        cleaned, calls = parse_tool_calls(text, tools=[{
            "type": "function",
            "function": {"name": "web_search", "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            }},
        }])
        assert cleaned == ""
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "OpenAI docs"}

    def test_multiple_mixed_tool_call_shapes(self):
        text = (
            '<function=web_search(query="OpenAI docs")</function>'
            '{"name":"read_file","arguments":{"path":"README.md"}}'
        )
        cleaned, calls = parse_tool_calls(text, tools=[
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "read_file"}},
        ])
        assert cleaned == ""
        assert [c["function"]["name"] for c in calls] == ["web_search", "read_file"]
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "OpenAI docs"}
        assert json.loads(calls[1]["function"]["arguments"]) == {"path": "README.md"}

    def test_unknown_tool_name_preserved_when_tools_are_known(self):
        text = '{"name":"unknown_tool","arguments":{"query":"OpenAI docs"}}'
        cleaned, calls = parse_tool_calls(text, tools=[{
            "type": "function",
            "function": {"name": "web_search"},
        }])
        assert calls == []
        assert cleaned == text

    def test_malformed_function_signature_is_preserved(self):
        text = '<function=web_search(query="unterminated"</function>'
        cleaned, calls = parse_tool_calls(text, tools=[{
            "type": "function",
            "function": {"name": "web_search"},
        }])
        assert calls == []
        assert cleaned == text


# ─── normalize_stop / first_stop_match ──────────────────────────────

class TestStopHelpers:
    def test_normalize_none(self):
        assert normalize_stop(None) == []

    def test_normalize_string(self):
        assert normalize_stop("stop") == ["stop"]

    def test_normalize_list(self):
        assert normalize_stop(["a", "b"]) == ["a", "b"]

    def test_normalize_empty_string(self):
        assert normalize_stop("") == []

    def test_first_stop_match_found(self):
        assert first_stop_match("hello world stop here", ["stop"]) == 12

    def test_first_stop_match_multiple(self):
        assert first_stop_match("ab cd ef", ["cd", "ab"]) == 0  # "ab" is earliest

    def test_first_stop_match_none(self):
        assert first_stop_match("hello world", ["xyz"]) == -1

    def test_first_stop_match_empty_list(self):
        assert first_stop_match("hello", []) == -1


# ─── /health endpoint ─────────────────────────────────────────────

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ─── CORS headers ─────────────────────────────────────────────────

def test_cors_headers(client):
    response = client.options(
        "/v1/models",
        headers={"Origin": "http://localhost:3000",
                 "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


# ─── GET /v1/models ────────────────────────────────────────────────

def test_models_endpoint(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == MODEL_NAME


def test_codex_models_endpoint(client):
    """Codex sends ?client_version= and expects {"models":[...]} format."""
    response = client.get("/v1/models?client_version=0.1.0")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "data" not in data  # must NOT have OpenAI format
    m = data["models"][0]
    assert m["slug"] == MODEL_NAME
    assert "context_window" in m
    assert "supported_reasoning_levels" in m
    assert m["shell_type"] == "shell_command"
    assert m["supports_parallel_tool_calls"] is False


# ─── POST /v1/chat/completions ─────────────────────────────────────

@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_completions_non_streaming(mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.return_value = "chain of thought</think>\n\nvisible answer"
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "visible answer"
    assert data["choices"][0]["message"]["reasoning_content"] == "chain of thought"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_completions_non_streaming_with_tool_call(mock_os_read, mock_pipe,
                                                        mock_tokenizer, app):
    """Non-streaming chat returns tool_calls when model outputs <tool_call>."""
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.return_value = (
        '<tool_call>'
        '<function=read_file><parameter=path>test.py</parameter></function>'
        '</tool_call>'
    )
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read test.py"}],
        "stream": False,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    tc = data["choices"][0]["message"]["tool_calls"]
    assert len(tc) == 1
    assert tc[0]["function"]["name"] == "read_file"


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_completions_replays_raw_tool_call_text(mock_os_read, mock_pipe,
                                                     mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    raw_tool_text = (
        "Before\n"
        "<tool_call>"
        "<function=read_file><parameter=path>test.py</parameter></function>"
        "</tool_call>\n"
        "After"
    )
    mock_tokenizer.decode.side_effect = [raw_tool_text, "followup"]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", -1),
        struct.pack("<i", 11), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    first = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read test.py"}],
        "stream": False,
    })
    assert first.status_code == 200
    assistant_msg = first.json()["choices"][0]["message"]

    second = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "read test.py"},
            assistant_msg,
            {"role": "tool", "tool_call_id": assistant_msg["tool_calls"][0]["id"], "content": "file body"},
            {"role": "user", "content": "what next?"},
        ],
        "stream": False,
    })
    assert second.status_code == 200

    msgs = mock_tokenizer.apply_chat_template.call_args_list[-1][0][0]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["content"] == raw_tool_text
    assert "tool_calls" not in assistant


@patch("server.compress_text_via_daemon")
@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_tool_requests_skip_compression(mock_os_read, mock_pipe, mock_compress,
                                             mock_tokenizer, app_with_prefill):
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app_with_prefill)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read test.py"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }],
        "tool_choice": "none",
        "stream": False,
    })

    assert response.status_code == 200
    mock_compress.assert_not_called()


@patch("server.os.pipe")
@patch("server.os.read")
def test_zero_token_prompt_is_rejected_before_daemon(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.encode.return_value = []

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [],
        "stream": False,
    })

    assert response.status_code == 400
    data = response.json()
    assert data["error"]["type"] == "invalid_request_error"
    assert data["error"]["param"] == "messages"
    assert "zero tokens" in data["error"]["message"]
    mock_os_read.assert_not_called()


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_completions_streaming(mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.apply_chat_template.return_value = "<think>\n"
    mock_tokenizer.decode.side_effect = ["thought", "</think>", "answer"]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11),
        struct.pack("<i", 12), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })

    assert response.status_code == 200
    text = response.text
    assert "data: [DONE]" in text
    # Parse all SSE chunks
    chunks = [json.loads(line[6:]) for line in text.strip().split("\n\n")
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert len(chunks) >= 1
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


@patch("server.PrefixCache.lookup_full")
@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_streaming_full_cache_hit_preserves_started_in_thinking(
        mock_os_read, mock_pipe, mock_lookup_full, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_lookup_full.return_value = (7, "cached_prompt.bin", 1)
    mock_tokenizer.apply_chat_template.return_value = "prompt<think>\n"

    def decode_side_effect(ids, *args, **kwargs):
        token_id = ids[0] if isinstance(ids, list) else ids
        return {
            10: "hidden reasoning",
            11: "</think>",
            12: "visible answer",
        }[token_id]

    mock_tokenizer.decode.side_effect = decode_side_effect
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11),
        struct.pack("<i", 12), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })

    assert response.status_code == 200
    chunks = _chat_sse_chunks(response.text)
    reasoning = "".join(
        choice.get("delta", {}).get("reasoning_content", "")
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    content = "".join(
        choice.get("delta", {}).get("content", "")
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    assert reasoning == "hidden reasoning"
    assert content == "visible answer"
    assert "</think>" not in content
    assert "hidden reasoning" not in content


@patch("server.os.pipe")
@patch("server.os.read")
@pytest.mark.parametrize(("decoded_chunks", "leaked_fragments"), [
    (
        [
            "<tool_call><function=read_file><parameter=path>test",
            ".py</parameter></function></tool_call>",
        ],
        ["<tool_call>", "<function="],
    ),
    (
        [
            "<function=read_file><parameter=path>test",
            ".py</parameter></function>",
        ],
        ["<function="],
    ),
    (
        [
            '{"name":"read_file","arguments":{"path":"test',
            '.py"}}',
        ],
        ['{"name":"read_file"'],
    ),
    (
        [
            '<tool_code>{"name":"read_file","arguments":{"path":"test',
            '.py"}}</tool_code>',
        ],
        ["<tool_code>", '{"name":"read_file"'],
    ),
])
def test_chat_completions_streaming_tool_call_deltas(mock_os_read, mock_pipe,
                                                     mock_tokenizer, app,
                                                     decoded_chunks, leaked_fragments):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.side_effect = decoded_chunks
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read test.py"}],
        "stream": True,
    })

    assert response.status_code == 200
    chunks = _chat_sse_chunks(response.text)
    tool_deltas = [
        chunk["choices"][0]["delta"]["tool_calls"][0]
        for chunk in chunks
        if chunk.get("choices")
        and chunk["choices"][0].get("delta", {}).get("tool_calls")
    ]
    assert len(tool_deltas) >= 2
    assert tool_deltas[0]["id"].startswith("call_")
    assert tool_deltas[0]["function"]["name"] == "read_file"
    assert "".join(
        delta.get("function", {}).get("arguments", "")
        for delta in tool_deltas
    ) == '{"path":"test.py"}'
    assert not any(
        any(fragment in choice.get("delta", {}).get("content", "") for fragment in leaked_fragments)
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    finish_chunk = next(
        chunk for chunk in reversed(chunks)
        if chunk.get("choices") and chunk["choices"][0]["finish_reason"] is not None
    )
    assert finish_chunk["choices"][0]["finish_reason"] == "tool_calls"


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_streaming_stop_sequence_preserves_tool_deltas(mock_os_read, mock_pipe,
                                                            mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    stop_marker = "<END>"
    mock_tokenizer.decode.return_value = (
        "<tool_call>"
        "<function=read_file><parameter=path>test.py</parameter></function>"
        "</tool_call>"
        f"{stop_marker}"
    )
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read test.py"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }],
        "stop": stop_marker,
        "stream": True,
    })

    assert response.status_code == 200
    chunks = _chat_sse_chunks(response.text)
    assert not any(
        "<tool_call>" in choice.get("delta", {}).get("content", "")
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    assistant_msg = _chat_stream_assistant_message(chunks)
    assert assistant_msg["content"] is None
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "read_file"
    assert assistant_msg["tool_calls"][0]["function"]["arguments"] == '{"path":"test.py"}'
    finish_chunk = next(
        chunk for chunk in reversed(chunks)
        if chunk.get("choices") and chunk["choices"][0]["finish_reason"] is not None
    )
    assert finish_chunk["choices"][0]["finish_reason"] == "tool_calls"


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_streaming_stop_sequence_after_bare_function_close_tag(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    stop_marker = "<END>"
    mock_tokenizer.decode.side_effect = [
        "<function=read_file><parameter=path>test",
        ".py</parameter></function>",
        f"{stop_marker}ignored",
    ]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11),
        struct.pack("<i", 12), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read test.py"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }],
        "stop": stop_marker,
        "stream": True,
    })

    assert response.status_code == 200
    chunks = _chat_sse_chunks(response.text)
    assistant_msg = _chat_stream_assistant_message(chunks)
    assert assistant_msg["content"] is None
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "read_file"
    assert assistant_msg["tool_calls"][0]["function"]["arguments"] == '{"path":"test.py"}'
    assert not any(
        stop_marker in choice.get("delta", {}).get("content", "")
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    finish_chunk = next(
        chunk for chunk in reversed(chunks)
        if chunk.get("choices") and chunk["choices"][0]["finish_reason"] is not None
    )
    assert finish_chunk["choices"][0]["finish_reason"] == "tool_calls"


@patch("server.os.pipe")
@patch("server.os.read")
@pytest.mark.parametrize(("decoded_chunks", "leaked_fragments"), [
    (
        [
            '{"name":"read_file","arguments":{"path":"test',
            '.py"}}<END>',
        ],
        ['{"name":"read_file"', "<END>"],
    ),
    (
        [
            '<tool_code>{"name":"read_file","arguments":{"path":"test',
            '.py"}}</tool_code><END>',
        ],
        ["<tool_code>", '{"name":"read_file"', "<END>"],
    ),
])
def test_chat_streaming_stop_sequence_after_json_tool_call(
        mock_os_read, mock_pipe, mock_tokenizer, app, decoded_chunks,
        leaked_fragments):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.side_effect = decoded_chunks
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read test.py"}],
        "stop": "<END>",
        "stream": True,
    })

    assert response.status_code == 200
    chunks = _chat_sse_chunks(response.text)
    assistant_msg = _chat_stream_assistant_message(chunks)
    assert assistant_msg["content"] is None
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "read_file"
    assert assistant_msg["tool_calls"][0]["function"]["arguments"] == '{"path":"test.py"}'
    assert not any(
        any(fragment in choice.get("delta", {}).get("content", "") for fragment in leaked_fragments)
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    finish_chunk = next(
        chunk for chunk in reversed(chunks)
        if chunk.get("choices") and chunk["choices"][0]["finish_reason"] is not None
    )
    assert finish_chunk["choices"][0]["finish_reason"] == "tool_calls"


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_streaming_replays_raw_tool_call_text(mock_os_read, mock_pipe,
                                                       mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    raw_tool_text = (
        "Before\\n"
        "<tool_call>"
        "<function=read_file><parameter=path>test.py</parameter></function>"
        "</tool_call>\\n"
        "After"
    )
    mock_tokenizer.decode.side_effect = [raw_tool_text, "followup"]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", -1),
        struct.pack("<i", 11), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    first = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read test.py"}],
        "stream": True,
    })
    assert first.status_code == 200
    assistant_msg = _chat_stream_assistant_message(_chat_sse_chunks(first.text))

    second = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "read test.py"},
            assistant_msg,
            {"role": "tool", "tool_call_id": assistant_msg["tool_calls"][0]["id"], "content": "file body"},
            {"role": "user", "content": "what next?"},
        ],
        "stream": False,
    })
    assert second.status_code == 200

    msgs = mock_tokenizer.apply_chat_template.call_args_list[-1][0][0]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["content"] == raw_tool_text
    assert "tool_calls" not in assistant


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_streaming_replays_stop_trimmed_raw_tool_call_text(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    stop_marker = "<END>"
    trimmed_raw_tool_text = (
        "Before\n"
        "<tool_call>"
        "<function=read_file><parameter=path>test.py</parameter></function>"
        "</tool_call>\n"
        "After"
    )
    mock_tokenizer.decode.side_effect = [
        trimmed_raw_tool_text + stop_marker + "ignored",
        "followup",
    ]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", -1),
        struct.pack("<i", 11), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    first = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read test.py"}],
        "stop": stop_marker,
        "stream": True,
    })
    assert first.status_code == 200
    assistant_msg = _chat_stream_assistant_message(_chat_sse_chunks(first.text))

    second = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "read test.py"},
            assistant_msg,
            {"role": "tool", "tool_call_id": assistant_msg["tool_calls"][0]["id"], "content": "file body"},
            {"role": "user", "content": "what next?"},
        ],
        "stream": False,
    })
    assert second.status_code == 200

    msgs = mock_tokenizer.apply_chat_template.call_args_list[-1][0][0]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["content"] == trimmed_raw_tool_text
    assert stop_marker not in assistant["content"]
    assert "ignored" not in assistant["content"]
    assert "tool_calls" not in assistant


@patch("server.os.pipe")
@patch("server.os.read")
@pytest.mark.parametrize("decoded_chunks", [
    [
        "<tool_call><function=read_file><parameter=path>test",
        ".py</parameter></function></tool_call><END>",
    ],
    [
        "<function=read_file><parameter=path>test",
        ".py</parameter></function><END>",
    ],
])
def test_chat_streaming_stop_hit_drains_daemon_before_next_stream(
        mock_os_read, mock_pipe, mock_tokenizer, app, decoded_chunks):
    mock_pipe.return_value = (1, 2)

    def decode_side_effect(ids, *args, **kwargs):
        token_id = ids[0] if isinstance(ids, list) else ids
        return {
            10: decoded_chunks[0],
            11: decoded_chunks[1],
            12: "STALE",
            20: "fresh reply",
        }[token_id]

    mock_tokenizer.decode.side_effect = decode_side_effect
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11),
        struct.pack("<i", 12), struct.pack("<i", -1),
        struct.pack("<i", 20), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    first = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read test.py"}],
        "stop": "<END>",
        "stream": True,
    })
    assert first.status_code == 200
    assistant_msg = _chat_stream_assistant_message(_chat_sse_chunks(first.text))
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "read_file"

    second = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    })
    assert second.status_code == 200
    second_msg = _chat_stream_assistant_message(_chat_sse_chunks(second.text))
    assert second_msg["content"] == "fresh reply"
    assert "tool_calls" not in second_msg


@patch("server.PrefixCache.abort_full_snap")
@patch("server.PrefixCache.confirm_full_snap")
@patch("server.PrefixCache.prepare_full_snap", return_value=(7, 0))
@patch("server.compress_text_via_daemon", return_value="compressed prompt")
@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_streaming_stop_hit_confirms_reserved_full_snapshot(
        mock_os_read, mock_pipe, _mock_compress, _mock_prepare_full_snap,
        mock_confirm_full_snap, mock_abort_full_snap, mock_tokenizer,
        app_with_prefill):
    mock_pipe.return_value = (1, 2)

    def decode_side_effect(ids, *args, **kwargs):
        token_id = ids[0] if isinstance(ids, list) else ids
        return {
            10: "hello",
            11: "<END>",
            12: "stale",
        }[token_id]

    mock_tokenizer.decode.side_effect = decode_side_effect
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11),
        struct.pack("<i", 12), struct.pack("<i", -1),
    ]

    client = TestClient(app_with_prefill)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hello"}],
        "stop": "<END>",
        "stream": True,
    })

    assert response.status_code == 200
    assistant_msg = _chat_stream_assistant_message(_chat_sse_chunks(response.text))
    assert assistant_msg["content"] == "hello"
    mock_confirm_full_snap.assert_called_once()
    slot, prompt_ids, cur_bin, cur_ids_len = mock_confirm_full_snap.call_args.args
    assert slot == 7
    assert prompt_ids == [1]
    assert isinstance(cur_bin, Path)
    assert cur_ids_len == 1
    mock_abort_full_snap.assert_not_called()


@patch("server.PrefixCache.abort_full_snap")
@patch("server.PrefixCache.prepare_full_snap", return_value=(7, 0))
@patch("server.compress_text_via_daemon", return_value="compressed prompt")
@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_streaming_write_failure_aborts_reserved_full_snapshot(
        mock_os_read, mock_pipe, _mock_compress, _mock_prepare_full_snap,
        mock_abort_full_snap, mock_tokenizer):
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = []
    dead_proc = MagicMock()
    dead_proc.poll.return_value = 1
    local_app = _build_app_with_process(
        mock_tokenizer, dead_proc, enable_prefill=True)

    client = TestClient(local_app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    })

    assert response.status_code == 200
    assert "dflash daemon has exited unexpectedly" in response.text
    mock_abort_full_snap.assert_called_once_with(7)


@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_tool_choice_required_rejects_plain_text(mock_os_read, mock_pipe,
                                                       mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.return_value = "plain text"
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read test.py"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }],
        "tool_choice": "required",
        "stream": False,
    })

    assert response.status_code == 400
    data = response.json()
    assert data["error"]["param"] == "tool_choice"
    assert "emit a tool call" in data["error"]["message"]


@patch("server.PrefixCache.abort_inline_snap")
@patch("server.PrefixCache.confirm_inline_snap")
@patch("server.PrefixCache.prepare_inline_snap", return_value=(3, 1))
@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_streaming_tool_choice_error_confirms_reserved_inline_snapshot(
        mock_os_read, mock_pipe, _mock_prepare_inline_snap,
        mock_confirm_inline_snap, mock_abort_inline_snap, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.return_value = (
        '<tool_call><function=write_file><parameter=path>file.txt'
        '</parameter></function></tool_call>'
    )
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "read file.txt"}],
        "tools": [
            {"type": "function", "function": {
                "name": "read_file",
                "description": "Read",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }},
            {"type": "function", "function": {
                "name": "write_file",
                "description": "Write",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }},
        ],
        "tool_choice": {"type": "function", "name": "read_file"},
        "stream": True,
    })

    assert response.status_code == 200
    chunks = _chat_sse_chunks(response.text)
    error = next(chunk for chunk in chunks if "error" in chunk)
    assert error["error"]["param"] == "tool_choice"
    mock_confirm_inline_snap.assert_called_once()
    slot, target_cut, cur_ids = mock_confirm_inline_snap.call_args.args
    assert slot == 3
    assert target_cut == 1
    assert cur_ids == [1]
    mock_abort_inline_snap.assert_not_called()


@patch("server.PrefixCache.abort_inline_snap")
@patch("server.PrefixCache.prepare_inline_snap", return_value=(3, 1))
@patch("server.os.pipe")
@patch("server.os.read")
def test_chat_streaming_write_failure_aborts_reserved_inline_snapshot(
        mock_os_read, mock_pipe, _mock_prepare_inline_snap,
        mock_abort_inline_snap, mock_tokenizer):
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = []
    dead_proc = MagicMock()
    dead_proc.poll.return_value = 1
    local_app = _build_app_with_process(mock_tokenizer, dead_proc)

    client = TestClient(local_app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    })

    assert response.status_code == 200
    assert "dflash daemon has exited unexpectedly" in response.text
    mock_abort_inline_snap.assert_called_once_with(3)


# ─── POST /v1/responses ───────────────────────────────────────────

@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_non_streaming(mock_os_read, mock_pipe, mock_tokenizer, app):
    """POST /v1/responses non-streaming returns ResponsesObject."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [{"type": "message", "role": "user", "content": "hello"}],
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert data["id"].startswith("resp_")
    assert data["output"][0]["type"] == "message"
    assert data["output"][0]["content"][0]["type"] == "output_text"
    assert data["usage"]["input_tokens"] > 0
    assert data["usage"]["output_tokens"] > 0
    assert data["usage"]["total_tokens"] == data["usage"]["input_tokens"] + data["usage"]["output_tokens"]


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_non_streaming_string_input(mock_os_read, mock_pipe,
                                                mock_tokenizer, app):
    """Responses API accepts a plain string as input."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "hello world",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_non_streaming_preserves_tool_call_before_trailing_text(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.return_value = (
        "<tool_call>"
        "<function=read_file><parameter=path>file.txt</parameter></function>"
        "</tool_call>"
        "After tool"
    )
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "read file.txt",
        "tools": [{
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }],
    })

    assert response.status_code == 200
    data = response.json()
    assert [item["type"] for item in data["output"]] == ["function_call", "message"]
    assert data["output"][1]["content"][0]["text"] == "After tool"
    assert data["output_text"] == "After tool"


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_non_streaming_started_in_thinking(mock_os_read, mock_pipe,
                                                       mock_tokenizer, app):
    """When prompt ends with <think>, reasoning without tags is not misclassified as content."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    # Simulate a chat template that prefills <think>\n
    mock_tokenizer.apply_chat_template.return_value = "prompt<think>\n"
    # Model output has no <think> tags — it's a continuation of the prefilled block
    mock_tokenizer.decode.return_value = "internal reasoning</think>\nactual answer"

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [{"type": "message", "role": "user", "content": "hello"}],
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    # The "actual answer" part should be the output text, not the reasoning
    assert "actual answer" in data["output_text"]
    # The reasoning should NOT leak into the output text
    assert "internal reasoning" not in data["output_text"]


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_with_instructions(mock_os_read, mock_pipe,
                                      mock_tokenizer, app):
    """Instructions are mapped to system message."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "hi",
        "instructions": "You are a coding assistant.",
    })

    assert response.status_code == 200
    # Verify apply_chat_template was called with system message
    calls = mock_tokenizer.apply_chat_template.call_args_list
    last_call = calls[-1]
    msgs = last_call[0][0]  # first positional arg
    assert msgs[0]["role"] == "system"
    assert "coding assistant" in msgs[0]["content"]


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_streaming(mock_os_read, mock_pipe, mock_tokenizer, app):
    """POST /v1/responses streaming emits proper SSE lifecycle events."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "say hello",
        "stream": True,
    })

    assert response.status_code == 200
    text = response.text
    # Must contain key SSE lifecycle events in order
    assert "event: response.created" in text
    assert "event: response.output_item.added" in text
    assert "event: response.content_part.added" in text
    assert "event: response.output_text.done" in text
    assert "event: response.content_part.done" in text
    assert "event: response.output_item.done" in text
    assert "event: response.completed" in text

    # Parse the completed event to verify structure
    for line_block in text.split("\n\n"):
        if "event: response.completed" in line_block:
            data_line = [l for l in line_block.split("\n") if l.startswith("data: ")][0]
            completed = json.loads(data_line[6:])
            assert completed["response"]["status"] == "completed"
            assert "usage" in completed["response"]
            break


@patch("server.PrefixCache.lookup_full")
@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_streaming_full_cache_hit_does_not_crash(
        mock_os_read, mock_pipe, mock_lookup_full, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_lookup_full.return_value = (7, "cached_prompt.bin", 1)
    mock_tokenizer.decode.return_value = "cache hit reply"
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "say hello",
        "stream": True,
    })

    assert response.status_code == 200
    events = _responses_sse_events(response.text)
    completed = next(data for event, data in events if event == "response.completed")
    assert completed["response"]["status"] == "completed"
    assert completed["response"]["output_text"] == "cache hit reply"


@patch("server.PrefixCache.lookup_full")
@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_streaming_full_cache_hit_preserves_started_in_thinking(
        mock_os_read, mock_pipe, mock_lookup_full, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_lookup_full.return_value = (7, "cached_prompt.bin", 1)
    mock_tokenizer.apply_chat_template.return_value = "prompt<think>\n"

    def decode_side_effect(ids, *args, **kwargs):
        token_id = ids[0] if isinstance(ids, list) else ids
        return {
            10: "hidden reasoning",
            11: "</think>",
            12: "visible answer",
        }[token_id]

    mock_tokenizer.decode.side_effect = decode_side_effect
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11),
        struct.pack("<i", 12), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "say hello",
        "stream": True,
    })

    assert response.status_code == 200
    events = _responses_sse_events(response.text)
    deltas = [
        data["delta"]
        for event, data in events
        if event == "response.output_text.delta"
    ]
    assert "".join(deltas) == "visible answer"
    assert not any("</think>" in delta or "hidden reasoning" in delta for delta in deltas)
    completed = next(data for event, data in events if event == "response.completed")
    assert completed["response"]["output_text"] == "visible answer"


@patch("server.os.pipe")
@patch("server.os.read")
@pytest.mark.parametrize("decoded_chunks", [
    [
        "<tool_call><function=read_file><parameter=path>test",
        ".py</parameter></function></tool_call>",
    ],
    [
        "<function=read_file><parameter=path>test",
        ".py</parameter></function>",
    ],
    [
        '{"name":"read_file","arguments":{"path":"test',
        '.py"}}',
    ],
    [
        '<tool_code>{"name":"read_file","arguments":{"path":"test',
        '.py"}}</tool_code>',
    ],
])
def test_responses_streaming_function_call_lifecycle(mock_os_read, mock_pipe,
                                                     mock_tokenizer, app,
                                                     decoded_chunks):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.side_effect = decoded_chunks
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "read test.py",
        "tools": [{
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }],
        "stream": True,
    })

    assert response.status_code == 200
    events = _responses_sse_events(response.text)
    event_types = [event for event, _ in events]
    assert "response.output_item.added" in event_types
    assert "response.function_call_arguments.delta" in event_types
    assert "response.function_call_arguments.done" in event_types
    assert "response.output_item.done" in event_types
    assert "response.completed" in event_types
    assert "response.content_part.added" not in event_types
    assert "response.output_text.done" not in event_types

    added = next(data for event, data in events if event == "response.output_item.added")
    assert added["item"]["type"] == "function_call"
    assert "".join(
        data["delta"] for event, data in events
        if event == "response.function_call_arguments.delta"
    ) == '{"path":"test.py"}'
    done = next(data for event, data in events if event == "response.function_call_arguments.done")
    assert done["name"] == "read_file"
    assert done["arguments"] == '{"path":"test.py"}'
    completed = next(data for event, data in events if event == "response.completed")
    assert completed["response"]["output"][0]["type"] == "function_call"


@patch("server.os.pipe")
@patch("server.os.read")
@pytest.mark.parametrize(("decoded_chunks", "leaked_fragments"), [
    (
        [
            '<tool_call><function=write_file><parameter=path>file.txt</parameter></function></tool_call>',
        ],
        ["<tool_call>", "<function="],
    ),
    (
        [
            '{"name":"write_file","arguments":{"path":"file.txt"}}',
        ],
        ['{"name":"write_file"'],
    ),
    (
        [
            '<tool_code>{"name":"write_file","arguments":{"path":"file.txt"}}</tool_code>',
        ],
        ["<tool_code>", '{"name":"write_file"'],
    ),
])
def test_responses_streaming_tool_choice_failure_suppresses_terminal_function_events(
        mock_os_read, mock_pipe, mock_tokenizer, app, decoded_chunks,
        leaked_fragments):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.side_effect = decoded_chunks
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "read file.txt",
        "tools": [
            {"type": "function", "name": "read_file", "description": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
            {"type": "function", "name": "write_file", "description": "Write", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
        ],
        "tool_choice": {"type": "function", "name": "read_file"},
        "stream": True,
    })

    assert response.status_code == 200
    events = _responses_sse_events(response.text)
    event_types = [event for event, _ in events]
    assert "response.output_item.added" in event_types
    assert "response.function_call_arguments.delta" in event_types
    assert "response.failed" in event_types
    assert "error" in event_types
    assert "response.completed" not in event_types
    assert "response.function_call_arguments.done" not in event_types
    assert not any(
        event in {"response.content_part.added", "response.output_text.delta", "response.output_text.done"}
        for event, _ in events
    )
    assert not any(
        event == "response.output_item.done"
        and data["item"]["type"] == "function_call"
        for event, data in events
    )
    assert not any(
        event == "response.output_text.delta"
        and any(fragment in data["delta"] for fragment in leaked_fragments)
        for event, data in events
    )
    error = next(data for event, data in events if event == "error")
    assert error["error"]["param"] == "tool_choice"


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_streaming_completed_output_matches_output_indices(mock_os_read, mock_pipe,
                                                                      mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.side_effect = [
        "<tool_call><function=read_file><parameter=path>file.txt</parameter></function></tool_call>",
        "After tool",
    ]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "read file.txt",
        "tools": [{
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }],
        "stream": True,
    })

    assert response.status_code == 200
    events = _responses_sse_events(response.text)
    added = [
        (data["output_index"], data["item"]["type"])
        for event, data in events
        if event == "response.output_item.added"
    ]
    assert added == [(0, "function_call"), (1, "message")]
    completed = next(data for event, data in events if event == "response.completed")
    assert [item["type"] for item in completed["response"]["output"]] == [
        "function_call", "message",
    ]
    assert completed["response"]["output"][1]["content"][0]["text"] == "After tool"


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_streaming_ignores_stray_think_closers(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.side_effect = ["</think>", "</think>", "8"]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", 11),
        struct.pack("<i", 12), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "4+4=?",
        "stream": True,
    })

    assert response.status_code == 200
    text = response.text
    assert "</think>" not in text
    assert '"delta":"8"' in text or '"delta": "8"' in text


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_with_tools(mock_os_read, mock_pipe, mock_tokenizer, app):
    """POST /v1/responses with function tools maps correctly."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [{"type": "message", "role": "user", "content": "read file.txt"}],
        "tools": [
            {"type": "function", "name": "read_file",
             "description": "Read a file",
             "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}}}
        ],
        "instructions": "You are a coding assistant.",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"


@patch("server.compress_text_via_daemon")
@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_tool_requests_skip_compression(mock_os_read, mock_pipe, mock_compress,
                                                  mock_tokenizer, app_with_prefill):
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app_with_prefill)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "read file.txt",
        "tools": [{
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }],
        "tool_choice": "none",
    })

    assert response.status_code == 200
    mock_compress.assert_not_called()


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_tool_choice_none_disables_tool_parsing(mock_os_read, mock_pipe,
                                                          mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    raw_tool_text = (
        '<tool_call>'
        '<function=read_file><parameter=path>file.txt</parameter></function>'
        '</tool_call>'
    )
    mock_tokenizer.decode.return_value = raw_tool_text
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "read file.txt",
        "tools": [{
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }],
        "tool_choice": "none",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["output"][0]["type"] == "message"
    assert data["output_text"] == raw_tool_text
    assert "tools" not in mock_tokenizer.apply_chat_template.call_args_list[-1].kwargs


def test_responses_tool_choice_required_without_tools_is_rejected(app):
    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "hi",
        "tool_choice": "required",
    })

    assert response.status_code == 400
    data = response.json()
    assert data["error"]["param"] == "tool_choice"


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_specific_tool_choice_is_enforced(mock_os_read, mock_pipe,
                                                    mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.return_value = (
        '<tool_call>'
        '<function=write_file><parameter=path>file.txt</parameter></function>'
        '</tool_call>'
    )
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "read file.txt",
        "tools": [
            {"type": "function", "name": "read_file", "description": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
            {"type": "function", "name": "write_file", "description": "Write", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
        ],
        "tool_choice": {"type": "function", "name": "read_file"},
    })

    assert response.status_code == 400
    data = response.json()
    assert data["error"]["param"] == "tool_choice"
    tools_arg = mock_tokenizer.apply_chat_template.call_args_list[-1].kwargs["tools"]
    assert [tool["function"]["name"] for tool in tools_arg] == ["read_file"]


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_specific_tool_choice_rejects_extra_tool_call(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.return_value = (
        '<tool_call>'
        '<function=write_file><parameter=path>other.txt</parameter></function>'
        '</tool_call>'
        '<tool_call>'
        '<function=read_file><parameter=path>file.txt</parameter></function>'
        '</tool_call>'
    )
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "read file.txt",
        "tools": [
            {"type": "function", "name": "read_file", "description": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
            {"type": "function", "name": "write_file", "description": "Write", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
        ],
        "tool_choice": {"type": "function", "name": "read_file"},
    })

    assert response.status_code == 400
    data = response.json()
    assert data["error"]["param"] == "tool_choice"
    assert "does not allow other tool calls" in data["error"]["message"]


def test_responses_specific_tool_choice_rejects_unknown_name(app):
    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": "read file.txt",
        "tools": [{"type": "function", "name": "read_file"}],
        "tool_choice": {"type": "function", "name": "missing"},
    })

    assert response.status_code == 400
    data = response.json()
    assert data["error"]["param"] == "tool_choice"


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_object_tool_choice(mock_os_read, mock_pipe,
                                       mock_tokenizer, app):
    """POST /v1/responses with object-style tool_choice must not 422."""
    mock_pipe.return_value = (1, 2)
    mock_tokenizer.decode.return_value = (
        '<tool_call>'
        '<function=read_file><parameter=path>file.txt</parameter></function>'
        '</tool_call>'
    )
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [{"type": "message", "role": "user", "content": "read file.txt"}],
        "tools": [
            {"type": "function", "name": "read_file",
             "description": "Read a file",
             "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}}}
        ],
        "tool_choice": {"type": "function", "name": "read_file"},
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_function_call_output(mock_os_read, mock_pipe,
                                          mock_tokenizer, app):
    """Responses API maps function_call + function_call_output items."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [
            {"type": "message", "role": "user", "content": "read file.txt"},
            {"type": "function_call", "call_id": "call_abc123",
             "name": "read_file", "arguments": '{"path":"file.txt"}'},
            {"type": "function_call_output", "call_id": "call_abc123",
             "output": "file content here"},
        ],
    })

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"

    # Verify multi-turn message mapping: user + assistant(tool_call) + tool(output)
    calls = mock_tokenizer.apply_chat_template.call_args_list
    msgs = calls[-1][0][0]
    roles = [m["role"] for m in msgs]
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_replay_raw_tool_call_text(mock_os_read, mock_pipe,
                                             mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    raw_tool_text = (
        '<tool_call>'
        '<function=read_file><parameter=path>file.txt</parameter></function>'
        '</tool_call>'
    )
    mock_tokenizer.decode.side_effect = [raw_tool_text, "followup"]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", -1),
        struct.pack("<i", 11), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    first = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [{"type": "message", "role": "user", "content": "read file.txt"}],
    })
    assert first.status_code == 200
    first_output = first.json()["output"][0]

    second = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [
            {"type": "message", "role": "user", "content": "read file.txt"},
            first_output,
            {"type": "function_call_output", "call_id": first_output["call_id"], "output": "file body"},
            {"type": "message", "role": "user", "content": "what next?"},
        ],
    })
    assert second.status_code == 200

    msgs = mock_tokenizer.apply_chat_template.call_args_list[-1][0][0]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["content"] == raw_tool_text
    assert "tool_calls" not in assistant


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_replay_mixed_text_and_tool_output_uses_single_assistant_turn(
        mock_os_read, mock_pipe, mock_tokenizer, app):
    mock_pipe.return_value = (1, 2)
    raw_tool_text = (
        '<tool_call>'
        '<function=read_file><parameter=path>file.txt</parameter></function>'
        '</tool_call>'
        'After tool'
    )
    mock_tokenizer.decode.side_effect = [raw_tool_text, "followup"]
    mock_os_read.side_effect = [
        struct.pack("<i", 10), struct.pack("<i", -1),
        struct.pack("<i", 11), struct.pack("<i", -1),
    ]

    client = TestClient(app)
    first = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [{"type": "message", "role": "user", "content": "read file.txt"}],
    })
    assert first.status_code == 200
    first_output = first.json()["output"]
    function_call = next(item for item in first_output if item["type"] == "function_call")

    second = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [
            {"type": "message", "role": "user", "content": "read file.txt"},
            *first_output,
            {"type": "function_call_output", "call_id": function_call["call_id"], "output": "file body"},
            {"type": "message", "role": "user", "content": "what next?"},
        ],
    })
    assert second.status_code == 200

    msgs = mock_tokenizer.apply_chat_template.call_args_list[-1][0][0]
    assistants = [m for m in msgs if m["role"] == "assistant"]
    assert len(assistants) == 1
    assert assistants[0]["content"] == raw_tool_text
    assert "tool_calls" not in assistants[0]


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_developer_role_mapped_to_system(mock_os_read, mock_pipe,
                                                     mock_tokenizer, app):
    """Codex sends role=developer which maps to system."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "input": [
            {"type": "message", "role": "developer",
             "content": "You are helpful."},
            {"type": "message", "role": "user", "content": "hi"},
        ],
    })

    assert response.status_code == 200
    calls = mock_tokenizer.apply_chat_template.call_args_list
    msgs = calls[-1][0][0]
    assert msgs[0]["role"] == "system"


@patch("server.os.pipe")
@patch("server.os.read")
def test_responses_instructions_and_developer_merged(mock_os_read, mock_pipe,
                                                      mock_tokenizer, app):
    """Instructions + developer messages merge into one system message."""
    mock_pipe.return_value = (1, 2)
    mock_os_read.side_effect = [struct.pack("<i", 10), struct.pack("<i", -1)]

    client = TestClient(app)
    response = client.post("/v1/responses", json={
        "model": MODEL_NAME,
        "instructions": "Top-level instructions.",
        "input": [
            {"type": "message", "role": "developer",
             "content": "Developer context."},
            {"type": "message", "role": "user", "content": "hi"},
        ],
    })

    assert response.status_code == 200
    calls = mock_tokenizer.apply_chat_template.call_args_list
    msgs = calls[-1][0][0]
    # Should be exactly one system message containing both
    system_msgs = [m for m in msgs if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "Top-level instructions." in system_msgs[0]["content"]
    assert "Developer context." in system_msgs[0]["content"]
