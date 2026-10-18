# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for mid-stream error handling in the Anthropic messages converter.

Regression for issue #46028: an upstream engine failure is delivered to
``message_stream_converter`` as an ``ErrorResponse`` payload
(``{"error": {...}}``), not a ``ChatCompletionStreamResponse``. The converter
must forward the real upstream message as an Anthropic ``error`` event instead
of masking it with a pydantic schema-validation error.

These tests are CPU-only and do not start a server.
"""

import json

import pytest

from vllm.entrypoints.anthropic.protocol import anthropic_error_type
from vllm.entrypoints.anthropic.serving import (
    AnthropicServingMessages,
    parse_streaming_error_chunk,
)
from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse


def _error_chunk(message: str) -> str:
    """Build the SSE line an upstream engine emits for a mid-stream failure.

    Mirrors ``create_streaming_error_response`` in the OpenAI serving layer:
    ``json.dumps(ErrorResponse(...).model_dump())`` wrapped in an SSE ``data:``
    frame.
    """
    payload = ErrorResponse(
        error=ErrorInfo(message=message, type="InternalServerError", code=500)
    )
    return f"data: {json.dumps(payload.model_dump())}\n\n"


def test_parse_streaming_error_chunk_detects_error_payload():
    err = parse_streaming_error_chunk(
        '{"error": {"message": "Internal Server Error", '
        '"type": "InternalServerError", "code": 500}}'
    )
    assert err is not None
    assert err.error.message == "Internal Server Error"


def test_parse_streaming_error_chunk_ignores_normal_chunk():
    # A regular (non-error) streaming chunk has no top-level ``error`` field.
    assert (
        parse_streaming_error_chunk(
            '{"id": "x", "object": "chat.completion.chunk", '
            '"created": 0, "model": "m", "choices": []}'
        )
        is None
    )


def test_parse_streaming_error_chunk_ignores_malformed_json():
    assert parse_streaming_error_chunk("not json") is None
    assert parse_streaming_error_chunk('{"foo": 1}') is None


@pytest.mark.asyncio
async def test_converter_forwards_upstream_error_message():
    async def upstream():
        yield _error_chunk("Internal Server Error")
        yield "data: [DONE]\n\n"

    # The error path never touches initialized instance state, so a bare
    # instance is sufficient to exercise the converter.
    serving = object.__new__(AnthropicServingMessages)
    events = [event async for event in serving.message_stream_converter(upstream())]

    joined = "".join(events)
    assert "event: error" in joined
    assert "Internal Server Error" in joined
    # The real message must not be masked by a schema-validation error.
    assert "validation error" not in joined.lower()

    # The error event must carry the upstream message in the Anthropic shape.
    error_lines = [e for e in events if "event: error" in e]
    assert len(error_lines) == 1
    data_str = error_lines[0].split("data:", 1)[1].strip()
    parsed = json.loads(data_str)
    assert parsed["type"] == "error"
    assert parsed["error"]["message"] == "Internal Server Error"
    assert parsed["error"]["type"] == "api_error"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (404, "not_found_error"),
        (413, "request_too_large"),
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (503, "overloaded_error"),
    ],
)
def test_anthropic_error_type(code, expected):
    assert anthropic_error_type(code) == expected


@pytest.mark.asyncio
async def test_converter_uses_anthropic_type_for_upstream_status():
    """Clients retry on Anthropic's types (overloaded_error), so the upstream
    status picks the type instead of vLLM's own name."""

    async def upstream():
        payload = ErrorResponse(
            error=ErrorInfo(message="busy", type="ServiceUnavailable", code=503)
        )
        yield f"data: {json.dumps(payload.model_dump())}\n\n"

    serving = object.__new__(AnthropicServingMessages)
    events = [event async for event in serving.message_stream_converter(upstream())]
    error = json.loads(events[-1].split("data:", 1)[1])["error"]
    assert error == {"type": "overloaded_error", "message": "busy"}
