# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Added-token text inside messages stays text (vllm/renderers/literal_tokens.py)."""

import os

import pytest

from vllm.renderers import literal_tokens
from vllm.tokenizers import get_tokenizer

MODEL = os.environ.get("VLLM_TEST_LITERAL_TOKENIZER", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL)


def render(tokenizer, conversation, tools=None):
    """Prompt ids the renderer would produce, and the plain rendered text."""
    plain = tokenizer.apply_chat_template(
        conversation, tools=tools, tokenize=False, add_generation_prompt=True
    )
    escaped = literal_tokens.escape(conversation, tools, tokenizer)
    if escaped is None:
        return tokenizer.encode(plain, add_special_tokens=False), plain, None
    conv, esc_tools, literals = escaped
    text = tokenizer.apply_chat_template(
        conv, tools=esc_tools, tokenize=False, add_generation_prompt=True
    )
    ids = literal_tokens.encode(tokenizer, text, literals)
    assert ids is not None
    assert literal_tokens.restore(text, literals) == plain
    return ids, plain, literals


def test_plain_request_unchanged(tokenizer):
    conv = [{"role": "user", "content": "hello <b>world</b>"}]
    assert literal_tokens.escape(conv, None, tokenizer) is None


def test_tool_result_with_im_end(tokenizer):
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    conv = [
        {"role": "user", "content": "read the doc"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": {"path": "a.md"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "P(<|im_end|>) is 1.6%"},
    ]
    ids, plain, literals = render(tokenizer, conv)
    assert literals == ["<|im_end|>"]
    # One <|im_end|> per message, none from the tool result text.
    plain_ids = tokenizer.encode(plain, add_special_tokens=False)
    assert plain_ids.count(im_end) == ids.count(im_end) + 1
    # Same text, now spelled with ordinary tokens.
    assert tokenizer.decode(ids) == plain


def test_think_tags_in_user_text_are_literal(tokenizer):
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    conv = [{"role": "user", "content": "what does </think> do?"}]
    ids, plain, literals = render(tokenizer, conv)
    assert literals == ["</think>"]
    assert think_end not in ids
    assert tokenizer.decode(ids) == plain


def test_assistant_reasoning_tags_left_to_template(tokenizer):
    conv = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>\nplan\n</think>\n\nhello"},
        {"role": "user", "content": "again"},
    ]
    assert literal_tokens.escape(conv, None, tokenizer) is None


def test_tool_definitions_and_arguments(tokenizer):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Writes a file; never write <tool_call> by hand",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    conv = [{"role": "user", "content": "go"}]
    ids, plain, literals = render(tokenizer, conv, tools)
    assert literals == ["<tool_call>"]
    assert tokenizer.decode(ids) == plain


def test_literal_split_matches_plain_text(tokenizer):
    # A true special token split by the tokenizer itself must match ours.
    for text in (" <|im_end|>", "<|im_end|>", "\n<|endoftext|>"):
        assert literal_tokens._encode_literal(
            tokenizer, text
        ) == tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)


def test_broken_sentinel_falls_back(tokenizer):
    literals = ["<|im_end|>"]
    assert literal_tokens.encode(tokenizer, "a \U000f0000x b", literals) is None
    assert literal_tokens.encode(tokenizer, 'a "\\U000f00000" b', literals) is None


def test_private_use_text_is_not_escaped(tokenizer):
    conv = [{"role": "user", "content": "\U000f0000 and <|im_end|>"}]
    assert literal_tokens.escape(conv, None, tokenizer) is None
