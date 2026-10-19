# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Message text that spells an added token stays text.

The rendered chat prompt is tokenized as one string, so text inside a message
that happens to spell an added token (``<|im_end|>``, ``</think>``,
``<tool_call>``, ...) became that control token: a file the model read ended
its tool result half way, a web page could open a new turn of its own, and a
quoted earlier answer closed the conversation. Before the chat template runs,
each such occurrence in the messages and tools is swapped for a sentinel; the
rendered prompt is then tokenized in segments: the template's own text as
before, each sentinel as plain text through the BPE model alone (no added-token
matching), as the tokenizer would have split it had the token not existed.

Requests without such text are rendered exactly as before. Assistant content
keeps its reasoning tags (``<think>`` / ``</think>``), which chat templates
parse to split reasoning from the answer.
"""

import re
from collections.abc import Sequence
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Plane 15 private-use code points: never produced by tokenizers or templates.
_OPEN = "\U000f0000"
_CLOSE = "\U000f0001"
_SENTINEL = re.compile(_OPEN + r"(\d+)" + _CLOSE)
# A sentinel the template escaped or cut would leave one of these behind.
_BROKEN = (_OPEN, _CLOSE, "\\U000f000", "\\udb80\\udc0")
_REASONING_TAGS = ("<think>", "</think>")
_SKIP_KEYS = frozenset({"role", "type", "id", "tool_call_id"})

_patterns: dict[int, tuple[re.Pattern[str] | None, re.Pattern[str] | None]] = {}


def _get_patterns(tokenizer: Any) -> tuple[re.Pattern[str] | None, re.Pattern[str] | None]:
    """(every added token, every added token except the reasoning tags)."""
    key = id(tokenizer)
    if key not in _patterns:
        try:
            added = [t for t in tokenizer.get_added_vocab() if t]
        except Exception:
            added = []
        # Longest first so `</think>` wins over a shorter token it contains.
        added.sort(key=len, reverse=True)

        def build(tokens: list[str]) -> re.Pattern[str] | None:
            return re.compile("|".join(map(re.escape, tokens))) if tokens else None

        _patterns[key] = (
            build(added),
            build([t for t in added if t not in _REASONING_TAGS]),
        )
    return _patterns[key]


class _Escaper:
    def __init__(self, full: re.Pattern[str], no_reasoning: re.Pattern[str] | None):
        self.full = full
        self.no_reasoning = no_reasoning
        self.literals: list[str] = []
        self._index: dict[str, int] = {}
        self.clash = False

    def _sub(self, m: re.Match[str]) -> str:
        text = m.group(0)
        i = self._index.get(text)
        if i is None:
            i = self._index[text] = len(self.literals)
            self.literals.append(text)
        return f"{_OPEN}{i}{_CLOSE}"

    def text(self, s: str, pattern: re.Pattern[str] | None) -> str:
        if _OPEN in s or _CLOSE in s:
            self.clash = True
        return pattern.sub(self._sub, s) if pattern is not None else s

    def value(self, x: Any, pattern: re.Pattern[str] | None) -> Any:
        if isinstance(x, str):
            return self.text(x, pattern)
        if isinstance(x, list):
            return [self.value(v, pattern) for v in x]
        if isinstance(x, tuple):
            return tuple(self.value(v, pattern) for v in x)
        if isinstance(x, dict):
            return {
                k: v if k in _SKIP_KEYS else self.value(v, pattern)
                for k, v in x.items()
            }
        return x

    def message(self, msg: Any) -> Any:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            return self.value(msg, self.full)
        return {
            k: v
            if k in _SKIP_KEYS
            else self.value(v, self.no_reasoning if k == "content" else self.full)
            for k, v in msg.items()
        }


def escape(
    conversation: Sequence[Any],
    tools: Any,
    tokenizer: Any,
) -> tuple[list[Any], Any, list[str]] | None:
    """Swap added-token text in messages and tools for sentinels.

    Returns (conversation, tools, literals), or None when there is nothing to
    escape (the request renders exactly as before)."""
    full, no_reasoning = _get_patterns(tokenizer)
    if full is None:
        return None
    esc = _Escaper(full, no_reasoning)
    new_conversation = [esc.message(m) for m in conversation]
    new_tools = esc.value(tools, full) if tools is not None else None
    if not esc.literals:
        return None
    if esc.clash:
        logger.warning_once(
            "A message contains the private-use characters used to keep "
            "added-token text literal; rendering it unescaped."
        )
        return None
    return new_conversation, new_tools, esc.literals


def _encode_literal(tokenizer: Any, text: str) -> list[int]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or backend.pre_tokenizer is None:
        # Slow tokenizers: only true special tokens can be split this way.
        return tokenizer.encode(
            text, add_special_tokens=False, split_special_tokens=True
        )
    if backend.normalizer is not None:
        text = backend.normalizer.normalize_str(text)
    return [
        token.id
        for piece, _ in backend.pre_tokenizer.pre_tokenize_str(text)
        for token in backend.model.tokenize(piece)
    ]


def restore(prompt: str, literals: Sequence[str]) -> str:
    return _SENTINEL.sub(lambda m: literals[int(m.group(1))], prompt)


def encode(tokenizer: Any, prompt: str, literals: Sequence[str]) -> list[int] | None:
    """Token ids of a rendered prompt holding sentinels (added-token text kept
    literal), or None if the template did not keep every sentinel intact."""
    rest = _SENTINEL.sub("", prompt)
    if any(b in rest for b in _BROKEN):
        logger.warning_once(
            "The chat template altered the markers that keep added-token "
            "text literal; rendering this request unescaped."
        )
        return None

    ids: list[int] = []
    pos = 0
    for m in _SENTINEL.finditer(prompt):
        segment = prompt[pos : m.start()]
        literal = literals[int(m.group(1))]
        # Byte-level pre-tokenizers attach the last space before a word to
        # it; hand that space to the literal so its split matches plain text.
        if segment.endswith(" "):
            segment, literal = segment[:-1], " " + literal
        if segment:
            ids.extend(tokenizer.encode(segment, add_special_tokens=False))
        ids.extend(_encode_literal(tokenizer, literal))
        pos = m.end()
    if pos < len(prompt):
        ids.extend(tokenizer.encode(prompt[pos:], add_special_tokens=False))
    return ids
