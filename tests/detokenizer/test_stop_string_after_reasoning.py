# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stop strings apply to the response, not the reasoning.

Claude Code's auto-mode classifier sends ``stop_sequences: ["</block>"]``; when
thinking is on, a model that writes ``</block>`` while reasoning stopped inside
``<think>`` with an empty answer (vllm RFC #58647 item 3).
"""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer

START, END = 1000, 1001  # <think>, </think>
MARKERS = ([START], [END])


class _CharDetokenizer(BaseIncrementalDetokenizer):
    """Token id = character code; the markers decode to their tag text."""

    def decode_next(self, next_token_id: int) -> str:
        return {START: "<think>", END: "</think>"}.get(next_token_id, chr(next_token_id))


def _request(prompt: list[int], stop: list[str]) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="test",
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=SamplingParams(stop=stop),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _ids(text: str) -> list[int]:
    return [ord(c) for c in text]


def _run(prompt, output, stop, markers=MARKERS, step=1):
    det = _CharDetokenizer(_request(prompt, stop), markers)
    for i in range(0, len(output), step):
        matched = det.update(output[i : i + step], stop_terminated=False)
        if matched is not None:
            return matched, det.output_text
    return None, det.output_text


@pytest.mark.parametrize("step", [1, 3, 64])  # 3, 64: several tokens per step (MTP)
def test_stop_string_inside_reasoning_is_ignored(step):
    output = _ids("the verdict goes in <block>x</block> form\n") + [END]
    output += _ids("\n\n<block>no</block> trailing")
    matched, text = _run(_ids("prompt") + [START], output, ["</block>"], step=step)
    assert matched == "</block>"
    assert text.endswith("</think>\n\n<block>no")
    assert "<block>x</block> form" in text  # reasoning kept whole


def test_stop_string_ends_response_when_prompt_is_not_reasoning():
    # Thinking disabled: the template closes the reasoning in the prompt.
    prompt = _ids("p") + [START] + _ids("\n\n") + [END]
    matched, text = _run(prompt, _ids("<block>no</block> more"), ["</block>"])
    assert (matched, text) == ("</block>", "<block>no")


def test_without_reasoning_parser_stop_strings_apply_everywhere():
    output = _ids("x</block>")
    matched, text = _run(_ids("p") + [START], output, ["</block>"], markers=None)
    assert (matched, text) == ("</block>", "x")


def test_match_may_not_span_the_reasoning_end():
    # "k>" ends the reasoning, "</bloc" + "k>" must not count as "</block>".
    output = _ids("</bloc") + [END] + _ids("k> answer")
    matched, _ = _run(_ids("p") + [START], output, ["</block>"])
    assert matched is None


def test_reasoning_never_closed_means_no_stop():
    matched, text = _run(_ids("p") + [START], _ids("a</block>b"), ["</block>"])
    assert matched is None
    assert text == "a</block>b"
