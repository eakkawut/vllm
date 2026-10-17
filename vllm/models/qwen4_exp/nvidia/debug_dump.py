# SPDX-License-Identifier: Apache-2.0
"""Debug only (branch debug/mixed-batch): record the hidden state of one token position after each part of every
decoder layer, to find the first layer where a prompt prefilled alone and the same prompt prefilled next to other
requests stop matching. Needs --enforce-eager (side effects are skipped while torch.compile traces).

VLLM_DEBUG_DUMP_POS   position to record (the last prompt token of the request under test), -1 = off
VLLM_DEBUG_DUMP_PATH  JSONL output; each record also saves the full row to <path>.d/<seq>.pt
"""

import itertools
import json
import os

import torch

POS = int(os.environ.get("VLLM_DEBUG_DUMP_POS", "-1"))
PATH = os.environ.get("VLLM_DEBUG_DUMP_PATH", "/tmp/vllm-dump.jsonl")
_seq = itertools.count()


def enabled() -> bool:
    return POS >= 0 and not torch.compiler.is_compiling()


def dump(tag: str, layer: int, stage: str, positions: torch.Tensor, t: torch.Tensor | None) -> None:
    if t is None or not enabled():
        return
    pos = positions[0] if positions.dim() == 2 else positions
    rows = (pos == POS).nonzero().flatten().tolist()
    if not rows:
        return
    os.makedirs(PATH + ".d", exist_ok=True)
    with open(PATH, "a") as f:
        for r in rows:
            if r >= t.shape[0]:
                continue
            v = t[r].detach().float()
            n = next(_seq)
            torch.save(v.cpu(), f"{PATH}.d/{n}.pt")
            f.write(json.dumps({"seq": n, "tag": tag, "layer": layer, "stage": stage, "row": r,
                                "batch_tokens": int(pos.shape[0]), "norm": v.norm().item(),
                                "head": v.flatten()[:4].tolist()}) + "\n")


_step = itertools.count()


def _summary(md) -> dict:
    out = {}
    for k, v in vars(md).items():
        if isinstance(v, torch.Tensor):
            out[k] = v.tolist() if v.numel() <= 64 else f"tensor{tuple(v.shape)}"
        elif isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = v
        else:
            out[k] = type(v).__name__
    return out


def patch_builders() -> None:
    """Log each GDN / PLE short-conv metadata build (state indices, query_start_loc, prefill/decode counts)."""
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    from .ple_attn import PLEAttentionMetadataBuilder

    def wrap(cls, kind):
        orig = cls.build

        def build(self, *a, **kw):
            md = orig(self, *a, **kw)
            try:
                with open(PATH + ".meta", "a") as f:
                    f.write(json.dumps({"step": next(_step), "kind": kind, **_summary(md)}, default=str) + "\n")
            except Exception as e:  # never break serving for a debug log
                print("debug_dump meta:", e)
            return md

        cls.build = build

    wrap(GDNAttentionMetadataBuilder, "gdn")
    wrap(PLEAttentionMetadataBuilder, "ple")


if POS >= 0:
    patch_builders()
