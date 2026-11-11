"""SDPA references trace under symbolic shapes.

`is_causal=tokens > 1` is a SymBool once shapes are symbolic (pipeline A's AtenIR
extraction, a recompiled torch.compile control), and SDPA's bool arguments refuse
it: OLMoE's attention lost its compile control and Qwen3.5's gated attention its
pipeline-A seed that way (2026-10-27). Qwen3.5's attention block holds the same call.
"""

import unittest

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from evograd.benchmark import get_task
from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.attention.reference import (
    attention_forward_production,
)


class SymbolicShapes(unittest.TestCase):
    def test_qwen3_5_attention_block_traces_with_symbolic_shapes(self):
        from evograd.opdecl.oracle import resolve_runtime_forward

        production = resolve_runtime_forward(get_task("qwen3_5_attention_block"))
        forward_production = lambda *args: production(*args)  # noqa: E731 (eps at its default)
        args = (torch.randn(1, 8, 32), torch.randn(128, 32), torch.randn(32, 32),
                torch.randn(32, 32), torch.randn(16), torch.randn(16), torch.randn(32, 64),
                torch.randn(1, 8, 4), torch.randn(1, 8, 4))
        graph = make_fx(forward_production, tracing_mode="symbolic")(*args)
        torch.testing.assert_close(graph(*args), forward_production(*args))

    def test_olmoe_attention_traces_with_symbolic_shapes(self):
        args = (torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16),
                torch.randn(32, 64))
        graph = make_fx(attention_forward_production, tracing_mode="symbolic")(*args)
        torch.testing.assert_close(graph(*args), attention_forward_production(*args))


if __name__ == "__main__":
    unittest.main()
