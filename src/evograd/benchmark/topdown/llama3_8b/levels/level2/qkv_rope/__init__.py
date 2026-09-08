"""Q/K/V projection and rotary embedding, as one boundary.

``task`` is the contract, ``reference`` defines it, and ``capture`` derives this
site's case from a captured layer. Unlike Qwen3's counterpart there is no
per-head query/key RMSNorm here: LlamaAttention projects and rotates, and that
difference is why this is its own declaration rather than a second suite on
Qwen3's.
"""

from .task import op

__all__ = ["op"]
