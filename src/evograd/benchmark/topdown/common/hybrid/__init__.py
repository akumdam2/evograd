"""Reference mathematics shared by the hybrid Gated-DeltaNet / gated-attention decoders.

The gated delta rule, the gated RMSNorm, the ``(1 + w)`` RMSNorm, gated attention
and the partial RoPE apply, as Transformers' hybrid decoders (Qwen3.5) run them
(checked on Transformers 5.16.1). A model owns its own *tasks* -- keys, cases,
tolerances, provenance -- but the definition of each computation is written
here once, so two declarations cannot drift into computing subtly different
things while claiming to measure the same boundary.

Where a production spelling must reach Transformers' own function -- the torch
chunk delta rule, ``apply_rotary_pos_emb`` -- it takes the ``modeling`` module
as an argument (``modeling_qwen3_5``): each modeling module is its own object
with its own import-time kernel dispatch, and nothing here may pick one for the
caller. Those functions are reached through
``__wrapped__`` so an importable FLA cannot redirect them.

Nothing here imports a workload package or reads a snapshot.
"""
