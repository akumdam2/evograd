"""Where the Qwen shapes came from.

One instrumented run of the canonical step records every operator invocation it
makes; the records are aggregated into a manifest, and the manifest is frozen
into ``snapshot.json``. Everything downstream -- the level-2 tolerances, the
tier-3 invocation counts -- is derived from that snapshot rather than from a
number somebody typed.

``snapshot.json`` is tracked provenance, not a run artifact: it is versioned
with the code and carries a semantic hash the tests check.
"""
