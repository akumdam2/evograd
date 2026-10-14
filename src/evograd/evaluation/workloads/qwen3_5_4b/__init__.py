"""Qwen3.5-4B, on the evaluation side: verdicts for its Level-1/2/3 cases.

A whole-model harvest and captured layers 14 (Gated DeltaNet) and 15 (gated
full attention): the descriptor names one representative layer per block kind,
and a site's harvested invocation is taken from the captured layer when the
artifact is present, else from a config-derived block
(``levels.level2.<site>.capture``).
"""
