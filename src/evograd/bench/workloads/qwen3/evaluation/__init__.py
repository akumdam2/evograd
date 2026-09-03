"""How carefully a measurement is checked, independent of what it measures.

The tier axis is orthogonal to the level axis in :mod:`..levels`: a tier says
what is timed and how correctness is established, not which operator is under
test. Only tier 3 -- whole-model, drop-in replacement -- needs Qwen-specific
machinery, and it lives in :mod:`.tier3`.
"""
