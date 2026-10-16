"""OLMoE-1B-7B, on the evaluation side: verdicts for its Level-1/2/3 cases.

A whole-model harvest and a captured layer 8: the descriptor names the
representative layer, and a site's harvested invocation is taken from the
captured layer when the artifact is present, else from a config-derived layer
(``levels.level2.<site>.capture``).
"""
