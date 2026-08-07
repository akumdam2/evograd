"""Scoring-policy math, checked against hand-computed values.

The formulas are ported from the old repo's ``_score_from_aggregate``; these
tests pin the arithmetic so refactors can't silently change scores."""

import unittest

from evograd.evolve.scoring import (
    POLICIES,
    ScoringPolicy,
    geomean,
    get_policy,
    score_from_aggregate,
    weighted_geomean,
)

AGGREGATE = {
    "speedup_vs_baseline_backward": 2.0,
    "speedup_vs_baseline_full_step": 1.5,
    "saved_memory_ratio": 0.5,
    "geomean_speedup_vs_baseline_backward": 1.8,
    "geomean_speedup_vs_baseline_full_step": 1.4,
    "weighted_geomean_speedup_vs_baseline_backward": 1.7,
    "weighted_geomean_speedup_vs_baseline_full_step": 1.3,
    "weighted_geomean_min_speedup_per_case": 1.25,
    "worst_case_min_speedup": 0.9,
}

PENALTY = 1.0 + 0.05 * 0.5  # 1.025


class TestScoreFromAggregate(unittest.TestCase):
    def test_speed_only(self):
        score, _ = score_from_aggregate(AGGREGATE, get_policy("speed"))
        self.assertAlmostEqual(score, 2.0)

    def test_speed_memory(self):
        # weighted = 0.5*2.0 + 0.5*1.5 = 1.75; / 1.025
        score, details = score_from_aggregate(AGGREGATE, get_policy("speed_memory"))
        self.assertAlmostEqual(score, 1.75 / PENALTY)
        self.assertAlmostEqual(details["memory_penalty_factor"], PENALTY)

    def test_speed_memory_min(self):
        score, _ = score_from_aggregate(AGGREGATE, get_policy("speed_memory_min"))
        self.assertAlmostEqual(score, 1.5 / PENALTY)

    def test_speed_memory_min_geomean(self):
        # min(1.8, 1.4) = 1.4
        score, _ = score_from_aggregate(AGGREGATE, get_policy("speed_memory_min_geomean"))
        self.assertAlmostEqual(score, 1.4 / PENALTY)

    def test_speed_memory_min_weighted_geomean_applies_worst_case_guard(self):
        # 1.25 * min(1, 0.9) / 1.025
        score, details = score_from_aggregate(
            AGGREGATE, get_policy("speed_memory_min_weighted_geomean")
        )
        self.assertAlmostEqual(score, 1.25 * 0.9 / PENALTY)
        self.assertAlmostEqual(details["worst_case_guard_factor"], 0.9)

    def test_worst_case_guard_disabled(self):
        policy = get_policy("speed_memory_min_weighted_geomean", worst_case_guard=False)
        score, _ = score_from_aggregate(AGGREGATE, policy)
        self.assertAlmostEqual(score, 1.25 / PENALTY)

    def test_full_step_weight_override(self):
        policy = get_policy("speed_memory", full_step_weight=0.0)
        score, _ = score_from_aggregate(AGGREGATE, policy)
        self.assertAlmostEqual(score, 2.0 / PENALTY)  # backward speedup only

    def test_all_policies_registered(self):
        self.assertEqual(
            sorted(POLICIES),
            [
                "speed",
                "speed_memory",
                "speed_memory_min",
                "speed_memory_min_geomean",
                "speed_memory_min_weighted_geomean",
            ],
        )
        for policy in POLICIES.values():
            self.assertIsInstance(policy, ScoringPolicy)


class TestGeomeans(unittest.TestCase):
    def test_geomean(self):
        self.assertAlmostEqual(geomean([2.0, 8.0]), 4.0)

    def test_geomean_skips_nonpositive(self):
        self.assertAlmostEqual(geomean([2.0, 8.0, 0.0, -1.0]), 4.0)

    def test_weighted_geomean_uniform_matches_geomean(self):
        self.assertAlmostEqual(weighted_geomean([2.0, 8.0], None), 4.0)

    def test_weighted_geomean_weights(self):
        # weights (1, 3): exp((ln2 + 3 ln8)/4) = 2^(10/4)
        self.assertAlmostEqual(weighted_geomean([2.0, 8.0], [1.0, 3.0]), 2 ** 2.5)


class TestShapeSpecialization(unittest.TestCase):
    @staticmethod
    def _case(rows, hidden, speedup):
        return {
            "dims": {"rows": rows, "hidden": hidden},
            "speedup_vs_baseline_full_step": speedup,
        }

    def test_large_shape_specialist_scores_above_one(self):
        from evograd.evolve.scoring import shape_specialization_from_cases

        cases = [
            self._case(1, 768, 1.0),
            self._case(8, 1024, 1.0),
            self._case(256, 4096, 2.0),
            self._case(4096, 8192, 2.0),
        ]
        self.assertAlmostEqual(shape_specialization_from_cases(cases), 2.0)

    def test_small_shape_specialist_scores_below_one(self):
        from evograd.evolve.scoring import shape_specialization_from_cases

        cases = [
            self._case(1, 768, 3.0),
            self._case(8, 1024, 3.0),
            self._case(256, 4096, 1.5),
            self._case(4096, 8192, 1.5),
        ]
        self.assertAlmostEqual(shape_specialization_from_cases(cases), 0.5)

    def test_degenerate_inputs_are_neutral(self):
        from evograd.evolve.scoring import shape_specialization_from_cases

        self.assertEqual(shape_specialization_from_cases([]), 1.0)
        self.assertEqual(
            shape_specialization_from_cases([self._case(4, 4, 2.0)]), 1.0
        )
        # All cases the same size: no small/large split exists.
        same = [self._case(4, 4, s) for s in (1.0, 2.0, 3.0)]
        self.assertEqual(shape_specialization_from_cases(same), 1.0)
        # Missing/invalid speedups are skipped.
        broken = [{"dims": {"r": 2}}, {"dims": {"r": 4096}, "speedup_vs_baseline_full_step": 0.0}]
        self.assertEqual(shape_specialization_from_cases(broken), 1.0)

    def test_registered_as_feature_dimension(self):
        from evograd.evolve.scoring import (
            CUSTOM_FEATURE_DIMENSION_DEFAULTS,
            validate_feature_dimensions,
        )

        self.assertEqual(CUSTOM_FEATURE_DIMENSION_DEFAULTS["shape_specialization"], 1.0)
        validate_feature_dimensions(("complexity", "shape_specialization"))


if __name__ == "__main__":
    unittest.main()
