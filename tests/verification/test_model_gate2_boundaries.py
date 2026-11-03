"""Model-scope Gate 2 runs without per-call boundary replay, and a cached policy
calibrated under the other setting is never reused."""

import unittest
from unittest import mock

from evograd.verification.integrations import gates


class ModelGate2Boundaries(unittest.TestCase):
    def test_model_scope_gate2_runs_without_boundary_replay(self):
        self.assertFalse(gates.GATE2_MODEL_BOUNDARIES)
        model = mock.Mock(gate2_model_gamma=None)
        self.assertIs(gates._gate2_model_rule(model)["boundaries"], False)

    def test_a_model_may_set_its_own_model_scope_and_gate3_gammas(self):
        self.assertEqual(gates.model_gamma(mock.Mock(gate2_model_gamma=None)), gates.GATE2_MODEL_GAMMA)
        self.assertEqual(gates._gate2_model_rule(mock.Mock(gate2_model_gamma=3.0))["gamma"], 3.0)
        self.assertEqual(gates.gate3_gamma(mock.Mock(gate3_gamma=None)), gates.GAMMA)
        self.assertEqual(gates.gate3_gamma(mock.Mock(gate3_gamma=3.0)), 3.0)

    def _calibrate(self, cached_boundaries, requested):
        cached = mock.Mock()
        cached.policy.boundaries = cached_boundaries
        fresh = mock.Mock(policy=None)
        with mock.patch.object(gates, "_cached_policy", return_value=cached), \
                mock.patch("evograd.verification.gate2.calibrate", return_value=fresh) as calibrate:
            result = gates._calibrate(object(), mock.Mock(patch_set=("site",)), [], None,
                                      boundaries=requested)
        return result, cached, fresh, calibrate

    def test_a_cached_policy_with_the_other_boundary_setting_is_recalibrated(self):
        result, _, fresh, calibrate = self._calibrate(cached_boundaries=True, requested=False)
        calibrate.assert_called_once()
        self.assertIs(result, fresh)

    def test_a_cached_policy_with_the_same_boundary_setting_is_reused(self):
        result, cached, _, calibrate = self._calibrate(cached_boundaries=False, requested=False)
        calibrate.assert_not_called()
        self.assertIs(result, cached)


if __name__ == "__main__":
    unittest.main()
