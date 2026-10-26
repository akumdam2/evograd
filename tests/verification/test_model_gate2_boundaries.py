"""Model-scope Gate 2 runs without per-call boundary replay, and a cached policy
calibrated under the other setting is never reused."""

import unittest
from unittest import mock

from evograd.verification.integrations import gates


class ModelGate2Boundaries(unittest.TestCase):
    def test_model_scope_gate2_runs_without_boundary_replay(self):
        self.assertFalse(gates.GATE2_MODEL_BOUNDARIES)
        self.assertIs(gates._gate2_model_rule()["boundaries"], False)

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
