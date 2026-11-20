"""Model-scope Gate 2's and Gate 3's factors: the protocol's, or a model's own."""

import unittest
from unittest import mock

from evograd.verification.integrations import gates


class ModelGammas(unittest.TestCase):
    def test_a_model_may_set_its_own_model_scope_and_gate3_gammas(self):
        self.assertEqual(gates.model_gamma(mock.Mock(gate2_model_gamma=None)), gates.GATE2_MODEL_GAMMA)
        self.assertEqual(gates._gate2_model_rule(mock.Mock(gate2_model_gamma=3.0))["gamma"], 3.0)
        self.assertEqual(gates.gate3_gamma(mock.Mock(gate3_gamma=None)), gates.GAMMA)
        self.assertEqual(gates.gate3_gamma(mock.Mock(gate3_gamma=3.0)), 3.0)

    def test_the_model_rule_names_its_reference_and_trusted_control(self):
        rule = gates._gate2_model_rule(mock.Mock(gate2_model_gamma=None))
        self.assertEqual((rule["reference"], rule["metric"], rule["gamma"]),
                         ("native eager", "E2", gates.GATE2_MODEL_GAMMA))


if __name__ == "__main__":
    unittest.main()
