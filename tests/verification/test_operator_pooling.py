"""Operator Gate 2 pools each result's trusted error over every payload of the case."""

import unittest
import warnings

import torch

from evograd.verification.common.execution import Sample
from evograd.verification.common.reports import FAILED, PASSED
from evograd.verification.gate2.adapters.operator import OperatorScope
from evograd.verification.gate2.operator import verify_operator
from tests.verification import _fixtures as F

warnings.filterwarnings("ignore")
PERTURB = "evograd_test_input_dependent_perturbation"


def _perturbing_backend(gm, example_inputs):
    """A test 'compiler' whose result differs from eager by an amount that
    depends on the call's first input: trusted-control error that differs
    between samples. Test-only; never a verification control."""
    def run(*args):
        outs = gm(*args)
        x = next(a for a in args if torch.is_tensor(a) and a.is_floating_point())
        scale = 1.0 + 1e-2 * float(x.detach().abs().mean()) ** 2
        if isinstance(outs, (tuple, list)):
            return type(outs)(o * scale if torch.is_tensor(o) and o.is_floating_point() else o
                              for o in outs)
        return outs * scale
    return run


try:
    torch._dynamo.register_backend(_perturbing_backend, name=PERTURB)
except Exception:  # already registered in this process
    pass


def codes(report):
    return sorted({f.code for f in report.all_findings()})


class _CallScope(OperatorScope):
    """Samples recorded at two call sites; layers.0's inputs are 10x larger."""

    def values(self, sample):
        if sample.sample_id not in self._values:
            values = dict(super().values(sample))
            if sample.config.get("call") == "layers.0":
                values = {k: (v.detach() * 10).requires_grad_(v.requires_grad)
                          if torch.is_tensor(v) and v.is_floating_point() else v
                          for k, v in values.items()}
            self._values[sample.sample_id] = values
        return self._values[sample.sample_id]


class TestOperatorScopePools(unittest.TestCase):
    def test_each_threshold_is_the_largest_trusted_error_over_every_payload(self):
        # Two call sites of one operator; compile drifts ~140x more at layers.0, whose inputs
        # are 10x larger (PERTURB's error grows with mean|x|^2). One threshold per result,
        # set by the payload where a trusted implementation is least accurate (user decision
        # 2026-11-19: pooled over every payload, as Gate 1 pools native's error).
        scope = _CallScope(F.SCALE_SHIFT, F.SCALE_SHIFT.benchmark[0], device="cpu")
        samples = [Sample(f"vt/{call}/seed{k}", k, {"call": call})
                   for call, seeds in (("layers.0", (1, 2)), ("layers.1", (3, 4))) for k in seeds]
        report = verify_operator(scope, scope.candidate(F.scale_shift_pair()), samples=samples,
                                 compile_backend=PERTURB)
        self.assertEqual(report.status, PASSED, codes(report))
        for name, entry in report.subject["thresholds"].items():
            by_case = [c.evidence["trusted"]["compile"][name]["E2"] for c in report.cases]
            self.assertGreater(min(by_case[:2]), 10 * max(by_case[2:]), name)   # layers.0 drifts
            self.assertEqual(entry["E2"]["trusted"], max(by_case), name)
            self.assertEqual((entry["E2"]["from"], entry["E2"]["threshold"]),
                             ("compile", 2 * max(by_case)), name)
        loud = report.subject["thresholds"]["grad:w"]["E2"]["threshold"]
        scale = 1 + 4 * loud            # a fault far beyond every trusted error is still caught
        bad = verify_operator(scope, scope.candidate(F.scale_shift_pair(dw_scale=scale)),
                              samples=samples, compile_backend=PERTURB)
        self.assertEqual({c.case_id for c in bad.cases if c.status == FAILED},
                         {s.sample_id for s in samples})


if __name__ == "__main__":
    unittest.main()
