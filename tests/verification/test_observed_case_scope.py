"""Every observed case of every integrated model can be judged at operator scope.

Operator-scope Gate 2 holds a declaration's ``parameter_args`` as module state
and refuses a declaration that leaves them undeclared, so a model case whose
reference op omits them fails for every candidate before any number is
measured (``gelu`` and ``quick_gelu`` did, for Qwen2-VL and ConvNeXt V2).
"""

import unittest

from evograd.verification.gate2.adapters.operator import OperatorScope
from evograd.verification.integrations import PACKAGES, load
from evograd.verification.integrations.base import observed_cases


class TestObservedCaseScope(unittest.TestCase):
    def test_every_reference_op_declares_its_parameter_split(self):
        from evograd.benchmark import TASKS
        from evograd.benchmark.topdown import load_snapshot

        for name in sorted(PACKAGES):
            stale = sorted(set(load_snapshot(name)["tasks"]) - set(TASKS))
            if stale:
                with self.subTest(model=name):
                    self.skipTest(f"the tracked snapshot lists tasks no longer declared "
                                  f"({stale}); the GPU harvest regenerates it")
                continue
            model = load(name)
            for case in observed_cases(model):
                with self.subTest(model=name, case=case.case_id):
                    op = model.reference_op(case.task)
                    self.assertIsNotNone(op.parameter_args)
                    OperatorScope(op, case.workload, device="cpu")


if __name__ == "__main__":
    unittest.main()
