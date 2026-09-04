"""Live-boundary invocation accounting: per site, in one namespace.

The defect these cover: a QKV-only patch installs the composite Qwen3Attention
adapter, which runs the ``attention`` boundary as well, so the tap sees twice
the invocations that the *patched* sites alone declare. Comparing the two
against one aggregate total failed a run in which every checked invocation had
passed. Expectations here are derived from the adapter grouping and the site
registry's own counts; no test states a total.
"""

from __future__ import annotations

import unittest

from evograd.bench.workloads.qwen3.evaluation.tier3.boundary import (
    BoundaryReport,
    SitePlan,
    invocation_id,
)
from evograd.bench.workloads.qwen3.evaluation.tier3.sites import (
    ADAPTER_GROUPS,
    SITE_ATTENTION,
    SITE_MLP,
    SITE_QKV,
    SITE_RESIDUAL,
    expected_counts,
    live_sites,
    supporting_sites,
)

LAYERS = 4


def _record(report: BoundaryReport, site: str, layer: int, ordinal: int, *,
            ok: bool = True) -> None:
    """One invocation, exactly the way the tap's listener records it."""
    identity = invocation_id(site, layer, ordinal)
    if identity in report.ids:
        report.duplicates.append(identity)
        return
    report.ids.add(identity)
    report.counts[site] = report.counts.get(site, 0) + 1
    report.invocations.append({
        "id": identity, "site": site, "layer": layer, "category": None,
        "ordinal": ordinal, "op": site,
        "outputs": [{"name": "out", "max_abs_err": 0.0 if ok else 9.0,
                     "atol": 1e-2, "rtol": 1e-2, "ok": ok, "finite": True}],
        "gradients": [],
    })


def _full_run(requested, *, layers: int = LAYERS, skip=(), duplicate=(),
              extra=()) -> dict:
    """A complete step's worth of taps for ``requested``, then the summary.

    ``skip``/``duplicate`` are ``(site, layer)`` pairs; ``extra`` names sites the
    tap fired for that no adapter was installed for.
    """
    plan = SitePlan.build(requested, layers=layers)
    report = BoundaryReport()
    ordinal = 0
    for site in live_sites(requested):
        for index in range(plan.expected[site]):
            if (site, index) in skip:
                continue
            ordinal += 1
            _record(report, site, index, ordinal)
            if (site, index) in duplicate:
                _record(report, site, index, ordinal)
    for site in extra:
        ordinal += 1
        _record(report, site, 0, ordinal)
    return report.to_dict(plan=plan)


class TestSitePlanDerivation(unittest.TestCase):
    def test_the_adapter_grouping_is_the_single_source_of_truth(self):
        # Nothing here names a pair; the grouping does.
        grouped = {site for group in ADAPTER_GROUPS for site in group}
        self.assertEqual(grouped, set(expected_counts(LAYERS)))
        for group in ADAPTER_GROUPS:
            for site in group:
                self.assertEqual(set(live_sites((site,))), set(group))

    def test_patching_qkv_carries_attention_and_says_so(self):
        plan = SitePlan.build((SITE_QKV,), layers=LAYERS)
        self.assertEqual(plan.patched, (SITE_QKV,))
        self.assertEqual(plan.supporting, (SITE_ATTENTION,))
        self.assertEqual(plan.role(SITE_QKV), "patched")
        self.assertEqual(plan.role(SITE_ATTENTION), "supporting")
        self.assertEqual(plan.role(SITE_MLP), "unexpected")

    def test_a_site_with_no_carrier_carries_nothing(self):
        for site in (SITE_MLP, SITE_RESIDUAL):
            plan = SitePlan.build((site,), layers=LAYERS)
            self.assertEqual(plan.supporting, ())
            self.assertEqual(set(plan.expected), {site})

    def test_expected_counts_come_from_the_registry_not_a_constant(self):
        for layers in (2, 4, 28):
            plan = SitePlan.build((SITE_QKV,), layers=layers)
            self.assertEqual(plan.expected, {
                SITE_QKV: expected_counts(layers)[SITE_QKV],
                SITE_ATTENTION: expected_counts(layers)[SITE_ATTENTION],
            })

    def test_supporting_is_the_carried_remainder(self):
        self.assertEqual(supporting_sites((SITE_QKV, SITE_ATTENTION)), ())
        self.assertEqual(supporting_sites((SITE_ATTENTION,)), (SITE_QKV,))


class TestCompleteRunsPass(unittest.TestCase):
    """Each patch shape, run to completion, must be accepted."""

    def test_qkv_only(self):
        summary = _full_run((SITE_QKV,))
        self.assertTrue(summary["ok"], summary["missing_or_extra"])
        self.assertEqual(summary["patched_sites"], [SITE_QKV])
        self.assertEqual(summary["supporting_sites"], [SITE_ATTENTION])
        # The exact shape that used to fail: more invocations than the patched
        # site alone declares, and every one of them correct.
        self.assertGreater(summary["checked_invocations"],
                           expected_counts(LAYERS)[SITE_QKV])
        self.assertEqual(summary["failure_count"], 0)

    def test_attention_only(self):
        summary = _full_run((SITE_ATTENTION,))
        self.assertTrue(summary["ok"], summary["missing_or_extra"])
        self.assertEqual(summary["patched_sites"], [SITE_ATTENTION])
        self.assertEqual(summary["supporting_sites"], [SITE_QKV])

    def test_qkv_and_attention_together(self):
        summary = _full_run((SITE_QKV, SITE_ATTENTION))
        self.assertTrue(summary["ok"], summary["missing_or_extra"])
        self.assertEqual(summary["supporting_sites"], [])
        self.assertEqual(sorted(summary["patched_sites"]),
                         sorted([SITE_QKV, SITE_ATTENTION]))

    def test_residual_rmsnorm_a_site_with_no_carrier(self):
        summary = _full_run((SITE_RESIDUAL,))
        self.assertTrue(summary["ok"], summary["missing_or_extra"])
        self.assertEqual(summary["supporting_sites"], [])
        self.assertEqual(summary["checked_invocations"],
                         expected_counts(LAYERS)[SITE_RESIDUAL])

    def test_every_live_site_is_reported_with_its_role(self):
        summary = _full_run((SITE_QKV,))
        self.assertEqual(summary["sites"][SITE_QKV]["role"], "patched")
        self.assertEqual(summary["sites"][SITE_ATTENTION]["role"], "supporting")
        for site, row in summary["sites"].items():
            self.assertEqual(row["observed"], row["expected"], site)


class TestStrictFailures(unittest.TestCase):
    def test_a_missing_patched_invocation_fails_and_names_the_site(self):
        summary = _full_run((SITE_QKV,), skip=((SITE_QKV, 2),))
        self.assertFalse(summary["ok"])
        self.assertFalse(summary["coverage_ok"])
        self.assertEqual(summary["missing_or_extra"], {SITE_QKV: 1})
        self.assertEqual(summary["sites"][SITE_QKV]["role"], "patched")

    def test_a_missing_supporting_invocation_also_fails(self):
        # A carried boundary is still a declared boundary; it is validated, not
        # waved through, so its own count must hold.
        summary = _full_run((SITE_QKV,), skip=((SITE_ATTENTION, 0),))
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["missing_or_extra"], {SITE_ATTENTION: 1})
        self.assertEqual(summary["sites"][SITE_ATTENTION]["role"], "supporting")

    def test_a_duplicate_invocation_fails(self):
        summary = _full_run((SITE_QKV,), duplicate=((SITE_QKV, 1),))
        self.assertFalse(summary["ok"])
        self.assertFalse(summary["coverage_ok"])
        self.assertTrue(summary["duplicate_ids"])

    def test_an_unexpected_site_fails_and_is_not_folded_into_missing(self):
        summary = _full_run((SITE_QKV,), extra=(SITE_MLP,))
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["unexpected_sites"], {SITE_MLP: 1})
        self.assertEqual(summary["missing_or_extra"], {})
        self.assertNotIn(SITE_MLP, summary["expected_counts"])
        self.assertEqual(summary["sites"][SITE_MLP]["role"], "unexpected")

    def test_a_numerical_failure_still_fails(self):
        plan = SitePlan.build((SITE_QKV,), layers=LAYERS)
        report = BoundaryReport()
        ordinal = 0
        for site in live_sites((SITE_QKV,)):
            for index in range(plan.expected[site]):
                ordinal += 1
                _record(report, site, index, ordinal,
                        ok=not (site == SITE_QKV and index == 0))
        summary = report.to_dict(plan=plan)
        self.assertFalse(summary["ok"])
        self.assertTrue(summary["coverage_ok"])   # counting was fine
        self.assertEqual(summary["failure_count"], 1)

    def test_a_lost_record_is_named_rather_than_passing(self):
        plan = SitePlan.build((SITE_RESIDUAL,), layers=LAYERS)
        report = BoundaryReport()
        for index in range(plan.expected[SITE_RESIDUAL]):
            _record(report, SITE_RESIDUAL, index, index + 1)
        report.invocations.pop()          # counted, but its record vanished
        summary = report.to_dict(plan=plan)
        self.assertTrue(summary["record_desync"])
        self.assertFalse(summary["ok"])


class TestGateReasonNamesTheSite(unittest.TestCase):
    def test_a_coverage_failure_reports_site_role_expected_and_observed(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.gate import (
            _boundary_reason,
        )

        summary = _full_run((SITE_QKV,), skip=((SITE_QKV, 0),))
        reason = _boundary_reason(summary)
        self.assertIn(SITE_QKV, reason)
        self.assertIn("patched", reason)
        self.assertNotEqual(reason, "the live-boundary validation failed")

    def test_an_unexpected_site_is_reported_as_such(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.gate import (
            _boundary_reason,
        )

        reason = _boundary_reason(_full_run((SITE_QKV,), extra=(SITE_MLP,)))
        self.assertIn(SITE_MLP, reason)
        self.assertIn("no adapter", reason)


if __name__ == "__main__":
    unittest.main()
