"""**Whole-model gate** -- the fourth part of tier 3, and the one that knows no model.

    tier3_model.py    the workload protocol, and the import path
    tier3_patch.py    kernels, sites, and the two ways to insert one
    tier3_runner.py   building, timing, and reporting
    tier3_gate/       this package -- what must be true before a timer starts

Site preflight proves a kernel correct at its declared shapes. It cannot prove
that a hundred of them assembled into a model still train, and the failure it
misses is the expensive one: a provider that is right on a 4096-row grid, wrong
in composition, and fast. This package is the machinery for catching that.

Everything here is a question about a *kernel*, not about an architecture:

    numerics.py   what a whole-model comparison measures, how results are
                  grouped into roles, and what a calibrated envelope is
    purity.py     is the provider a function of its arguments, or does it
                  remember what it was called with before?

A workload supplies the model, the sites and the calibration artifact; the
runner reaches all of this through ``TrainingWorkload.model_correctness``. The
dependency runs one way -- a workload package imports this, never the reverse --
which is what keeps a threshold from quietly acquiring a branch on a model name.

Nothing here is imported by the runner directly. It is reached through the
workload's gate, so a workload that has not calibrated one pays nothing.
"""
