"""Qwen-specific tests, grouped by the level or tier they exercise.

Generic tests -- the opdecl contract, the tier machinery, the structured-output
protocol -- stay at the ``tests/`` root; they are not about this workload.
"""

#: Why a test that reads the tracked snapshot's Level-2 case list is skipped.
QWEN3_SNAPSHOT_STALE = (
    "the tracked Qwen3 snapshot predates harvest/snapshot.py's TASK_SOURCES; "
    "this runs once the snapshot is regenerated from a GPU harvest of the real run")


def qwen3_snapshot_current() -> bool:
    """Whether the tracked snapshot's Level-2 tasks are ``TASK_SOURCES``'s.

    ``harvest.snapshot --write`` writes exactly those tasks, so this holds
    after every regeneration and fails only while the code has moved on.
    """
    from evograd.benchmark.topdown.qwen3_0_6b.harvest.snapshot import TASK_SOURCES, load

    return set(load()["tasks"]) == set(TASK_SOURCES)
