"""Whether a tracked snapshot still predates the Level-2 tasks the code declares.

A harvest change that removes or merges a Level-2 task leaves the tracked
``snapshot.json`` naming the old tasks until the real-run harvest regenerates it
on a GPU. Tests that read its Level-2 case list skip until then; once every
tracked snapshot is regenerated these guards can go.
"""

#: Why such a test is skipped.
STALE_SNAPSHOT = ("the tracked snapshot names Level-2 tasks the code no longer declares; "
                  "this runs once the GPU harvest of the real run regenerates it")


def snapshot_is_current(model: str) -> bool:
    """Whether ``model``'s tracked snapshot names only tasks the code declares."""
    from evograd.benchmark import TASKS
    from evograd.benchmark.topdown import load_snapshot

    return set(load_snapshot(model)["tasks"]) <= set(TASKS)
