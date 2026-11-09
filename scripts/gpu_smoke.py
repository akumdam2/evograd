"""GPU smoke test for the opdecl stack's oracle. Run on a CUDA node:

    # shapes/dtypes of ground-truth grads for every op
    PYTHONPATH=src python scripts/gpu_smoke.py

    # smoke an externally generated declaration
    PYTHONPATH=src python scripts/gpu_smoke.py --declaration output/scaffold/my_op.py:op

A candidate is judged by Gate 1 on its task pack's payloads (``evograd seed
--pack`` or ``python -m evograd.adbench_tools evaluate``), not here.
"""

from __future__ import annotations

import argparse
import sys


def output_shapes(op, y) -> dict[str, tuple[int, ...]]:
    """Declared output name -> shape, for one-output and structured operators.

    ``oracle`` returns a Tensor for a one-output declaration and a tuple for a
    structured one, so a single ``y.shape`` only ever described half of the
    registry. ``as_output_tuple`` is the declaration contract that normalizes
    the two and refuses a result whose arity disagrees with the declaration,
    which keeps that disagreement a reported failure rather than a formatting
    accident.
    """
    from evograd.opdecl.inputs import as_output_tuple

    return {
        name: tuple(tensor.shape)
        for name, tensor in zip(op.output_names, as_output_tuple(op, y), strict=True)
    }


def oracle_smoke(device: str, declaration: str | None = None) -> int:
    from evograd.opdecl import make_case_inputs, oracle
    from evograd.benchmark import TASKS, load_task

    failed = []
    if declaration:
        external = load_task(declaration)
        selected = {external.name: external}
    else:
        selected = TASKS
    for name, op in sorted(selected.items()):
        cases = op.benchmark or op.coverage
        if not cases:
            print(f"[skip] {name}: no workload declared")
            continue
        workload = cases[0]
        try:
            inputs = make_case_inputs(op, workload, device=device)
            y, grads = oracle(op, inputs)
            outputs = output_shapes(op, y)
            shapes = {g: tuple(t.shape) for g, t in grads.items()}
            print(f"[ok]   {name}: outputs {outputs} grads {shapes}")
        except Exception as exc:  # noqa: BLE001 — smoke test reports and moves on
            failed.append(name)
            print(f"[FAIL] {name}: {exc}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--declaration",
        help="external declaration as path.py:op (instead of every registered task)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return oracle_smoke(args.device, args.declaration)


if __name__ == "__main__":
    sys.exit(main())
