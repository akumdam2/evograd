"""Every emitted `torch.ops.aten.*` fallback call must match its schema arity.

This is the check the invalid artifact would have failed: its seed emitted
`aten._log_softmax.default(mm, 1, False, 1, False)` against a three-argument
schema, so the program raised on its first call.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--program", required=True, type=Path)
    a = ap.parse_args()
    src = a.program.read_text()
    calls = re.findall(r"_resolve_aten\('([^']+)'\)\(([^\n]*?)\)\s*(?:#|$)", src)
    bad = 0
    for target, args in calls:
        n = len([t for t in args.split(", ") if t.strip()])
        try:
            obj = torch.ops.aten
            for part in target[len("aten."):].split("."):
                obj = getattr(obj, part)
            want = len(obj._schema.arguments)
        except Exception:
            want = None
        mismatch = want is not None and n != want
        bad += mismatch
        print(f"  {target:<44} emitted={n} schema={want}"
              + ("   <-- ARITY MISMATCH" if mismatch else ""))
    print(f"  {len(calls)} aten fallback call(s), {bad} arity mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
