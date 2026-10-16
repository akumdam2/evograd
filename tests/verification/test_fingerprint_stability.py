"""Reference fingerprints are stable across processes and change with semantics.

Version 1 hashed code constants by ``repr()``; a frozenset constant iterates in
``PYTHONHASHSEED`` order, so a policy frozen in one process failed its
reference binding in the next (run 20261013_a2). Version 2 encodes constants
canonically. These tests run fresh interpreters under several seeds -- never
only ``PYTHONHASHSEED=0`` -- and check that real semantic changes still change
the digest and that unidentifiable references still fail closed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
import warnings

from evograd.verification.common import fingerprint as fp
from evograd.verification.common.fingerprint import (
    FINGERPRINT_VERSION, UnidentifiableReference, fingerprint)

ROOT = Path(__file__).resolve().parents[2]

warnings.filterwarnings("ignore")

SEEDS = ("0", "1", "2", "3", "4242", "random")

#: Digests printed by a fresh interpreter, including an entire operator scope's
#: reference identity (input recipe, sampling, module wrapper, native site).
#: Fixture model contracts keep this regression independent of model migrations.
PROBE = textwrap.dedent("""
    import json, sys
    import torch
    from evograd.verification.common.fingerprint import fingerprint
    from tests.verification import _fingerprint_subjects as S
    from tests.verification import _fixtures as F
    from tests.verification.test_reference_contract import model_op, rounded_forward
    from evograd.verification.gate1.adapters import declared_values
    from evograd.verification.gate2.adapters.operator import OperatorScope

    out = {name: fingerprint(getattr(S, name))["digest"]
           for name in ("set_member", "nested_constants", "nested_code", "reaches_global")}
    for name in ("scale_shift_ref", "scale_shift_runtime", "split_ref", "cancel_ref"):
        out["reference." + name] = fingerprint(getattr(F, name))["digest"]
    out["model_reference"] = fingerprint(rounded_forward)["digest"]
    out["declared_values"] = fingerprint(declared_values)["digest"]
    for name, op in (("generic", F.SCALE_SHIFT), ("model", model_op())):
        identity = OperatorScope(op, op.correctness[0], device="cpu").reference_identity()
        out[name + "_operator_scope"] = json.dumps(identity, sort_keys=True)
    print(json.dumps(out, sort_keys=True))
""")


def run_probe(seed: str, extra: str = "") -> dict:
    env = {**os.environ, "PYTHONHASHSEED": seed,
           "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    done = subprocess.run([sys.executable, "-c", extra + PROBE], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)
    if done.returncode:
        raise AssertionError(done.stderr[-4000:])
    return json.loads(done.stdout.strip().splitlines()[-1])


def made(source: str, name: str = "site", **scope):
    """A fresh function from source, in an unimportable module (always analysed)."""
    namespace = {"__name__": "fingerprint_probe", **scope}
    exec(compile(textwrap.dedent(source), "<probe>", "exec"), namespace)
    return namespace[name]


class TestAcrossProcesses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {seed: run_probe(seed) for seed in SEEDS}

    def test_every_seed_gives_the_same_digests(self):
        first = self.runs[SEEDS[0]]
        for seed, digests in self.runs.items():
            with self.subTest(seed=seed):
                self.assertEqual(digests, first)

    def test_the_subjects_really_have_unordered_constants(self):
        """Guard against a vacuous pass: the frozenset's iteration order is
        what used to differ between processes."""
        from tests.verification import _fingerprint_subjects as S

        constants = S.set_member.__code__.co_consts
        self.assertTrue(any(type(c) is frozenset for c in constants), constants)
        nested = S.nested_constants.__code__.co_consts
        self.assertTrue(any(type(c) is frozenset and all(type(i) is tuple for i in c)
                            for c in nested), nested)

    def test_import_state_does_not_change_the_identity(self):
        """Importing more modules first (as a candidate file does) is not a change."""
        before = self.runs["1"]
        after = run_probe("1", extra="import json, importlib; [importlib.import_module(m) for m in "
                                      "('torch.nn.attention', 'torch.nn.functional')]\n")
        self.assertEqual(after, before)


class TestSemantics(unittest.TestCase):
    def assertDiffers(self, a: str, b: str, **scope):
        self.assertNotEqual(fingerprint(made(a, **scope))["digest"],
                            fingerprint(made(b, **scope))["digest"])

    def test_equivalent_sources_match(self):
        a = made("def site(x):\n    return x * 2 if x in {'a', 'b'} else x\n")
        b = made("def site(x):\n    # a comment changes nothing\n    return x * 2 if x in {'b', 'a'} else x\n")
        self.assertEqual(fingerprint(a)["digest"], fingerprint(b)["digest"])

    def test_meaningful_constant_changes_invalidate(self):
        cases = [
            ("x in {'a', 'b'}", "x in {'a', 'c'}"),                 # frozenset member
            ("x in {'a', 'b'}", "x in {'a', 'b', 'c'}"),           # frozenset size
            ("x in (1, 2)", "x in (2, 1)"),                         # tuple order
            ("x == 1", "x == 1.0"),                                 # int vs float
            ("x == 1", "x == True"),                                # int vs bool
            ("x == 0.0", "x == -0.0"),                              # signed zero
            ("x == '1'", "x == b'1'"),                              # str vs bytes
            ("x in (1, 2)", "x in {1, 2}"),                         # tuple vs frozenset
            ("x in {(1, 'a'), (2, 'b')}", "x in {(1, 'a'), (2, 'c')}"),  # nested
        ]
        for left, right in cases:
            with self.subTest(left=left, right=right):
                self.assertDiffers(f"def site(x):\n    return {left}\n",
                                   f"def site(x):\n    return {right}\n")

    def test_nested_code_changes_invalidate(self):
        self.assertDiffers(
            "def site(x):\n    def inner(y):\n        return y in {'k1', 'k2'}\n    return inner(x)\n",
            "def site(x):\n    def inner(y):\n        return y in {'k1', 'k3'}\n    return inner(x)\n")
        self.assertDiffers("def site(x):\n    return [v * 2 for v in x]\n",
                           "def site(x):\n    return [v * 3 for v in x]\n")

    def test_defaults_and_captured_globals_still_bind(self):
        self.assertDiffers("def site(x, k=2.0):\n    return x * k\n",
                           "def site(x, k=3.0):\n    return x * k\n")
        a = made("def site(x):\n    return x * K\n", K=2.0)
        b = made("def site(x):\n    return x * K\n", K=3.0)
        self.assertNotEqual(fingerprint(a)["digest"], fingerprint(b)["digest"])

    def test_unknown_constant_types_fail_closed(self):
        with self.assertRaises(UnidentifiableReference):
            fp._constant(object())

        class Opaque:
            def __call__(self, x):
                return x

        with self.assertRaises(UnidentifiableReference):
            fingerprint(Opaque())

    def test_the_digest_is_versioned(self):
        site = made("def site(x):\n    return x in {'a', 'b'}\n")
        current = fingerprint(site)
        self.assertEqual(current["fingerprint_version"], FINGERPRINT_VERSION)
        self.assertEqual(FINGERPRINT_VERSION, 2)
        original = fp.FINGERPRINT_VERSION
        try:
            fp.FINGERPRINT_VERSION = 1
            self.assertNotEqual(fingerprint(site)["digest"], current["digest"])
        finally:
            fp.FINGERPRINT_VERSION = original

    def test_canonical_encoding_is_type_preserving(self):
        encodings = {fp._constant(v) for v in (1, 1.0, True, "1", b"1", (1,), frozenset({1}),
                                               0.0, -0.0, None, ...)}
        self.assertEqual(len(encodings), 11)
        self.assertEqual(fp._constant(frozenset({"a", "b", "c"})),
                         fp._constant(frozenset({"c", "a", "b"})))


if __name__ == "__main__":
    unittest.main()
