"""Pure tests for the final-main interface, dispatch, scaffold, and NCU layers."""

from __future__ import annotations

import inspect
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from evograd.api import _file_forward_spec
from evograd.dispatch import _emit_dispatcher
from evograd.ncu.profile import _parse_csv, _script
from evograd.ncu.roofline import analyze
from evograd.ops import get_op, load_op
from evograd.scaffold import _source, _validate_spec


def _example_forward(x, eps=1e-5):
    return x


_SPEC = {
    "dims": ["rows", "cols"],
    "args": [
        {
            "name": "x",
            "kind": "active_tensor",
            "shape": "[rows, cols]",
            "dtype": None,
        },
        {"name": "eps", "kind": "inactive_scalar", "default": 1e-5},
    ],
    "output": {"name": "out", "shape": "[rows, cols]"},
    "input_body": "x = torch.randn((rows, cols), device=device, dtype=dtype)",
    "forward_semantics": "identity",
    "backward_semantics": "return dx",
    "correctness": [{"dims": {"rows": 8, "cols": 64}, "dtype": "float32"}],
    "benchmark": [
        {"dims": {"rows": 8, "cols": 64}, "dtype": "float32"},
        {"dims": {"rows": 32, "cols": 64}, "dtype": "float32"},
    ],
    "regime_dim": "rows",
    "regime_split": 16,
    "reduced_grads": [],
}


class TestForwardBoundary(unittest.TestCase):
    def test_file_reference_infers_single_public_function(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "forward.py"
            path.write_text("def forward(x):\n    return x\n", encoding="utf-8")
            self.assertEqual(_file_forward_spec(path), f"{path.resolve()}:forward")

    def test_file_reference_requires_name_when_ambiguous(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "forward.py"
            path.write_text(
                "def one(x):\n    return x\n\ndef two(x):\n    return x\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "2 public"):
                _file_forward_spec(path)


class TestExternalDeclaration(unittest.TestCase):
    def test_generated_declaration_compiles_loads_and_resolves_from_env(self):
        spec = dict(_SPEC)
        spec["args"] = [dict(arg) for arg in _SPEC["args"]]
        _validate_spec("external_identity", inspect.signature(_example_forward), spec)
        source = _source(
            "external_identity",
            "some.module:forward",
            spec,
        )
        compile(source, "<external-declaration>", "exec")
        with TemporaryDirectory() as td:
            path = Path(td) / "declaration.py"
            path.write_text(source, encoding="utf-8")
            op = load_op(f"{path}:op")
            self.assertEqual(op.name, "external_identity")
            self.assertEqual(op.declaration, f"{path}:op")
            with mock.patch.dict(
                os.environ, {"EVOGRAD_DECLARATION": f"{path}:op"}
            ):
                self.assertEqual(get_op("external_identity").name, op.name)


class TestShapeDispatch(unittest.TestCase):
    def test_emitted_dispatcher_compiles_and_uses_plain_route_tag(self):
        op = get_op("softmax")
        with TemporaryDirectory() as td:
            root = Path(td)
            programs = {}
            for tag in ("small", "large"):
                path = root / f"{tag}.py"
                path.write_text("# candidate placeholder\n", encoding="utf-8")
                programs[tag] = path
            output = root / "dispatcher.py"
            _emit_dispatcher(op, programs, "small", "large", 4096.0, output)
            source = output.read_text(encoding="utf-8")
            compile(source, str(output), "exec")
            self.assertIn("return output, (route, *saved)", source)
            self.assertNotIn("torch.tensor(route", source)


class TestNCUHelpers(unittest.TestCase):
    def test_profile_scripts_compile(self):
        op = get_op("softmax")
        for warmup in (5, None):
            source = _script(
                op,
                Path("/tmp/candidate.py"),
                op.correctness[0],
                Path("/tmp/inputs.pt"),
                warmup=warmup,
            )
            compile(source, "<ncu-script>", "exec")

    def test_csv_metrics_are_normalized(self):
        csv_text = "\n".join(
            (
                '"Kernel Name","Metric Name","Metric Unit","Metric Value"',
                '"kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","70.0"',
                '"kernel","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","40.0"',
            )
        )
        metrics, rows = _parse_csv(csv_text)
        self.assertEqual(metrics["sm_throughput_pct"], 70.0)
        self.assertEqual(metrics["dram_throughput_pct"], 40.0)
        self.assertEqual(len(rows), 2)

    def test_roofline_classification_and_skip(self):
        result = analyze(
            {"sm_throughput_pct": 96.0, "dram_throughput_pct": 35.0}
        )
        self.assertEqual(result.bottleneck, "compute")
        self.assertTrue(result.at_roofline)


if __name__ == "__main__":
    unittest.main()
