"""MegaFold's real input data: a pool of frozen, featurized PDB structures.

MegaFold featurizes an mmCIF (plus its MSAs and template hits) on the fly, and
that is not reproducible bit for bit: RDKit's ETKDG reference conformers are
drawn unseeded. So each structure is featurized once, with MegaFold's own
training featurization (``PDBInput`` -> ``maybe_transform_to_atom_input``, the
``configs/megafold_1x1.yaml`` crop, MSA, template and constraint settings) and
every data-side draw seeded, and frozen with MegaFold's ``atom_input_to_file``.
The pool's ``pool.json`` records the rule, the inputs' and outputs' hashes and
what each structure contains; a workload loads only a pinned file and checks
its hash first.

    python -m evograd.benchmark.topdown.megafold.data featurize --megafold PATH --out DIR \\
        [--structure a4/7a4d-assembly1.cif ...]

Featurization needs MegaFold's checkout (its test structures, MSAs and CCD
table live there, read relative to its root) and a GPU (MegaFold resolves a
Triton driver at import). Failures raise; nothing is substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

#: The structure every MegaFold case is recorded on: 384 tokens after the crop
#: (MegaFold's crop size), six chains, a 12,097-row MSA. One structure, because
#: every site's shapes depend on its token, atom and MSA counts.
SAMPLE = "7a4d-assembly1"
#: Where the pool lives, unless ``EVOGRAD_MEGAFOLD_POOL`` says otherwise.
DEFAULT_POOL = "/work/hdd/bcjw/wzhan/ADBench_payload/megafold/pool_seed0"
POOL_MANIFEST = "pool.json"
#: The warm-started weights verification and evaluation start from
#: (``integrations.warm_start``), unless ``EVOGRAD_MEGAFOLD_WARM_START`` says otherwise.
DEFAULT_WARM_START = "/work/hdd/bcjw/wzhan/ADBench_payload/megafold/warm_start_order909_200/state.pt"

SEED = 0
TEST_DATA = "data/test/pdb_data"
STRUCTURES = ("a4/7a4d-assembly1.cif",)
#: ``configs/megafold_1x1.yaml`` ``training.initial_training.dataset_config``.
CROP = {"contiguous_weight": 0.2, "spatial_weight": 0.4, "spatial_interface_weight": 0.4, "n_res": 384}
PDB_INPUT_KWARGS = {"max_msas_per_chain": 16384, "max_num_msa_tokens": 16777216,
                    "max_templates_per_chain": 20, "num_templates_per_chain": 4,
                    "max_num_template_tokens": 3072, "max_num_atoms": 7000, "min_length": None,
                    "filter_for_alphabetic_chain_orderings": True, "max_length": 2560,
                    "constraints": ["pocket", "contact", "docking"], "constraints_ratio": 0.1}


def pool_dir() -> Path:
    return Path(os.environ.get("EVOGRAD_MEGAFOLD_POOL") or DEFAULT_POOL)


def warm_start_path() -> Path:
    return Path(os.environ.get("EVOGRAD_MEGAFOLD_WARM_START") or DEFAULT_WARM_START)


def sha_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pool_manifest(pool: Path | None = None) -> dict:
    pool = Path(pool or pool_dir())
    try:
        return json.loads((pool / POOL_MANIFEST).read_text())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"no MegaFold pool at {pool} (set EVOGRAD_MEGAFOLD_POOL, or build one with "
            f"`python -m evograd.benchmark.topdown.megafold.data featurize`)") from None


def load_sample(file_id: str = SAMPLE, pool: Path | None = None):
    """``(AtomInput, entry)`` for one pooled structure, its file hash checked first."""
    pool = Path(pool or pool_dir())
    entry = next((s for s in pool_manifest(pool)["samples"] if s["file_id"] == file_id), None)
    if entry is None:
        raise KeyError(f"{file_id} is not in the MegaFold pool at {pool}")
    path = pool / Path(entry["path"]).name
    if sha_file(path) != entry["sha256"]:
        raise ValueError(f"{path} does not match {POOL_MANIFEST} (sha256)")
    from megafold.inputs import file_to_atom_input

    return file_to_atom_input(path), entry


def _summary(atom_input) -> dict:
    import torch

    tensors = {name: {"shape": list(v.shape), "dtype": str(v.dtype)}
               for name, v in atom_input.dict().items() if torch.is_tensor(v)}
    perms = atom_input.molecule_atom_perms or []  # row i: atom i's image under each permutation
    constraints = atom_input.token_constraints
    return {"tokens": int(atom_input.molecule_ids.shape[0]), "atoms": int(atom_input.atom_inputs.shape[0]),
            "atom_inputs_width": int(atom_input.atom_inputs.shape[-1]),
            "msa_rows": None if atom_input.msa is None else int(atom_input.msa.shape[0]),
            "template_mask": None if atom_input.template_mask is None else atom_input.template_mask.tolist(),
            "constraints_applied": bool(constraints is not None and (constraints != -1).any()),
            "atoms_moved_by_a_permutation": sum(1 for row in perms if len(set(row)) > 1),
            "example_source": atom_input.example_source, "tensors": tensors}


def featurize(megafold: Path, out: Path, structures=STRUCTURES) -> Path:
    """Featurize ``structures`` (relative to the checkout's test mmCIF dir) into a new pool."""
    import numpy as np

    megafold = megafold.resolve()
    head = subprocess.run(["git", "-C", str(megafold), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    out.mkdir(parents=True, exist_ok=False)  # a pool is written once
    os.chdir(megafold)  # the CCD table and the test data are read relative to the checkout
    sys.path.insert(0, str(megafold))
    if not hasattr(np, "row_stack"):
        # numpy 2.5 removed the alias MegaFold calls in its atom-permutation
        # code, inside a try/except that otherwise degrades every molecule's
        # permutations to the identity; numpy 2.3 defines it as vstack.
        np.row_stack = np.vstack
    import torch
    from megafold.inputs import PDBInput, atom_input_to_file, maybe_transform_to_atom_input

    mmcifs, msas, templates = (f"{TEST_DATA}/mmcifs", f"{TEST_DATA}/data_caches/msa/msas",
                               f"{TEST_DATA}/data_caches/template/templates")
    manifest = {"rule": "MegaFold training featurization (PDBInput + maybe_transform_to_atom_input, "
                        "raise_exception=True), data-side RNG seeded per structure, frozen with "
                        "atom_input_to_file", "megafold_commit": head, "seed": SEED, "crop": CROP,
                "pdb_input_kwargs": PDB_INPUT_KWARGS, "msa_dir": msas, "templates_dir": templates,
                "kalign_binary_path": None, "numpy": np.__version__, "torch": torch.__version__,
                "config_sha256": sha_file(megafold / "configs/megafold_1x1.yaml"),
                "ccd_sha256": sha_file(megafold / "data/ccd_data/components_smiles.json"), "samples": []}
    for rel in structures:
        file_id = Path(rel).name.split(".")[0]
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        started = time.time()
        pdb_input = PDBInput(mmcif_filepath=f"{mmcifs}/{rel}", chains=(None, None), cropping_config=dict(CROP),
                             training=True, inference=False, example_source="pdb", msa_dir=msas,
                             templates_dir=templates, **PDB_INPUT_KWARGS)
        atom_input = maybe_transform_to_atom_input(pdb_input, raise_exception=True)
        path = atom_input_to_file(atom_input, out / f"{file_id}.pt")
        inputs = {str(p): sha_file(p) for d in (mmcifs, msas, templates)
                  for p in sorted(Path(d, file_id[1:3]).glob(f"{file_id}*"))}
        manifest["samples"].append({"file_id": file_id, "path": Path(path).name, "sha256": sha_file(path),
                                    "featurize_seconds": round(time.time() - started, 1), "inputs": inputs,
                                    **_summary(atom_input)})
        (out / POOL_MANIFEST).write_text(json.dumps(manifest, indent=1))
    return out / POOL_MANIFEST


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("featurize")
    build.add_argument("--megafold", type=Path, required=True, help="MegaFold checkout root")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--structure", action="append", help="mmCIF path under the test mmcifs dir")
    args = parser.parse_args(argv)
    print(featurize(args.megafold, args.out, tuple(args.structure or STRUCTURES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
