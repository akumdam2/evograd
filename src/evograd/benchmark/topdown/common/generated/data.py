"""Prepare fixed real inputs for DiT-S/2 and TorchRec DLRM workloads.

DiT needs its image shard and VAE in the Hugging Face cache. DLRM needs local
TorchRec-preprocessed Criteo NumPy files. Preparation never downloads data.
"""

import hashlib
import os
from pathlib import Path
import tempfile

import torch
from evograd.benchmark.topdown.common.data_paths import hf_cache_dir, prepared_dir

DATASET = ("ShaomuTan/imagenette", "061f0ab5ca70523e13982a7ba463c076af657386",
           "320px/train-00000-of-00001.parquet")
VAE = ("stabilityai/sd-vae-ft-mse", "31f26fdeee1355a5c34592e401dd41e45d25a493")
CLASS_IDS = (0, 217, 482, 491, 497, 566, 569, 571, 574, 701)
CLASS_NAMES = ("tench", "English springer", "cassette player", "chain saw", "church",
               "French horn", "garbage truck", "gas pump", "golf ball", "parachute")
SCHEMA = "evograd-dit-real-data/1"


def _dataset_file():
    from huggingface_hub import hf_hub_download
    repo, revision, filename = DATASET
    try:
        return hf_hub_download(repo, filename=filename, repo_type="dataset", revision=revision,
                               cache_dir=hf_cache_dir("dit_s_2"),
                               local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"Dataset {repo}@{revision}/{filename} is not cached; "
                           "download it before --real-input") from exc


def _vae_path():
    from huggingface_hub import snapshot_download
    repo, revision = VAE
    try:
        path = Path(snapshot_download(repo, revision=revision,
                                     allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"],
                                     cache_dir=hf_cache_dir("dit_s_2"),
                                     local_files_only=True))
    except Exception as exc:
        raise RuntimeError(f"VAE {repo}@{revision} is not cached; download its config "
                           "and safetensors weights before --real-input") from exc
    missing = [name for name in ("config.json", "diffusion_pytorch_model.safetensors")
               if not (path / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Pinned VAE cache is incomplete: {missing}. Download the missing files with "
            f"`hf download {repo} {' '.join(missing)} --revision {revision}`"
        )
    return path


def _crop(image, size=256):
    from PIL import Image
    while min(image.size) >= 2 * size:
        image = image.resize(tuple(dim // 2 for dim in image.size), Image.Resampling.BOX)
    scale = size / min(image.size)
    image = image.resize(tuple(round(dim * scale) for dim in image.size), Image.Resampling.BICUBIC)
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


def prepare_pinned(spec, *, batch_count=10, source_dir=None):
    """Encode fixed Imagenette rows with the pinned VAE, then freeze DiT inputs."""
    if spec.model == "torchrec_dlrm":
        from .dlrm_data import prepare_pinned as prepare_dlrm
        return prepare_dlrm(spec, source_dir=source_dir, batch_count=batch_count)
    if source_dir is not None:
        raise ValueError("--criteo-dir applies only to DLRM")
    from datasets import load_dataset
    from diffusers.models import AutoencoderKL
    from PIL import ImageOps
    import numpy as np

    if spec.input_mode != "real" or spec.arch_overrides:
        raise ValueError("pinned inputs require real mode and the published architecture")
    if batch_count < 1:
        raise ValueError("batch_count must be positive")
    count = batch_count * spec.batch_size
    shard = _dataset_file()
    dataset = load_dataset("parquet", data_files={"train": shard}, split=f"train[:{count}]")
    if len(dataset) != count:
        raise ValueError(f"dataset shard has {len(dataset)} rows, need {count}")
    names = tuple(dataset.features["label"].names)
    if names != CLASS_NAMES:
        raise ValueError(f"Imagenette ClassLabel order changed: {names}")
    vae = AutoencoderKL.from_pretrained(str(_vae_path()), local_files_only=True,
                                       use_safetensors=True).to(spec.device).eval()
    vae.requires_grad_(False)
    image_rng = torch.Generator(device="cpu").manual_seed(spec.data_seed)
    latent_rng = torch.Generator(device=spec.device).manual_seed(spec.data_seed + 1)
    batches, selection = [], []
    try:
        for start in range(0, count, spec.batch_size):
            latents = []
            for offset in range(0, spec.batch_size, 8):
                end = min(start + offset + 8, start + spec.batch_size)
                pixels = []
                for row_index in range(start + offset, end):
                    row = dataset[row_index]
                    image = row["image"].convert("RGB")
                    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
                    image = _crop(image)
                    flipped = bool(torch.rand((), generator=image_rng) < 0.5)
                    if flipped:
                        image = ImageOps.mirror(image)
                    pixels.append(torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1))
                    selection.append({"row": row_index, "label": int(row["label"]),
                                      "image_sha256": image_hash, "flipped": flipped})
                x = torch.stack(pixels).to(device=spec.device, dtype=torch.float32).div_(127.5).sub_(1)
                with torch.no_grad():
                    z = vae.encode(x).latent_dist.sample(generator=latent_rng) * 0.18215
                latents.append(z.cpu())
            latent = torch.cat(latents).to(getattr(torch, spec.dtype))
            expected = (spec.batch_size, spec.arch["in_channels"],
                        spec.arch["input_size"], spec.arch["input_size"])
            if tuple(latent.shape) != expected:
                raise ValueError(f"VAE latent shape {tuple(latent.shape)} differs from {expected}")
            rng = torch.Generator(device="cpu").manual_seed(spec.data_seed + 2 + start)
            batches.append({"latent": latent,
                            "noise": torch.randn(expected, generator=rng).to(latent.dtype),
                            "timesteps": torch.randint(spec.arch["diffusion_steps"],
                                                       (spec.batch_size,), generator=rng),
                            "labels": torch.tensor([CLASS_IDS[s["label"]]
                                                    for s in selection[start:start + spec.batch_size]]),
                            "drop_mask": (torch.rand(spec.batch_size, generator=rng)
                                          < spec.arch["class_dropout_prob"]).long()})
    finally:
        del vae
        if spec.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return {"schema_version": SCHEMA, "model": spec.model, "workload_id": spec.workload_id,
            "dataset_repo": DATASET[0], "dataset_revision": DATASET[1], "dataset_file": DATASET[2],
            "vae_repo": VAE[0], "vae_revision": VAE[1], "vae_scale": 0.18215,
            "examples": count, "selection": selection, "batches": batches}


def default_data_path(spec, source_dir=None):
    if spec.model == "torchrec_dlrm":
        from .dlrm_data import default_data_path as dlrm_path
        return dlrm_path(spec, source_dir)
    if source_dir is not None:
        raise ValueError("--criteo-dir applies only to DLRM")
    root = Path(os.environ.get("EVOGRAD_DIT_DATA_CACHE", prepared_dir("dit_s_2")))
    return root / f"{spec.workload_hash}-pinned.pt"


def ensure_pinned_data(spec, path=None, *, batch_count=10, source_dir=None):
    if spec.model == "torchrec_dlrm":
        from .dlrm_data import ensure_pinned_data as ensure_dlrm
        return ensure_dlrm(spec, path, source_dir=source_dir, batch_count=batch_count)
    if source_dir is not None:
        raise ValueError("--criteo-dir applies only to DLRM")
    supplied = path is not None
    path = Path(path) if supplied else default_data_path(spec)
    from types import SimpleNamespace
    if path.exists():
        existing, _ = load_data(SimpleNamespace(spec=spec, data_path=str(path)))
        if len(existing["batches"]) >= batch_count:
            return str(path)
        if supplied:
            raise ValueError(f"prepared artifact has {len(existing['batches'])} batches, need {batch_count}")
    if not path.exists() or not supplied:
        data = prepare_pinned(spec, batch_count=batch_count)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(data, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    data, _ = load_data(SimpleNamespace(spec=spec, data_path=str(path)))
    if len(data["batches"]) < batch_count:
        raise ValueError(f"prepared artifact has {len(data['batches'])} batches, need {batch_count}")
    return str(path)


def load_data(workload):
    if workload.spec.model == "torchrec_dlrm":
        from .dlrm_data import load_data as load_dlrm
        return load_dlrm(workload)
    if workload.spec.input_mode != "real" or not workload.data_path:
        raise ValueError("real DiT inputs require a prepared pinned-data artifact")
    path = Path(workload.data_path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data.get("schema_version") != SCHEMA or data.get("model") != workload.spec.model
            or data.get("workload_id") != workload.spec.workload_id
            or (data.get("dataset_repo"), data.get("dataset_revision"), data.get("dataset_file")) != DATASET
            or (data.get("vae_repo"), data.get("vae_revision")) != VAE
            or data.get("vae_scale") != 0.18215):
        raise ValueError("DiT data artifact does not match the pinned real-input contract")
    batches, selection = data.get("batches", []), data.get("selection", [])
    if not batches or data.get("examples") != len(batches) * workload.spec.batch_size or len(selection) != data["examples"]:
        raise ValueError("DiT data artifact has inconsistent batch or selection counts")
    for index, row in enumerate(selection):
        image_hash = row.get("image_sha256", "")
        if (row.get("row") != index or row.get("label") not in range(len(CLASS_IDS))
                or type(row.get("flipped")) is not bool or len(image_hash) != 64
                or any(char not in "0123456789abcdef" for char in image_hash)):
            raise ValueError("DiT data artifact has invalid image selection metadata")
    a, b = workload.spec.arch, workload.spec.batch_size
    shape = (b, a["in_channels"], a["input_size"], a["input_size"])
    expected_shapes = {"latent": shape, "noise": shape, "timesteps": (b,), "labels": (b,), "drop_mask": (b,)}
    for i, batch in enumerate(batches):
        if set(batch) != set(expected_shapes) or any(tuple(batch[k].shape) != s for k, s in expected_shapes.items()):
            raise ValueError("DiT batch shape differs from the specialist contract")
        if any(batch[k].dtype != getattr(torch, workload.spec.dtype) for k in ("latent", "noise")) or any(
            batch[k].dtype != torch.int64 for k in ("timesteps", "labels", "drop_mask")
        ):
            raise ValueError("DiT batch dtype differs from the specialist contract")
        rows = selection[i*b:(i+1)*b]
        if not torch.equal(batch["labels"], torch.tensor([CLASS_IDS[row["label"]] for row in rows])):
            raise ValueError("Imagenette labels differ from the ImageNet class mapping")
        if (batch["timesteps"] < 0).any() or (batch["timesteps"] >= a["diffusion_steps"]).any():
            raise ValueError("diffusion timestep outside the declared range")
        if not ((batch["drop_mask"] == 0) | (batch["drop_mask"] == 1)).all() or not all(
            torch.isfinite(v).all() for v in batch.values()
        ):
            raise ValueError("invalid DiT batch values")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return data, h.hexdigest()


# ── verification: a frozen latent pool and per-step batches of one data order ──

POOL_SCHEMA = "evograd-dit-latent-pool/1"
POOL_RULE = ("every Imagenette 320px train row: RGB, _crop to 256, mirrored iff "
             "sha256('dit-pool/flip/<row>')[0] is odd, VAE posterior mean + std * N(0, 1) drawn "
             "from torch.Generator('cpu').manual_seed(row), times 0.18215; encoded 8 rows at a "
             "time in float32")
STREAM_RULE = ("data order o, step i: epoch e = i // (n // B) is torch.randperm(n) under the seed "
               "of 'dit/order<o>/epoch<e>'; the batch is its rows [(i mod (n // B)) * B, +B); "
               "noise ~ N(0, 1), timesteps ~ U{0..steps-1} and drop_mask ~ Bernoulli("
               "class_dropout_prob) come from a generator under the seed of 'dit/order<o>/step<i>'")


def seed_of(text):
    """A 63-bit generator seed from a name: stable across processes and machines."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") & (2**63 - 1)


def default_pool_path():
    return Path(os.environ.get("EVOGRAD_DIT_LATENT_POOL",
                               Path(prepared_dir("dit_s_2")) / "latent_pool.pt"))


def read_imagenette():
    """``[(row, image bytes, label index)]`` of the pinned shard, read with pyarrow."""
    import json

    import pyarrow.parquet as pq

    table = pq.read_table(_dataset_file())
    features = json.loads((table.schema.metadata or {}).get(b"huggingface", b"{}"))
    names = tuple(features.get("info", {}).get("features", {}).get("label", {}).get("names", ()))
    if names != CLASS_NAMES:
        raise ValueError(f"Imagenette ClassLabel order changed: {names}")
    images, labels = table.column("image").to_pylist(), table.column("label").to_pylist()
    return [(i, image["bytes"], int(label)) for i, (image, label) in enumerate(zip(images, labels))]


def build_latent_pool(path=None, *, device="cuda", chunk=8):
    """Encode every training image once with the pinned VAE; write the pool atomically."""
    import io

    import numpy as np
    from diffusers.models import AutoencoderKL
    from PIL import Image, ImageOps

    path = Path(path) if path is not None else default_pool_path()
    rows = read_imagenette()
    vae = AutoencoderKL.from_pretrained(str(_vae_path()), local_files_only=True,
                                       use_safetensors=True).to(device).eval()
    vae.requires_grad_(False)
    latents, selection = [], []
    try:
        for start in range(0, len(rows), chunk):
            pixels, eps = [], []
            for row, data, label in rows[start:start + chunk]:
                image = Image.open(io.BytesIO(data)).convert("RGB")
                image_hash = hashlib.sha256(image.tobytes()).hexdigest()
                image = _crop(image)
                flipped = bool(hashlib.sha256(f"dit-pool/flip/{row}".encode()).digest()[0] & 1)
                if flipped:
                    image = ImageOps.mirror(image)
                pixels.append(torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1))
                eps.append(torch.randn((4, 32, 32), generator=torch.Generator("cpu").manual_seed(row)))
                selection.append({"row": row, "label": label, "image_sha256": image_hash,
                                  "flipped": flipped})
            x = torch.stack(pixels).to(device=device, dtype=torch.float32).div_(127.5).sub_(1)
            with torch.no_grad():
                dist = vae.encode(x).latent_dist
                z = (dist.mean + dist.std * torch.stack(eps).to(device)) * 0.18215
            latents.append(z.float().cpu())
    finally:
        del vae
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    pool = {"schema": POOL_SCHEMA, "dataset": list(DATASET), "vae": list(VAE), "vae_scale": 0.18215,
            "rule": POOL_RULE, "latents": torch.cat(latents),
            "labels": torch.tensor([CLASS_IDS[s["label"]] for s in selection]),
            "selection": selection}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(pool, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


_POOLS = {}


def load_latent_pool(path=None):
    path = Path(path) if path is not None else default_pool_path()
    key = str(path.resolve())
    if key not in _POOLS:
        if not path.is_file():
            raise FileNotFoundError(f"no DiT latent pool at {path}; build it with "
                                    "build_latent_pool() on a GPU")
        pool = torch.load(path, map_location="cpu", weights_only=True)
        if (pool.get("schema") != POOL_SCHEMA or tuple(pool.get("dataset", ())) != DATASET
                or tuple(pool.get("vae", ())) != VAE or pool.get("rule") != POOL_RULE
                or pool["latents"].shape[1:] != (4, 32, 32)
                or pool["labels"].shape != (pool["latents"].shape[0],)):
            raise ValueError(f"{path} is not a DiT latent pool of the pinned contract")
        digest_ = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest_.update(block)
        _POOLS[key] = (pool, digest_.hexdigest())
    return _POOLS[key]


class LatentStream:
    """The batch sequence of one data order: a fresh draw at every step."""

    def __init__(self, spec, order, pool_path=None):
        if spec.model != "dit_s_2" or spec.input_mode != "real":
            raise ValueError("a latent stream needs the real-input DiT workload")
        self.spec, self.order = spec, int(order)
        self.pool, self.pool_sha256 = load_latent_pool(pool_path)
        self.rows = int(self.pool["latents"].shape[0])
        self.per_epoch = self.rows // spec.batch_size
        if self.per_epoch < 1:
            raise ValueError(f"pool of {self.rows} rows is smaller than one batch")
        self._epoch = None

    def indices(self, index):
        epoch, offset = divmod(int(index), self.per_epoch)
        if self._epoch is None or self._epoch[0] != epoch:
            g = torch.Generator("cpu").manual_seed(seed_of(f"dit/order{self.order}/epoch{epoch}"))
            self._epoch = (epoch, torch.randperm(self.rows, generator=g))
        b = self.spec.batch_size
        return self._epoch[1][offset * b:(offset + 1) * b]

    def batch(self, index):
        a, b = self.spec.arch, self.spec.batch_size
        idx = self.indices(index)
        g = torch.Generator("cpu").manual_seed(seed_of(f"dit/order{self.order}/step{int(index)}"))
        shape = (b, a["in_channels"], a["input_size"], a["input_size"])
        noise = torch.randn(shape, generator=g)
        timesteps = torch.randint(a["diffusion_steps"], (b,), generator=g)
        drop_mask = (torch.rand(b, generator=g) < a["class_dropout_prob"]).long()
        dtype, device = getattr(torch, self.spec.dtype), self.spec.device
        return {"latent": self.pool["latents"][idx].to(device=device, dtype=dtype),
                "noise": noise.to(device=device, dtype=dtype), "timesteps": timesteps.to(device),
                "labels": self.pool["labels"][idx].to(device), "drop_mask": drop_mask.to(device)}

    def describe(self):
        return {"source": "imagenette latent pool", "order": self.order,
                "batch_size": self.spec.batch_size, "rule": STREAM_RULE}


def pool_identity(pool_path=None):
    pool, sha = load_latent_pool(pool_path)
    return {"dataset": list(DATASET), "vae": list(VAE), "pool_sha256": sha,
            "rows": int(pool["latents"].shape[0]), "pool_rule": POOL_RULE, "rule": STREAM_RULE}
