"""Read each payload storage once per case, returning private device copies."""
from contextlib import contextmanager
import types


@contextmanager
def case_payload_cache(corpora):
    import torch

    storages, originals = {}, []
    for corpus in {id(item): item for item in corpora}.values():
        original = corpus._read_storage
        had_override = "_read_storage" in corpus.__dict__
        previous = corpus.__dict__.get("_read_storage")
        originals.append((corpus, had_override, previous))

        def cached(self, key, device, _original=original):
            identity = (str(self.manifest_path.resolve()), self.sha256, key)
            if identity not in storages:
                storages[identity] = _original(key, "cpu")
            storage = storages[identity]
            source = torch.empty(0, dtype=torch.uint8, device="cpu").set_(storage, 0, (storage.nbytes(),), (1,))
            return source.to(device=device, copy=True).untyped_storage()

        corpus._read_storage = types.MethodType(cached, corpus)
    try:
        yield storages
    finally:
        for corpus, had_override, previous in reversed(originals):
            if had_override:
                corpus._read_storage = previous
            else:
                del corpus._read_storage
        storages.clear()
