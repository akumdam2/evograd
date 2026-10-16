from evograd.benchmark.topdown.common.vision.operators import bindings


def install(model, kernels):
    """Replace forward methods only; retain parameter names, identity and ties."""
    import types
    from evograd.evaluation.tier3.patch import PatchProvenance

    paths = {}
    for base_site, path, module, _, pack in list(bindings(model)):
        site = (
            f"convnext_mlp_stage{path.split('.stages.')[1].split('.')[0]}"
            if base_site == "convnext_mlp"
            else base_site
        )
        if site not in kernels.patched:
            continue
        fn = kernels.kernel_for(site)
        if base_site == "convnext_mlp":

            def call(self, x, _fn=fn, _pack=pack):
                h = self.layernorm(self.dwconv(x).permute(0, 2, 3, 1))
                return x + self.drop_path(_fn(*_pack(h)).permute(0, 3, 1, 2))

        else:

            def call(self, x, _fn=fn, _pack=pack):
                return _fn(*_pack(x))

        module.forward = types.MethodType(call, module)
        paths.setdefault(site, []).append(path)
    if set(paths) != set(kernels.patched):
        raise ValueError(f"unmatched requested sites: {set(kernels.patched)-set(paths)}")
    return PatchProvenance(
        "module_surgery",
        tuple(kernels.patched),
        tuple(sorted(paths)),
        {key: tuple(value) for key, value in paths.items()},
    )


def preflight_cases(spec, site):
    """Use matching observed layouts, otherwise explicit configuration-derived shapes.

    A custom batch/image/text size must never reuse the canonical site's grid.
    These fallback cases are for verification, not observed timing suites.
    """
    from evograd.benchmark.topdown import has_snapshot, load_snapshot
    from evograd.benchmark.topdown.common.vision.contracts import observed
    from evograd.opdecl import Workload, Provenance

    arch = spec.arch
    if site.startswith("convnext_mlp_stage"):
        dims = [
            dict(
                B=spec.batch_size,
                Y=spec.image_size // (4 * 2**i),
                X=spec.image_size // (4 * 2**i),
                H=width,
                I=4 * width,
            )
            for i, width in enumerate(arch["hidden_sizes"])
            if i == int(site.removeprefix("convnext_mlp_stage"))
        ]
    elif site == "text_mlp":
        dims = [
            dict(
                B=spec.batch_size,
                T=spec.seq_len,
                H=arch["hidden_size"],
                I=arch["intermediate_size"],
            )
        ]
    else:
        vision = arch["vision_config"]
        hidden = vision["embed_dim"]
        dims = [
            dict(
                T=spec.batch_size * (spec.image_size // vision["patch_size"]) ** 2,
                H=hidden,
                I=int(hidden * vision.get("mlp_ratio", 4)),
            )
        ]
    if has_snapshot(spec.model) and load_snapshot(spec.model)["workload_id"] == spec.workload_id:
        task_site = "convnext_mlp" if site.startswith("convnext_mlp_stage") else site
        cases = tuple(
            case for case in observed(spec.model, spec.model + "_" + task_site) if case.dims in dims
        )
        if not cases:
            raise ValueError(f"{site}: matching snapshot has no matching specialization")
        return cases
    return tuple(
        Workload(
            dims=d,
            dtype=spec.dtype,
            provenance=Provenance(
                model=spec.model,
                component=site,
                source="specification",
                note="Derived from pinned architecture plus current execution configuration; not a harvested timing case",
            ),
        )
        for d in dims
    )
