SITES = {f"{site}_stage{i}": f"convnextv2_tiny_{site}"
         for site in ("convnext_mlp", "dwconv_layernorm") for i in range(4)}
