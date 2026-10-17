"""DiT-S/2 generated-input training mathematics.

The architectures are local implementations of the referenced projects. No
code, weights, or data are loaded from ``references`` at runtime.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(t, width=256):
    half = width // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    phase = t.float()[:, None] * freqs[None]
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


def dit_mlp(x, w1, b1, w2, b2):
    return F.linear(F.gelu(F.linear(x, w1, b1), approximate="tanh"), w2, b2)


def dit_modulation(c, weight, bias):
    """Produce six adaLN-Zero shift, scale, and gate vectors."""
    return F.linear(F.silu(c), weight, bias)


def ada_residual(x, c, mod_weight, mod_bias, qkv_weight, qkv_bias, proj_weight,
                 proj_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, heads,
                 mlp_fn=dit_mlp, mod_fn=dit_modulation):
    h = x.shape[-1]
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod_fn(c, mod_weight, mod_bias).chunk(6, dim=-1)
    a = F.layer_norm(x, (h,), eps=1e-6)
    a = modulate(a, shift_a, scale_a)
    b, n, _ = a.shape
    qkv = F.linear(a, qkv_weight, qkv_bias).reshape(b, n, 3, heads, h // heads)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, h)
    x = x + gate_a[:, None] * F.linear(a, proj_weight, proj_bias)
    m = modulate(F.layer_norm(x, (h,), eps=1e-6), shift_m, scale_m)
    return x + gate_m[:, None] * mlp_fn(m, fc1_weight, fc1_bias, fc2_weight, fc2_bias)


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, ratio):
        super().__init__()
        self.hidden, self.heads = hidden, heads
        self.mod = nn.Linear(hidden, 6 * hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.fc1 = nn.Linear(hidden, ratio * hidden)
        self.fc2 = nn.Linear(ratio * hidden, hidden)
        self.mlp_fn = dit_mlp
        self.mod_fn = dit_modulation

    def forward(self, x, c):
        return ada_residual(x, c, self.mod.weight, self.mod.bias,
                            self.qkv.weight, self.qkv.bias, self.proj.weight,
                            self.proj.bias, self.fc1.weight, self.fc1.bias,
                            self.fc2.weight, self.fc2.bias, self.heads, self.mlp_fn,
                            self.mod_fn)


class DiTModel(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        p, h, channels = arch["patch_size"], arch["hidden_size"], arch["in_channels"]
        size = arch["input_size"]
        self.patch = nn.Conv2d(channels, h, p, stride=p)
        self.t_embedder = nn.Sequential(nn.Linear(256, h), nn.SiLU(), nn.Linear(h, h))
        self.y_embedder = nn.Embedding(arch["num_classes"] + 1, h)
        self.register_buffer("pos_embed", self._position(size // p, h), persistent=True)
        self.blocks = nn.ModuleList(DiTBlock(h, arch["num_heads"], arch["mlp_ratio"])
                                    for _ in range(arch["depth"]))
        self.final_mod = nn.Linear(h, 2 * h)
        self.final_linear = nn.Linear(h, p * p * channels * 2)
        self._initialize()

    @staticmethod
    def _position(grid, hidden):
        if hidden % 4:
            raise ValueError("DiT positional width must be divisible by four")
        yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        freqs = 1 / (10000 ** (torch.arange(hidden // 4, dtype=torch.float32) / (hidden // 4)))
        parts = [xx.reshape(-1, 1) * freqs, yy.reshape(-1, 1) * freqs]
        return torch.cat([fn(part) for part in parts for fn in (torch.sin, torch.cos)], dim=-1)[None]

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch.weight.view(self.patch.weight.shape[0], -1))
        nn.init.zeros_(self.patch.bias)
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        for i in (0, 2):
            nn.init.normal_(self.t_embedder[i].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.mod.weight)
            nn.init.zeros_(block.mod.bias)
        for layer in (self.final_mod, self.final_linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, latent, timesteps, labels, drop_mask=None):
        a = self.arch
        p = a["patch_size"]
        x = self.patch(latent).flatten(2).transpose(1, 2) + self.pos_embed
        if drop_mask is not None:
            labels = torch.where(drop_mask.bool(), a["num_classes"], labels)
        c = self.t_embedder(timestep_embedding(timesteps).to(self.patch.weight.dtype)) + self.y_embedder(labels)
        for block in self.blocks:
            x = block(x, c)
        shift, scale = F.linear(F.silu(c), self.final_mod.weight, self.final_mod.bias).chunk(2, -1)
        x = self.final_linear(modulate(F.layer_norm(x, (x.shape[-1],), eps=1e-6), shift, scale))
        b, _, _ = x.shape
        side = a["input_size"] // p
        return x.reshape(b, side, side, p, p, 2 * a["in_channels"]).permute(0, 5, 1, 3, 2, 4).reshape(
            b, 2 * a["in_channels"], a["input_size"], a["input_size"])


def _normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                  + (mean1 - mean2).square() * torch.exp(-logvar2))


def _approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def _discretized_gaussian_log_likelihood(x, mean, log_scale):
    centered = x - mean
    inv_std = torch.exp(-log_scale)
    plus = _approx_standard_normal_cdf(inv_std * (centered + 1 / 255))
    minus = _approx_standard_normal_cdf(inv_std * (centered - 1 / 255))
    return torch.where(x < -0.999, torch.log(plus.clamp_min(1e-12)),
                       torch.where(x > 0.999, torch.log((1 - minus).clamp_min(1e-12)),
                                   torch.log((plus - minus).clamp_min(1e-12))))


def diffusion_loss(model, batch):
    """Guided-diffusion epsilon MSE plus learned-range variance bound."""
    x0, noise, t = batch["latent"], batch["noise"], batch["timesteps"]
    steps = model.arch["diffusion_steps"]
    betas = torch.linspace(1e-4, 0.02, steps, device=x0.device, dtype=torch.float64)
    alphas = 1 - betas
    cumulative = alphas.cumprod(0)
    previous = torch.cat((torch.ones_like(cumulative[:1]), cumulative[:-1]))
    def at(values):
        return values[t].to(x0.dtype).reshape(-1, 1, 1, 1)
    xt = at(cumulative.sqrt()) * x0 + at((1 - cumulative).sqrt()) * noise
    pred = model(xt, t, batch["labels"], batch["drop_mask"])
    epsilon, variance = pred.chunk(2, dim=1)
    mse = (epsilon - noise).square().mean(dim=(1, 2, 3))
    posterior_variance = betas * (1 - previous) / (1 - cumulative)
    posterior_logvar = torch.log(torch.cat((posterior_variance[1:2], posterior_variance[1:])).clamp_min(1e-20))
    posterior_mean_a = betas * previous.sqrt() / (1 - cumulative)
    posterior_mean_b = (1 - previous) * alphas.sqrt() / (1 - cumulative)
    true_mean = at(posterior_mean_a) * x0 + at(posterior_mean_b) * xt
    model_x0 = (xt - at((1 - cumulative).sqrt()) * epsilon.detach()) / at(cumulative.sqrt())
    model_mean = at(posterior_mean_a) * model_x0 + at(posterior_mean_b) * xt
    model_logvar = (variance + 1) / 2 * at(torch.log(betas)) + (1 - (variance + 1) / 2) * at(posterior_logvar)
    kl = _normal_kl(true_mean, at(posterior_logvar), model_mean, model_logvar).mean(dim=(1, 2, 3)) / math.log(2)
    decoder = -_discretized_gaussian_log_likelihood(x0, model_mean, 0.5 * model_logvar).mean(dim=(1, 2, 3)) / math.log(2)
    vb = torch.where(t == 0, decoder, kl)
    return (mse + vb).mean(), pred


def dlrm_dense(x, weights, biases):
    for weight, bias in zip(weights, biases):
        x = F.relu(F.linear(x, weight, bias))
    return x


def dlrm_interaction(dense, sparse):
    """Dense vector plus strict lower triangular pairwise dot products."""
    features = torch.cat((dense[:, None], sparse), dim=1)
    dots = torch.bmm(features, features.transpose(1, 2))
    row, col = torch.tril_indices(features.shape[1], features.shape[1], offset=-1,
                                  device=features.device)
    return torch.cat((dense, dots[:, row, col]), dim=1)


class DLRMModel(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.tables = nn.ModuleList(nn.EmbeddingBag(count, arch["embedding_dim"], mode="sum")
                                    for count in arch["num_embeddings_per_feature"])
        dense_sizes = [arch["num_dense_features"], *arch["dense_arch_layer_sizes"]]
        self.dense_layers = nn.ModuleList(nn.Linear(i, o) for i, o in zip(dense_sizes, dense_sizes[1:]))
        features = arch["num_sparse_features"] + 1
        over_sizes = [arch["embedding_dim"] + features * (features - 1) // 2,
                      *arch["over_arch_layer_sizes"]]
        self.over_layers = nn.ModuleList(nn.Linear(i, o) for i, o in zip(over_sizes, over_sizes[1:]))
        self.interaction_fn = dlrm_interaction

    def forward(self, dense, sparse_ids):
        weights = [layer.weight for layer in self.dense_layers]
        biases = [layer.bias for layer in self.dense_layers]
        dense = dlrm_dense(dense, weights, biases)
        b = dense.shape[0]
        bags = []
        for i, table in enumerate(self.tables):
            width = self.arch["multi_hot_sizes"][i]
            values = sparse_ids[i, :, :width].reshape(-1)
            offsets = torch.arange(0, b * width, width, device=values.device)
            bags.append(table(values, offsets))
        x = self.interaction_fn(dense, torch.stack(bags, dim=1))
        for layer in self.over_layers[:-1]:
            x = F.relu(layer(x))
        return self.over_layers[-1](x).squeeze(-1)


def build_model(spec):
    with torch.random.fork_rng():
        torch.manual_seed(spec.seed)
        model = DiTModel(spec.arch) if spec.model == "dit_s_2" else DLRMModel(spec.arch)
    return model.to(device=spec.device, dtype=getattr(torch, spec.dtype)).train()


def make_inputs(spec, seed):
    arch = spec.arch
    g = torch.Generator(device="cpu").manual_seed(spec.data_seed + seed)
    b = spec.batch_size
    if spec.model == "dit_s_2":
        shape = (b, arch["in_channels"], arch["input_size"], arch["input_size"])
        batch = dict(latent=torch.randn(shape, generator=g), noise=torch.randn(shape, generator=g),
                     timesteps=torch.randint(arch["diffusion_steps"], (b,), generator=g),
                     labels=torch.randint(arch["num_classes"], (b,), generator=g),
                     drop_mask=(torch.rand((b,), generator=g) < arch["class_dropout_prob"]).long())
    else:
        max_hot = max(arch["multi_hot_sizes"])
        ids = torch.zeros(arch["num_sparse_features"], b, max_hot, dtype=torch.long)
        for i, count in enumerate(arch["num_embeddings_per_feature"]):
            ids[i, :, :arch["multi_hot_sizes"][i]] = torch.randint(
                count, (b, arch["multi_hot_sizes"][i]), generator=g)
        batch = dict(dense=torch.randn((b, arch["num_dense_features"]), generator=g),
                     sparse_ids=ids, labels=torch.randint(2, (b,), generator=g).float())
    return {k: v.to(device=spec.device, dtype=getattr(torch, spec.dtype) if v.is_floating_point() else None)
            for k, v in batch.items()}


def forward_loss(model, batch, model_name):
    if model_name == "dit_s_2":
        return diffusion_loss(model, batch)
    if model_name == "torchrec_dlrm":
        logits = model(batch["dense"], batch["sparse_ids"])
        return F.binary_cross_entropy_with_logits(logits, batch["labels"]), logits
    raise ValueError(model_name)
