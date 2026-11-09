"""Operator declaration: partial rotary embedding (``R`` of ``D`` channels).

The generic ``rope`` primitive rotates the whole head; this one rotates only
the leading ``R`` channels and passes the rest through, which is what models
with ``partial_rotary_factor < 1`` run. A kernel that rotated all ``D``
channels would be wrong on three quarters of every head at a factor of 0.25.
"""

from evograd.opdecl import Active, Inactive, declare_op
from evograd.ops._common import dtype_for, is_head_major_view


_DIMS = ("B", "H", "T", "D", "R")


#: A rotation base for the generated tables of a case with no model behind it.
#: The operator does not depend on it -- cos/sin are inputs -- but a random pair
#: would not be a rotation.
_THETA = 10000.0


def rotation_base(workload) -> float:
    """The theta a case's tables are generated at: its model's ``rope_theta``
    when the provenance names a registered model (1e7 for
    Qwen3.5), else :data:`_THETA`. An observed case then carries the
    frequencies the model rotates with, not a generic base."""
    provenance = getattr(workload, "provenance", None)
    if provenance is None:
        return _THETA
    from evograd.opdecl.models import MODELS

    record = MODELS.get(getattr(provenance, "model", None))
    theta = getattr(record, "rope_theta", None)
    return float(theta) if theta else _THETA


def make_partial_rope_inputs(torch, op, workload, device="cuda"):
    """Head-major input, a non-contiguous view where the model's is one, and
    real rotary tables of width ``R`` at the model's own theta."""
    dims = workload.dims
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(dims["B"] * 100003 + dims["H"] * 1009 + dims["T"] * 31 + dims["R"])
    shape = (dims["B"], dims["H"], dims["T"], dims["D"])
    if is_head_major_view(workload):
        x = torch.randn((dims["B"], dims["T"], dims["H"], dims["D"]),
                        device=device, dtype=dtype).transpose(1, 2)
    else:
        x = torch.randn(shape, device=device, dtype=dtype)
    rotary = dims["R"]
    inv_freq = 1.0 / (rotation_base(workload) ** (torch.arange(0, rotary, 2, device=device,
                                                               dtype=torch.float32) / rotary))
    position = torch.arange(dims["T"], device=device, dtype=torch.float32)[:, None]
    angles = position * inv_freq[None, :]
    table = torch.cat((angles, angles), dim=-1)[None]
    return {
        "x": x,
        "cos": table.cos().to(dtype),
        "sin": table.sin().to(dtype),
        "dy": torch.randn(shape, device=device, dtype=dtype),
    }


op = declare_op(
    name="partial_rope",
    level=1,
    family="positional",
    forward="evograd.ops.level1.partial_rope.forward_ref:partial_rope_forward_ref",
    runtime_forward=(
        "evograd.ops.level1.partial_rope.forward_ref:partial_rope_forward_production"
    ),
    dims=_DIMS,
    args=(
        Active("x", "[B, H, T, D]"),
        Inactive("cos", "[1, T, R]", note="rotary cosine table, one row of positions"),
        Inactive("sin", "[1, T, R]", note="rotary sine table, same layout as cos"),
    ),
    output=Active("y", "[B, H, T, D]"),
    parameter_args=(),
    forward_semantics=(
        "Rotate the leading R channels of every head and pass the rest through. "
        "With cos and sin unsqueezed at dim 1 (broadcast over heads), "
        "xr = x[..., :R], rotate_half(t) = cat((-t[..., R/2:], t[..., :R/2]), -1): "
        "y[..., :R] = xr * cos + rotate_half(xr) * sin and y[..., R:] = x[..., R:]. "
        "Return a contiguous [B, H, T, D] tensor in x's dtype. Do not call "
        "autograd in the generated math."
    ),
    backward_semantics=(
        "Return dx. On the rotary slice, dx[..., :R] = dy_r * cos - "
        "rotate_half(dy_r * sin) with dy_r = dy[..., :R] (rotate_half is its own "
        "negative inverse); dx[..., R:] = dy[..., R:]. cos and sin are inactive "
        "and receive no gradient."
    ),
    extra_constraints=(
        "x may be a non-contiguous head-major view (a decoder reaches it by "
        "transposing [B, T, H, D]); the output is contiguous either way. R is "
        "even and at most D."
    ),
    memory_inputs=("x",),
    make_inputs=make_partial_rope_inputs,
)
