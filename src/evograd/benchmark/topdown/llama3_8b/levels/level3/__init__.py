"""One decoder layer, captured once and replayed without a GPU model.

Layer 16's inputs, parameters and gradients are saved to a ``.pt`` artifact so
the level-2 operators can be derived and checked against the tensors the real
model produced, rather than against synthetic ones.

**Nothing here is tracked.** A capture is a several-hundred-megabyte tensor
file, and it is a *result*: it is produced by running the canonical Llama-3-8B
step on a GPU, and it is reproducible from the spec plus the seed. No such run
has been executed, so ``results/llama3-level4/layer16.pt`` does not exist and
every level-2 derivation that consumes it refuses by name rather than
substituting synthetic tensors.
"""
