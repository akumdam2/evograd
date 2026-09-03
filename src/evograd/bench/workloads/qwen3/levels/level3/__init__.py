"""One decoder layer, captured once and replayed without a GPU model.

Layer 14's inputs, parameters and gradients are saved to a ``.pt`` artifact so
the level-2 operators can be derived and checked against the tensors the real
model produced, rather than against synthetic ones.
"""
