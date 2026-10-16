"""Functions whose code constants are unordered or nested, for fingerprint tests.

Imported in fresh subprocesses under different ``PYTHONHASHSEED`` values; the
digest of each must not depend on the seed.
"""

SCALE = 2.0


def set_member(x, mode):
    return x * 2 if mode in {"alpha", "beta", "gamma", "delta"} else x


def nested_constants(x):
    # Folded into one frozenset constant whose members are tuples: an unordered
    # container of ordered ones.
    return x in {("a", 1), ("b", 2.5), ("c", None), ("d", "e")}


def nested_code(x):
    def inner(y):
        return y if y in {"k1", "k2", "k3"} else 0

    return [inner(v) for v in (x,) if v not in {"m", "n", "o"}]


def reaches_global(x):
    return x * SCALE
