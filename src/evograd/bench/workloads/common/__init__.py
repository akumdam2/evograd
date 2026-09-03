"""Machinery every workload shares, owning nothing model-specific.

A workload package under ``bench/workloads/<name>/`` says what *its* model is:
the architecture, the classes its boundaries live on, the adapters that swap a
kernel into it, and the snapshot its own harvest produced. Everything that is
the same procedure whichever model is being described lives here -- reading a
snapshot, aggregating observations into a manifest, capturing and replaying a
layer, measuring a tolerance floor.

The split is by *what a module knows*, not by what it does. A module belongs
here when it would read identically for a second architecture; it belongs in the
workload package when it names a class, a module path, or a shape.

Nothing here may import a workload package. The dependency runs one way, which
is what stops shared machinery from quietly growing a branch on a model name.
"""
