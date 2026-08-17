"""Level 1 — primitive operators: one mathematical operation plus the saved
state its backward needs.

A grouping package, not a declaration: it exposes no ``op``, and the registry in
``evograd.ops`` recurses through it. ``OpDecl.level`` remains the authority on
which level an operator belongs to; ``tests/test_ops_layout.py`` asserts this
directory and that field agree.
"""
