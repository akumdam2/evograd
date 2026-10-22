"""Scope adapters: they build and run a scope; they never decide a verdict.

    operator.py  OperatorScope -- an OpDecl as an nn.Module (native / artifact / compile)
    payload.py   PayloadOperatorScope -- the same operator scope on recorded training payloads
    module.py    ModuleScope -- any model built from named sites (model scope)
    model.py     model_scope(...) (generic, loss-seeded), WorkloadModelScope (next-token LM
                 bridge), ObjectiveModelScope (a workload's own loss and primary output)

Model packages are imported inside the methods that need them.
"""
