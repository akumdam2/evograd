# Result layout

The package reorganization does not move or overwrite historical experiment
output. Existing repository `results/` and `/u/wzhan/.cache/evograd-mF` paths
remain read-only evidence and are interpreted at the paths that produced them.

New runs use this layout:

```text
results/
├── benchmark/
│   ├── operator_suite/<run>/...
│   └── topdown/<workload>/<run>/...
└── evaluation/
    ├── tier1/<workload-or-op>/<run>/...
    ├── tier2/<workload-or-op>/<run>/...
    └── tier3/<workload>/<run>/...
```

Legacy directories such as `results/qwen3-level4/` are path indexes for old
runs and must not be renamed in bulk. Report schemas, fields, and provenance
remain authoritative; directory names organize results but do not redefine
them.
