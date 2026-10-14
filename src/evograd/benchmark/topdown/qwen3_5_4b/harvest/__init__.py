"""Where the Qwen3.5 shapes come from: one observed whole-model training step.

The canonical Level-3 step (B1 x T2048) runs once with the observer attached;
the manifest is frozen into ``snapshot.json`` (provenance ``observed``). The
Level-1 observed suite ``qwen3_5_4b_observed`` is built from it,
:func:`.snapshot.check_declared_dims` holds every Level-2 site's declared dims to
it, and the whole-model call counts (24 GDN / 8 attention / 32 per layer) are
read from it rather than derived.
"""
