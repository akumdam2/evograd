"""Unified pipeline: compiler-aware, deterministic AtenIR seed synthesis.

Inductor supplies the external-library partition and initial saved-state
contract. Handwritten primitives implement the remaining ATen computation;
fusion and performance optimization belong to evolution.
"""
