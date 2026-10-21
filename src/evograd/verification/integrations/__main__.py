"""``python -m evograd.verification.integrations <model> {capture,fidelity,verify} ...``."""

import sys

from . import PACKAGES, load
from .cli import main

if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
    sys.exit(f"usage: python -m evograd.verification.integrations <model> ...; models: {sorted(PACKAGES)}")
sys.exit(main(load(sys.argv[1]), sys.argv[2:]))
