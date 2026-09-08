"""Level-2 verification, calibration and controls for the four Qwen3 sites.

One module per site, because the four boundaries have genuinely different
tensors, gradients and gates; plus the shared calibration inventory, the
negative controls that show a calibrated tolerance still rejects a wrong
kernel, and one CLI that routes to all of them.
"""
