from evograd.benchmark.topdown.common.vision.harvest import snapshot_main
from evograd.benchmark.topdown import load_snapshot


def load():
    return load_snapshot("convnextv2_tiny")


if __name__ == "__main__":
    snapshot_main("convnextv2_tiny")
