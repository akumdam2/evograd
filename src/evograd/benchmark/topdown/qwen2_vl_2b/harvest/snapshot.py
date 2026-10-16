from evograd.benchmark.topdown.common.vision.harvest import snapshot_main
from evograd.benchmark.topdown import load_snapshot


def load():
    return load_snapshot("qwen2_vl_2b")


if __name__ == "__main__":
    snapshot_main("qwen2_vl_2b")
