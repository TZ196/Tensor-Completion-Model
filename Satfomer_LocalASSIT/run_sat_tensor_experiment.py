import argparse
import importlib.util
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention-mode", choices=["dense", "local"], default="local")
    parser.add_argument("--attention-neighbor-size", type=int, default=3)
    known, remaining = parser.parse_known_args()

    os.environ["SATFORMER_ATTENTION_MODE"] = known.attention_mode
    os.environ["SATFORMER_ATTENTION_NEIGHBOR_SIZE"] = str(known.attention_neighbor_size)

    here = Path(__file__).resolve().parent
    parent_project = here.parent / "Satfomer"
    parent_runner = parent_project / "run_sat_tensor_experiment.py"

    if "--tensor-path" not in remaining and not (here / "sat_path_bytes_mb_tensor.npy").exists():
        remaining.extend(["--tensor-path", str(parent_project / "sat_path_bytes_mb_tensor.npy")])
    if (
        "--adjacency-path" not in remaining
        and not (here / "sat_connectivity_tensor_dynamic_60s_1000ms.npz").exists()
    ):
        remaining.extend([
            "--adjacency-path",
            str(parent_project / "sat_connectivity_tensor_dynamic_60s_1000ms.npz"),
        ])

    sys.argv = [str(parent_runner)] + remaining
    sys.path.insert(0, str(here))

    spec = importlib.util.spec_from_file_location("satformer_parent_runner", parent_runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
