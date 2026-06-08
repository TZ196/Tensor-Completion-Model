import argparse
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def run_command(command):
    print("Running:", " ".join(command))
    subprocess.check_call(command, cwd=SCRIPT_DIR)


def default_split_path(observed_ratio, val_ratio, seed):
    return os.path.join(
        PROJECT_DIR,
        "splits",
        "random_observed%d_val%d_seed_%d.npz" % (
            int(round(observed_ratio * 100)),
            int(round(val_ratio * 100)),
            seed,
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the multimodal text preparation pipeline in a strict order: "
            "split -> train-only stats -> DeepSeek texts -> embeddings."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["stats", "deepseek", "encode", "all"],
        default="all",
    )
    parser.add_argument(
        "--tensor-path",
        default=os.path.join(PROJECT_DIR, "sat_path_bytes_mb_tensor.npy"),
    )
    parser.add_argument(
        "--topology-path",
        default=os.path.join(
            PROJECT_DIR,
            "sat_connectivity_tensor_dynamic_60s_1000ms.npz",
        ),
    )
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "text_data"))
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--missing-rate", type=float, default=0.9)
    parser.add_argument("--observed-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--create-split",
        action="store_true",
        help="Create the fixed global random split explicitly if needed.",
    )
    parser.add_argument(
        "--overwrite-split",
        action="store_true",
        help="Recreate split even if it already exists.",
    )
    parser.add_argument("--env-path", default=os.path.join(SCRIPT_DIR, "deepseek.env"))
    parser.add_argument(
        "--config-path",
        default=os.path.join(SCRIPT_DIR, "experiment_description.md"),
    )
    parser.add_argument(
        "--endo-source",
        choices=["topo", "topoflow"],
        default="topo",
    )
    parser.add_argument("--embedding-dim", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    observed_ratio = args.observed_ratio
    if observed_ratio is None:
        observed_ratio = 1.0 - args.missing_rate
    split_path = args.split_path
    if split_path is None:
        split_path = default_split_path(observed_ratio, args.val_ratio, args.seed)

    stats_path = os.path.join(args.output_dir, "time_stats_train_only.json")
    endo_path = os.path.join(args.output_dir, "endo_texts.json")
    exo_path = os.path.join(args.output_dir, "exo_text_segments.json")

    if args.stage in ("stats", "all"):
        command = [
            sys.executable,
            "build_satellite_texts.py",
            "--tensor-path",
            args.tensor_path,
            "--topology-path",
            args.topology_path,
            "--output-dir",
            args.output_dir,
            "--split-path",
            split_path,
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
        ]
        if args.observed_ratio is not None:
            command.extend(["--observed-ratio", str(args.observed_ratio)])
        else:
            command.extend(["--missing-rate", str(args.missing_rate)])
        if args.create_split:
            command.append("--create-split")
        if args.overwrite_split:
            command.append("--overwrite-split")
        run_command(command)

    if args.stage in ("deepseek", "all"):
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                "Missing stats file: %s. Run --stage stats first." % stats_path
            )
        command = [
            sys.executable,
            "build_texts_deepseek.py",
            "--env-path",
            args.env_path,
            "--mode",
            "both",
            "--config-path",
            args.config_path,
            "--stats-path",
            stats_path,
            "--endo-source",
            args.endo_source,
            "--exo-output-path",
            exo_path,
            "--endo-output-path",
            endo_path,
        ]
        run_command(command)

    if args.stage in ("encode", "all"):
        if not os.path.exists(endo_path):
            raise FileNotFoundError(
                "Missing endogenous text file: %s. Run --stage deepseek first."
                % endo_path
            )
        if not os.path.exists(exo_path):
            raise FileNotFoundError(
                "Missing exogenous text file: %s. Run --stage deepseek first."
                % exo_path
            )
        command = [
            sys.executable,
            "encode_satellite_texts.py",
            "--text-dir",
            args.output_dir,
            "--dim",
            str(args.embedding_dim),
        ]
        run_command(command)

    print("Pipeline stage completed:", args.stage)
    print("Split path:", split_path)
    print("Text data dir:", args.output_dir)


if __name__ == "__main__":
    main()
