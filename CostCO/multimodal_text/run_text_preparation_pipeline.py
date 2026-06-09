import argparse
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def run_command(command):
    print("Running:", " ".join(command))
    subprocess.check_call(command, cwd=SCRIPT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the multimodal text preparation pipeline in a strict order: "
            "topology-only stats -> DeepSeek texts -> embeddings."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["stats", "deepseek", "encode", "all"],
        default="all",
    )
    parser.add_argument(
        "--topology-path",
        default=os.path.join(
            PROJECT_DIR,
            "sat_connectivity_tensor_dynamic_60s_1000ms.npz",
        ),
    )
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "text_data"))
    parser.add_argument("--env-path", default=os.path.join(SCRIPT_DIR, "deepseek.env"))
    parser.add_argument(
        "--config-path",
        default=os.path.join(SCRIPT_DIR, "experiment_description.md"),
    )
    parser.add_argument(
        "--endo-source",
        choices=["topo"],
        default="topo",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--endo-chunk-size", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    topo_stats_path = os.path.join(args.output_dir, "time_stats_topo_only.json")
    stats_path = topo_stats_path
    endo_path = os.path.join(args.output_dir, "endo_texts.json")
    exo_path = os.path.join(args.output_dir, "exo_text_segments.json")

    if args.stage in ("stats", "all"):
        command = [
            sys.executable,
            "build_satellite_texts.py",
            "--topology-path",
            args.topology_path,
            "--output-dir",
            args.output_dir,
        ]
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
            "--endo-chunk-size",
            str(args.endo_chunk_size),
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
            "--model-name",
            args.embedding_model,
            "--batch-size",
            str(args.embedding_batch_size),
        ]
        run_command(command)

    print("Pipeline stage completed:", args.stage)
    print("Stats path:", stats_path)
    print("Text data dir:", args.output_dir)


if __name__ == "__main__":
    main()
