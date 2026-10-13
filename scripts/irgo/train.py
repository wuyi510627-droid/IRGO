"""Prepare the archived IRGO training config; add --execute to start training."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from common import (
    BASE_MODEL_HASH,
    REPO,
    runtime_environment,
    sha256,
    training_config,
    verify_webshop,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Local initial HF model, not a trained checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing train.parquet and test.parquet.",
    )
    parser.add_argument("--output", type=Path, required=True, help="A new experiment directory.")
    parser.add_argument("--profile", choices=["full150", "smoke"], default="full150")
    parser.add_argument(
        "--allow-different-model",
        action="store_true",
        help="Explicitly mark a new experiment using other initial weights.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = training_config(args.model, args.data_dir, args.output, args.profile)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "rounds": cfg["trainer"]["total_training_steps"],
                "natural_tasks": cfg["data"]["train_batch_size"],
                "rollouts_per_task": cfg["env"]["rollout"]["n"],
                "max_probes": cfg["algorithm"]["u"]["max_probes"],
                "output": str(args.output.resolve()),
                "execute": args.execute,
            },
            indent=2,
        )
    )
    if not args.execute:
        return
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; archived runs are never overwritten.")
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError("Expected a local Hugging Face model directory")
    weight = args.model / "model.safetensors"
    same_initial = weight.is_file() and sha256(weight) == BASE_MODEL_HASH
    if not same_initial and not args.allow_different_model:
        raise ValueError(
            "Initial weights do not match the archived baseline; see --allow-different-model."
        )
    from pyarrow import parquet

    rows = {}
    for split, batch in [
        ("train", cfg["data"]["train_batch_size"]),
        ("test", cfg["data"]["val_batch_size"]),
    ]:
        file = args.data_dir / f"{split}.parquet"
        rows[split] = parquet.read_metadata(file).num_rows
        if rows[split] < batch:
            raise ValueError(f"{file} has fewer rows than the configured batch")
    assets = verify_webshop()
    import torch

    env = runtime_environment(args.profile)
    # Apply GPU selection before checking CUDA, using the same value the child
    # process receives. CUDA_VISIBLE_DEVICES normally comes from the shell.
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", env["CUDA_VISIBLE_DEVICES"])
    if torch.cuda.device_count() != 2:
        raise RuntimeError("This reference configuration requires two visible CUDA GPUs")
    args.output.mkdir(parents=True)
    (args.output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (args.output / "launch.json").write_text(
        json.dumps(
            {
                "same_initial_model_as_archived_run": same_initial,
                "metadata_rows": rows,
                "data_sha256": {
                    f"{s}.parquet": sha256(args.data_dir / f"{s}.parquet") for s in rows
                },
                "webshop_sha256": assets,
                "profile": args.profile,
            },
            indent=2,
        )
    )
    env.setdefault("WANDB_DIR", str(args.output.resolve() / "wandb"))
    Path(env["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-B",
        "-u",
        "-m",
        "recipe.GraphGPO.main_graphgpo",
        "--config-path",
        str(args.output.resolve()),
        "--config-name",
        "config",
    ]
    with (args.output / "train.log").open("w") as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    (args.output / "exitcode").write_text(str(result.returncode) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
