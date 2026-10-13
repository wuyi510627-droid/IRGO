"""Run the archived 3 x 512 text_rich evaluation on a final HF model export."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

from common import REPO, runtime_environment, sha256, verify_webshop
from prepare_data import metadata_rows


def evaluation_config(model, output, seed):
    """Validation is deliberately independent of intervention and policy updates."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    # Use the archived evaluation overrides, not training runtime defaults:
    # the independent evaluation intentionally has a different prompt limit,
    # horizon and inference memory layout.
    overrides = json.loads((REPO / "configs/irgo/evaluation_overrides.json").read_text())
    paths = {
        "data.train_files": output.resolve() / "data/train.parquet",
        "data.val_files": output.resolve() / "data/test.parquet",
        "actor_rollout_ref.model.path": model.resolve(),
        "trainer.default_local_dir": output.resolve() / f"seed{seed}/run",
    }
    overrides += [f"{key}={json.dumps(str(value))}" for key, value in paths.items()]
    overrides += [f"env.seed={seed}", f"trainer.experiment_name=irgo_step150_seed{seed}"]
    with initialize_config_dir(config_dir=str(REPO / "recipe/GraphGPO/config"), version_base=None):
        cfg = compose(config_name="graphgpo_trainer", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def parse_result(log, seed, returncode):
    # Read the full-precision final dict, not the rounded console step metrics.
    matches = re.findall(r"['\"]val/success_rate['\"]\s*:\s*([0-9.eE+-]+)", log)
    if returncode != 0 or not matches:
        raise RuntimeError(f"Seed {seed} did not finish successfully; inspect its log")
    rate = float(matches[-1])
    successes = round(rate * 512)
    if abs(successes / 512 - rate) > 1e-10 or not 0 <= successes <= 512:
        raise ValueError("Result does not match a 512-episode evaluation")
    # A rate from only part of the dataset must never be called a completed seed.
    if not re.search(r"4/4", log):
        raise ValueError("Missing completion of all four validation batches")
    return dict(
        seed=seed, episodes=512, successes=successes, failures=512 - successes, success_rate=rate
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Exported final model, not initial weights"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    jobs = {seed: evaluation_config(args.model, args.output, seed) for seed in (123, 456, 789)}
    if not args.execute:
        print(
            json.dumps(
                dict(
                    environment_seeds=list(jobs),
                    episodes_per_seed=512,
                    max_steps=30,
                    intervention=False,
                    training_updates=0,
                    model=str(args.model),
                    execute=False,
                ),
                indent=2,
            )
        )
        return
    if args.output.exists():
        raise FileExistsError("Use a new evaluation output directory")
    weights = list(args.model.glob("*.safetensors"))
    if not weights or not (args.model / "config.json").is_file():
        raise FileNotFoundError("Expected a final Hugging Face model export")
    assets = verify_webshop()
    import os
    import torch
    from datasets import Dataset

    env = runtime_environment("evaluate")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", env["CUDA_VISIBLE_DEVICES"])
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Select exactly two CUDA GPUs")
    args.output.mkdir(parents=True)
    data = args.output / "data"
    data.mkdir()
    for split, count in [("train", 16), ("test", 512)]:
        Dataset.from_list(metadata_rows(split, count)).to_parquet(str(data / f"{split}.parquet"))
    hashes = {p.name: sha256(p) for p in weights}
    (args.output / "protocol.json").write_text(
        json.dumps(
            dict(
                environment_seeds=list(jobs),
                episodes_per_seed=512,
                products=1000,
                observation_mode="text_rich",
                max_steps=30,
                model_sha256=hashes,
                data_sha256=assets,
                intervention=False,
                training_updates=0,
            ),
            indent=2,
        )
    )
    env.setdefault("WANDB_DIR", str(args.output.resolve() / "wandb"))
    Path(env["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    runs = []
    for seed, cfg in jobs.items():
        config_dir = args.output / f"seed{seed}"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        command = [
            sys.executable,
            "-u",
            "-m",
            "recipe.GraphGPO.main_graphgpo",
            "--config-path",
            str(config_dir.resolve()),
            "--config-name",
            "config",
        ]
        logfile = args.output / f"output_seed{seed}.log"
        with logfile.open("w") as stream:
            result = subprocess.run(
                command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        (args.output / f"exit_seed{seed}.json").write_text(
            json.dumps({"returncode": result.returncode})
        )
        runs.append(parse_result(logfile.read_text(), seed, result.returncode))
        # Publish each completed seed immediately while the rest are still running.
        (args.output / "results.json").write_text(
            json.dumps({"complete": False, "runs": runs}, indent=2)
        )
        print(json.dumps(runs[-1]), flush=True)
    if hashes != {p.name: sha256(p) for p in weights}:
        raise RuntimeError("Model weights changed during evaluation")
    total = sum(r["successes"] for r in runs)
    (args.output / "results.json").write_text(
        json.dumps(
            dict(
                complete=True,
                runs=runs,
                total_successes=total,
                total_episodes=1536,
                success_rate=total / 1536,
                weights_unchanged=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
