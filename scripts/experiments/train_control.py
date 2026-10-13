"""Prepare or run frozen P/N/R/F/F6 controls. No execution without --execute."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/irgo"))
from common import training_config, sha256
from train import launch
from recipe.GraphGPO.task_protocol import FixedTaskSchedule


def control_config(arm, model, data_dir, output, manifest, seed, profile="full150"):
    if arm not in ("P", "N", "R", "F", "F6") or profile not in ("full150", "smoke"):
        raise ValueError("Unknown arm or profile")
    schedule = FixedTaskSchedule(manifest, True, 16, seed)
    cfg = training_config(model, data_dir, output)
    cfg["env"]["seed"] = seed
    cfg["env"]["rollout"]["n"] = {"N": 10, "F6": 6}.get(arm, 8)
    cfg["data"]["seed"] = seed
    cfg["actor_rollout_ref"]["rollout"]["seed"] = seed
    u = cfg["algorithm"]["u"]
    # Shared diagnostic cohort across controls; does not decide branch learning.
    u["reference_natural_count"] = 6
    u["site_selection"] = "random" if arm == "R" else "heuristic"
    u["seed"] = 20260927 + seed
    if arm in ("P", "N"):
        # Keep the evidence observer and natural-only Prob credit, with no actor
        # proposal calls, environment branches or auxiliary gradient updates.
        u.update(
            max_probes=0,
            max_probes_per_task=0,
            max_extra_env_calls=0,
            auxiliary_enabled=False,
        )
    cfg["trainer"]["experiment_name"] = f"irgo_v1_{arm}_seed{seed}_{profile}"
    cfg["ray_init"]["runtime_env"]["env_vars"]["IRGO_TASK_MANIFEST"] = str(
        Path(manifest).resolve()
    )
    if profile == "smoke":
        cfg["trainer"].update(
            total_training_steps=2,
            total_epochs=2,
            save_freq=2,
            test_freq=1,
            val_before_train=False,
        )
        cfg["data"]["val_batch_size"] = 8
    if (
        len(schedule.manifest["train_schedule"])
        < cfg["trainer"]["total_training_steps"]
    ):
        raise ValueError("Frozen schedule is too short")
    if len(schedule.manifest["development_schedule"]) < cfg["data"]["val_batch_size"]:
        raise ValueError("Frozen development schedule is too short")
    return cfg


def source_hashes():
    # Include all tracked and new source/config files; reject dirty-code claims
    # by preserving byte hashes in addition to the informational Git revision.
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO,
        text=True,
    ).splitlines()
    return {
        n: sha256(REPO / n)
        for n in sorted(set(names))
        if n.endswith((".py", ".yaml", ".json")) and (REPO / n).is_file()
    }


def check_smoke(path, experiment):
    if path is None:
        raise ValueError(
            "A verified smoke output is required before a full150 execution"
        )
    previous = json.loads((path / "experiment.json").read_text())
    for key in ("arm", "seed", "manifest_sha256", "source_sha256"):
        if previous[key] != experiment[key]:
            raise ValueError(f"Smoke differs from requested run: {key}")
    if previous["profile"] != "smoke" or (path / "exitcode").read_text().strip() != "0":
        raise ValueError("Smoke did not finish successfully")
    for stage in (1, 2):
        folder = path / "run/u_evidence" / f"stage_{stage:06d}"
        done = json.loads((folder / "complete.json").read_text())
        if not done.get("actor_update_completed"):
            raise ValueError("Smoke did not complete both actor updates")
        metrics = json.loads((folder / "logged_metrics.json").read_text())
        if any(
            not math.isfinite(v)
            for v in metrics.values()
            if isinstance(v, (int, float))
        ):
            raise ValueError("Smoke contains nonfinite training metrics")
        attempts = metrics["actor/optimizer_step_attempts"]
        if attempts <= 0 or attempts != metrics["actor/optimizer_steps_completed"]:
            raise ValueError("Smoke skipped an optimizer update")
        if not any(k.startswith("val/") for k in metrics):
            raise ValueError("Smoke did not validate the development reset path")
        if experiment["arm"] in ("P", "N") and any(
            metrics[k] != 0
            for k in ("u/probes", "u/query_requests", "u/extra_physical_env_calls")
        ):
            raise ValueError("Natural-only smoke performed an intervention")
    launch_record = json.loads((path / "launch.json").read_text())
    if not launch_record["same_initial_model_as_archived_run"]:
        raise ValueError("Smoke used different initial weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["P", "N", "R", "F", "F6"], required=True)
    for arg in ["model", "data-dir", "output", "manifest"]:
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=["full150", "smoke"], default="full150")
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    args = parser.parse_args()
    args.allow_different_model = False
    cfg = control_config(
        args.arm,
        args.model,
        args.data_dir,
        args.output,
        args.manifest,
        args.seed,
        args.profile,
    )
    protocol = FixedTaskSchedule(args.manifest, True, 16, args.seed)
    experiment = dict(
        arm=args.arm,
        seed=args.seed,
        profile=args.profile,
        manifest_sha256=protocol.manifest_sha256,
        source_sha256=source_hashes(),
        historical_result_applies=False,
    )
    print(
        json.dumps(
            dict(
                arm=args.arm,
                seed=args.seed,
                rounds=cfg["trainer"]["total_training_steps"],
                natural_per_task=cfg["env"]["rollout"]["n"],
                probes=cfg["algorithm"]["u"]["max_probes"],
                manifest_sha256=protocol.manifest_sha256,
                execute=args.execute,
            ),
            indent=2,
        )
    )
    if not args.execute:
        return
    if args.profile == "full150":
        check_smoke(args.smoke_output, experiment)
    os.environ["WANDB_MODE"] = args.wandb_mode
    launch(args, cfg, experiment=experiment)


if __name__ == "__main__":
    main()
