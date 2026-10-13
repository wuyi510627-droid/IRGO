"""Summarize complete historical receipts; reconstruct (not observe) validation identities."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recipe.GraphGPO.task_protocol import structural_key


def audit(history, catalog):
    rows = history["rounds"]
    if [r["stage"] for r in rows] != list(range(1, 151)) or not all(
        r["complete"] for r in rows
    ):
        raise ValueError(
            "Expected all 150 completed stages; incomplete data cannot be extrapolated"
        )
    keys = [
        "environment/u_total_physical_calls",
        "environment/u_natural_physical_calls",
        "u/extra_physical_env_calls",
        "u/probes",
        "u/extra_prefix_steps",
        "u/natural_generated_tokens",
        "u/extra_generated_tokens",
        "u/query_generated_tokens",
        "u/extra_seconds",
        "timing_s/step",
        "aux/tasks",
        "tasks/initial_all_failed_count",
    ]
    totals = {k: sum(r["metrics"][k] for r in rows) for k in keys}
    if (
        totals["environment/u_total_physical_calls"]
        != totals["environment/u_natural_physical_calls"]
        + totals["u/extra_physical_env_calls"]
    ):
        raise ValueError("Physical cost ledger is inconsistent")
    goals = catalog["goals"]
    size = len(goals)

    def order(seed):
        ids = list(range(size))
        random.Random(seed).shuffle(ids)
        return ids

    canonical = [None] * size
    for goal, index in zip(goals, order(catalog["environment_seed"])):
        canonical[index] = goal
    orders = {seed: order(seed) for seed in list(range(16)) + list(range(1000, 1128))}

    def norm(text):
        return (
            re.sub(r", and price lower than [0-9.]+ dollars", "", text).strip().lower()
        )

    seen = defaultdict(set)
    train_rng = np.random.RandomState(0)
    for row in rows:
        tasks = row["tasks"]
        expected = train_rng.choice(range(500, size), 16, replace=False).tolist()
        if sorted((t["environment_seed"], t["goal_index"]) for t in tasks) != list(
            enumerate(expected)
        ):
            raise ValueError(
                "Observed training resets differ from reconstructed historical schedule"
            )
        for task in tasks:
            goal = canonical[orders[task["environment_seed"]][task["goal_index"]]]
            if norm(goal["instruction_text"]) != norm(task["task_text"]):
                raise ValueError(
                    "Reconstructed structural identity does not match observed task text"
                )
            seen[row["stage"]].add(structural_key(goal))
    # Validation task IDs were not archived. Label this a deterministic schedule
    # reconstruction, even though all 2400 training identities were cross-checked.
    rng = np.random.RandomState(1000)
    validation = []
    for stage in range(0, 151, 5):
        indices = rng.choice(range(500), 128, replace=False)
        prior = set().union(*(ks for s, ks in seen.items() if s <= stage))
        matches = sum(
            structural_key(canonical[orders[1000 + slot][int(idx)]]) in prior
            for slot, idx in enumerate(indices)
        )
        validation.append(dict(stage=stage, episodes=128, structural_overlap=matches))
    return dict(
        schema="irgo-historical-audit-v1",
        rounds=150,
        observed_task_groups=sum(len(r["tasks"]) for r in rows),
        observed_natural_trajectories=sum(
            t["trajectories"] for r in rows for t in r["tasks"]
        ),
        unique_structural_training_tasks=len(set().union(*seen.values())),
        unique_observed_instruction_texts=len(
            {t["task_text"] for r in rows for t in r["tasks"]}
        ),
        totals=totals,
        mean_per_round={k: v / 150 for k, v in totals.items()},
        generation_input_token_cost="not recorded historically; unavailable",
        extra_generated_tokens_include_queries=True,
        validation_identity_evidence="Reconstructed from archived seed/reset rules; not observed validation traces. Structural identity ignores seed-dependent price.",
        validation=validation,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["history", "catalog", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(
        json.loads(gzip.decompress(args.history.read_bytes())),
        json.loads(args.catalog.read_text()),
    )
    result["input_sha256"] = {
        name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
        for name in ["history", "catalog"]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["validation", "mean_per_round"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
