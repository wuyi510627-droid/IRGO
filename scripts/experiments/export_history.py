"""Read-only source audit: export numeric metrics and minimal task exposure."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--evidence", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.evidence
if args.output.exists():
    raise FileExistsError(args.output)
rows = []
for stage in range(1, 151):
    folder = root / f"stage_{stage:06d}"
    completion = json.loads((folder / "complete.json").read_text())
    metrics = json.loads((folder / "logged_metrics.json").read_text())
    raw = (folder / "natural.json").read_bytes()
    traces = json.loads(raw)
    tasks = {}
    for t in traces:
        task = tasks.setdefault(
            t["task_id"],
            {
                k: t[k]
                for k in [
                    "task_id",
                    "goal_id",
                    "task_text",
                    "goal_index",
                    "environment_seed",
                ]
            },
        )
        task["trajectories"] = task.get("trajectories", 0) + 1
        task["successes"] = task.get("successes", 0) + int(t["success"])
    rows.append(
        dict(
            stage=stage,
            complete=completion.get("actor_update_completed"),
            metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            tasks=list(tasks.values()),
            natural_sha256=hashlib.sha256(raw).hexdigest(),
        )
    )
    if stage % 25 == 0:
        print("audited", stage, flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
with gzip.open(args.output, "wt") as stream:
    json.dump(
        dict(schema="irgo-historical-audit-v1", source=str(root), rounds=rows), stream
    )
print(args.output)
print(
    "rounds",
    len(rows),
    "complete",
    sum(bool(r["complete"]) for r in rows),
    "groups",
    sum(len(r["tasks"]) for r in rows),
)
