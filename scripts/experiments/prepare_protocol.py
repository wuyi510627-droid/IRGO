"""Freeze task partitions from a verified WebShop catalog, without sampling models."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/irgo"))
from common import PRODUCT_HASHES
from recipe.GraphGPO.task_protocol import build_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    catalog = json.loads(args.catalog.read_text())
    if catalog["products"] != 1000 or catalog["data_sha256"] != PRODUCT_HASHES:
        raise ValueError("Expected the verified historical 1000-product catalog")
    if args.output.exists():
        raise FileExistsError("Choose a new protocol directory")
    manifests = [build_manifest(catalog, train_seed=seed) for seed in args.seeds]
    args.output.mkdir(parents=True)
    summary = []
    for manifest in manifests:
        name = f"seed{manifest['training_seed']}.json"
        (args.output / name).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)
        )
        summary.append(
            dict(
                file=name,
                manifest_sha256=manifest["manifest_sha256"],
                partitions={k: len(v) for k, v in manifest["partitions"].items()},
                heldout_evaluated=False,
            )
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
