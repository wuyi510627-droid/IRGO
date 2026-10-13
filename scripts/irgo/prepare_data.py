"""Generate text-modality metadata, not WebShop tasks or historical trajectories.

The environment supplies task goals. These empty prompts reproduce the schema
of upstream examples.data_preprocess.prepare without downloading Geometry3k.
For exact historical replay, use the archived parquet files and compare hashes.
"""

import argparse
from pathlib import Path


def metadata_rows(split, count):
    if count < 1:
        raise ValueError("Row count must be positive")
    return [
        dict(
            data_source="text",
            prompt=[dict(role="user", content="")],
            ability="agent",
            extra_info=dict(split=split, index=i),
        )
        for i in range(count)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=16)
    parser.add_argument("--val-rows", type=int, default=128)
    args = parser.parse_args()
    from datasets import Dataset

    rows = [
        ("train", metadata_rows("train", args.train_rows)),
        ("test", metadata_rows("test", args.val_rows)),
    ]
    if any((args.output / f"{split}.parquet").exists() for split, _ in rows):
        raise FileExistsError("Refusing to overwrite existing metadata")
    args.output.mkdir(parents=True, exist_ok=True)
    for split, data in rows:
        Dataset.from_list(data).to_parquet(str(args.output / f"{split}.parquet"))
    print("Prepared metadata:", args.output)


if __name__ == "__main__":
    main()
