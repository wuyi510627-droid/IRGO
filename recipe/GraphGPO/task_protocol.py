"""Frozen task identities and schedules for the new experimental protocol.

Only the dataset split changes here. Credit still merges task + page, while
branch restoration still checks the complete context. No model outcomes are
read while making the split or deciding the training order.
"""

import hashlib
import json
from pathlib import Path
import random

SOURCE_FILES = (
    "web_agent_site/engine/goal.py",
    "web_agent_site/engine/engine.py",
    "web_agent_site/envs/web_agent_text_env.py",
)


def environment_source_hashes(webshop_root):
    return {
        name: hashlib.sha256((Path(webshop_root) / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def structural_key(goal):
    return fingerprint([goal["asin"], sorted(goal["attributes"]), goal["goal_options"]])


def instruction_key(goal):
    return fingerprint(" ".join(goal["instruction_text"].casefold().split()))


def _goals(rows):
    if [g["index"] for g in rows] != list(range(len(rows))):
        raise ValueError("Goal indices must match the original ordered catalog")
    return [{k: v for k, v in g.items() if k != "index"} for g in rows]


def build_manifest(
    catalog,
    train_seed=0,
    split_seed=20261013,
    rounds=150,
    tasks_per_round=16,
    dev_count=512,
    test_count=512,
):
    if min(rounds, tasks_per_round, dev_count, test_count) < 1:
        raise ValueError("Empty experiment schedule")
    goals = _goals(catalog["goals"])
    if set(catalog["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("Incomplete environment source hashes")
    parents = list(range(len(goals)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    # Join on either identity: a structural goal can have differently worded
    # prices, and different ASINs can have exactly the same model instruction.
    seen = {}
    records = []
    for i, goal in enumerate(goals):
        s, text = structural_key(goal), instruction_key(goal)
        records.append(dict(index=i, structural_key=s, instruction_key=text))
        for key in [("structure", s), ("instruction", text)]:
            if key in seen:
                parents[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i in range(len(goals)):
        groups.setdefault(root(i), []).append(i)
    partitions = {name: [] for name in ("train", "development", "test")}
    for group in groups.values():
        identity = sorted(records[i]["structural_key"] for i in group)
        bucket = int(fingerprint([split_seed, identity])[:16], 16) % 100
        name = "train" if bucket < 80 else "development" if bucket < 90 else "test"
        partitions[name].extend(group)
    for values in partitions.values():
        values.sort()
    if len(partitions["train"]) < tasks_per_round:
        raise ValueError("Not enough training goals")
    if (
        len(partitions["development"]) < dev_count
        or len(partitions["test"]) < test_count
    ):
        raise ValueError(
            "Fixed split lacks enough evaluation goals; do not silently recycle tasks"
        )
    rng = random.Random(train_seed)
    train_schedule = [
        rng.sample(partitions["train"], tasks_per_round) for _ in range(rounds)
    ]
    manifest = dict(
        schema="irgo-task-protocol-v1",
        environment_seed=catalog["environment_seed"],
        products=catalog["products"],
        data_sha256=catalog["data_sha256"],
        source_sha256=catalog["source_sha256"],
        catalog_sha256=fingerprint(goals),
        training_seed=train_seed,
        split_seed=split_seed,
        records=records,
        partitions=partitions,
        tasks_per_round=tasks_per_round,
        train_schedule=train_schedule,
        development_schedule=random.Random(split_seed + 1).sample(
            partitions["development"], dev_count
        ),
        heldout_schedule=random.Random(split_seed + 2).sample(
            partitions["test"], test_count
        ),
        heldout_evaluation_enabled=False,
    )
    manifest["manifest_sha256"] = fingerprint(manifest)
    return manifest


class FixedTaskSchedule:
    """Validated train/development schedule; never exposes held-out test resets."""

    def __init__(self, manifest, is_train, env_num, train_seed):
        if isinstance(manifest, (str, Path)):
            manifest = json.loads(Path(manifest).read_text())
        if manifest.get("schema") != "irgo-task-protocol-v1":
            raise ValueError("Unsupported task manifest schema")
        payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if fingerprint(payload) != manifest.get("manifest_sha256"):
            raise ValueError("Task manifest hash mismatch")
        self.manifest = manifest
        if set(manifest["source_sha256"]) != set(SOURCE_FILES):
            raise ValueError("Incomplete environment source hashes")
        self.is_train, self.env_num, self.cursor = is_train, env_num, 0
        self.environment_seed = manifest["environment_seed"]
        self.manifest_sha256 = manifest["manifest_sha256"]
        self.allowed_indices = set(
            manifest["partitions"]["train" if is_train else "development"]
        )
        records = manifest["records"]
        indices = set(range(len(records)))
        union = set()
        identities = {key: set() for key in ("structural_key", "instruction_key")}
        for ids in manifest["partitions"].values():
            if (
                len(set(ids)) != len(ids)
                or union.intersection(ids)
                or not set(ids) <= indices
            ):
                raise ValueError("Overlapping or invalid task partition")
            union.update(ids)
            for key, earlier in identities.items():
                current = {records[i][key] for i in ids}
                if current & earlier:
                    raise ValueError("Real task identity crosses a partition")
                earlier.update(current)
        if union != indices:
            raise ValueError("Task partition is incomplete")
        for name, rows in [
            ("train", manifest["train_schedule"]),
            ("development", [manifest["development_schedule"]]),
            ("test", [manifest["heldout_schedule"]]),
        ]:
            for row in rows:
                if len(row) != len(set(row)) or not set(row) <= set(
                    manifest["partitions"][name]
                ):
                    raise ValueError("Schedule violates its task partition")
        if is_train:
            if (
                train_seed != manifest["training_seed"]
                or env_num != manifest["tasks_per_round"]
            ):
                raise ValueError("Training seed/batch differs from frozen schedule")
            if any(len(row) != env_num for row in manifest["train_schedule"]):
                raise ValueError("Training row width differs from environment batch")
        elif env_num > len(manifest["development_schedule"]):
            raise ValueError("Development schedule is too short")

    def verify_sources(self, webshop_root):
        if environment_source_hashes(webshop_root) != self.manifest["source_sha256"]:
            raise ValueError("Runtime environment source differs from frozen manifest")

    def verify_goals(self, goals):
        if fingerprint(goals) != self.manifest["catalog_sha256"]:
            raise ValueError("Runtime goal catalog differs from frozen manifest")
        expected = [
            dict(
                index=i,
                structural_key=structural_key(goal),
                instruction_key=instruction_key(goal),
            )
            for i, goal in enumerate(goals)
        ]
        if expected != self.manifest["records"]:
            raise ValueError("Task identity records differ from actual goal catalog")

    def next_indices(self):
        if not self.is_train:
            return list(self.manifest["development_schedule"][: self.env_num])
        if self.cursor >= len(self.manifest["train_schedule"]):
            raise ValueError("Frozen training schedule exhausted")
        row = list(self.manifest["train_schedule"][self.cursor])
        self.cursor += 1
        return row

    def identities(self, indices, group_n):
        role = "train" if self.is_train else "development"
        tasks, trajectories = [], []
        for slot, index in enumerate(indices):
            task = fingerprint([self.manifest_sha256, role, self.cursor, slot, index])
            for replica in range(group_n):
                tasks.append(task)
                trajectories.append(fingerprint([task, replica]))
        return tasks, trajectories
