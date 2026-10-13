import copy
import pytest

from recipe.GraphGPO.task_protocol import build_manifest, FixedTaskSchedule, fingerprint


def catalog():
    goals = [
        dict(
            index=i,
            asin=f"P{i}",
            attributes=[f"attribute {i}"],
            goal_options={},
            instruction_text=f"Buy item {i}",
            price_upper=10,
        )
        for i in range(400)
    ]
    # Both a repeated structural task with another price and a repeated input
    # referring to another ASIN must stay together when partitions are built.
    goals += [
        dict(
            goals[0], index=400, price_upper=20, instruction_text="Buy item 0 under $20"
        ),
        dict(goals[1], index=401, asin="another-product"),
    ]
    return dict(
        products=1000,
        environment_seed=0,
        data_sha256={"items": "fixture"},
        source_sha256={},
        goals=goals,
    )


def test_real_identity_and_identical_inputs_cannot_cross_splits():
    manifest = build_manifest(
        catalog(), train_seed=7, rounds=3, tasks_per_round=4, dev_count=8, test_count=8
    )
    partitions = manifest["partitions"]
    owner = {i: name for name, ids in partitions.items() for i in ids}
    assert owner[0] == owner[400]
    assert owner[1] == owner[401]
    assert len(owner) == 402
    assert sum(map(len, partitions.values())) == 402
    assert manifest == build_manifest(
        catalog(), train_seed=7, rounds=3, tasks_per_round=4, dev_count=8, test_count=8
    )


def test_manifest_drives_identical_tasks_independent_of_group_size():
    manifest = build_manifest(
        catalog(), train_seed=7, rounds=3, tasks_per_round=4, dev_count=8, test_count=8
    )
    a = FixedTaskSchedule(manifest, is_train=True, env_num=4, train_seed=7)
    b = FixedTaskSchedule(manifest, is_train=True, env_num=4, train_seed=7)
    for _ in range(3):
        assert a.next_indices() == b.next_indices()
    with pytest.raises(ValueError, match="exhausted"):
        a.next_indices()
    dev = FixedTaskSchedule(manifest, is_train=False, env_num=8, train_seed=1007)
    assert dev.next_indices() == dev.next_indices()


def test_wrong_catalog_seed_or_modified_manifest_is_rejected():
    source = catalog()
    manifest = build_manifest(
        source, rounds=2, tasks_per_round=4, dev_count=8, test_count=8
    )
    schedule = FixedTaskSchedule(manifest, is_train=True, env_num=4, train_seed=0)
    goals = [{k: v for k, v in g.items() if k != "index"} for g in source["goals"]]
    schedule.verify_goals(goals)
    goals[0]["instruction_text"] = "different goal"
    with pytest.raises(ValueError, match="catalog"):
        schedule.verify_goals(goals)
    changed = copy.deepcopy(manifest)
    changed["train_schedule"][0][0] = manifest["partitions"]["test"][0]
    with pytest.raises(ValueError, match="hash"):
        FixedTaskSchedule(changed, is_train=True, env_num=4, train_seed=0)
    changed["manifest_sha256"] = fingerprint(
        {k: v for k, v in changed.items() if k != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="partition"):
        FixedTaskSchedule(changed, is_train=True, env_num=4, train_seed=0)
