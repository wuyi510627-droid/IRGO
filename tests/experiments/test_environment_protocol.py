"""Exercise the real vector wrapper with in-process Ray transport doubles."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from recipe.GraphGPO.task_protocol import build_manifest
from test_task_protocol import catalog


class Remote:
    def __init__(self, function):
        self.remote = function


def wrapper(monkeypatch, goals):
    class Worker:
        def __init__(self, seeds, *args):
            self.seeds = seeds if isinstance(seeds, list) else [seeds]
            self.get_goals = Remote(lambda: goals)
            self.reset = Remote(
                lambda ids: (
                    [(str(i), {}) for i in ids]
                    if isinstance(ids, list)
                    else (str(ids), {})
                )
            )
            self.close = Remote(lambda: None)

    ray = SimpleNamespace(
        is_initialized=lambda: True,
        get=lambda x: x,
        remote=lambda **kw: lambda cls: Remote(Worker),
        kill=lambda x: None,
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "gym", SimpleNamespace(Env=object))
    path = (
        Path(__file__).resolve().parents[2]
        / "agent_system/environments/env_package/webshop/envs.py"
    )
    spec = importlib.util.spec_from_file_location("test_webshop_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WebshopMultiProcessEnv


@pytest.mark.parametrize("group_n,per_actor", [(8, 1), (8, 8), (10, 10)])
def test_vector_reset_uses_frozen_tasks_and_seed(
    monkeypatch, tmp_path, group_n, per_actor
):
    source = catalog()
    goals = [{k: v for k, v in g.items() if k != "index"} for g in source["goals"]]
    manifest = build_manifest(
        source, rounds=2, tasks_per_round=4, dev_count=8, test_count=8
    )
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv("IRGO_TASK_MANIFEST", str(path))
    monkeypatch.setenv("WEBSHOP_ENVS_PER_ACTOR", str(per_actor))
    cls = wrapper(monkeypatch, goals)
    env = cls(0, 4, group_n, {}, True)
    _, infos = env.reset()
    assert [i["u_goal_index"] for i in infos[::group_n]] == manifest["train_schedule"][
        0
    ]
    assert {i["u_environment_seed"] for i in infos} == {0}
    assert all(set(w.seeds) == {0} for w in env._workers)
    assert len({i["irgo_task_instance_id"] for i in infos}) == 4
    assert len({i["irgo_trajectory_id"] for i in infos}) == 4 * group_n
    assert [i["u_goal_index"] for i in env.reset()[1][::group_n]] == manifest[
        "train_schedule"
    ][1]
    with pytest.raises(ValueError, match="training split"):
        env.branch(0, manifest["heldout_schedule"][0])
    env.close()
