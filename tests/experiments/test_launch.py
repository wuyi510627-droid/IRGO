import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/experiments"))
from train_control import control_config
from recipe.GraphGPO.task_protocol import build_manifest
from test_task_protocol import catalog


@pytest.mark.parametrize(
    "arm,n,probes,aux,site",
    [
        ("P", 8, 0, False, "heuristic"),
        ("N", 10, 0, False, "heuristic"),
        ("R", 8, 32, True, "random"),
        ("F", 8, 32, True, "heuristic"),
    ],
)
def test_controls_share_protocol_and_only_intended_changes(
    tmp_path, arm, n, probes, aux, site
):
    source = catalog()
    source["goals"] = [
        dict(source["goals"][0], index=i, asin=f"X{i}", instruction_text=f"buy {i}")
        for i in range(2000)
    ]
    manifest = build_manifest(source, train_seed=7, dev_count=128, test_count=8)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    cfg = control_config(
        arm, tmp_path / "model", tmp_path / "data", tmp_path / "run", path, 7
    )
    u = cfg["algorithm"]["u"]
    assert (
        cfg["env"]["rollout"]["n"],
        u["max_probes"],
        u["auxiliary_enabled"],
        u["site_selection"],
    ) == (n, probes, aux, site)
    assert cfg["env"]["seed"] == cfg["actor_rollout_ref"]["rollout"]["seed"] == 7
    assert cfg["trainer"]["total_training_steps"] == 150
    assert cfg["data"]["train_batch_size"] == 16
    assert cfg["ray_init"]["runtime_env"]["env_vars"]["IRGO_TASK_MANIFEST"] == str(path)
    assert cfg["trainer"]["resume_mode"] == "disable"
    if arm in ["P", "N"]:
        assert u["max_extra_env_calls"] == u["max_probes_per_task"] == 0
    if arm != "P":
        assert 16 * n * 15 + u["max_extra_env_calls"] == 2400


def test_smoke_keeps_full_batch_but_never_runs_150_rounds(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(build_manifest(catalog(), dev_count=8, test_count=8)))
    cfg = control_config(
        "N", tmp_path / "m", tmp_path / "d", tmp_path / "o", path, 0, "smoke"
    )
    assert cfg["env"]["rollout"]["n"] == 10 and cfg["data"]["train_batch_size"] == 16
    assert cfg["trainer"]["total_training_steps"] == 2
    assert cfg["trainer"]["test_freq"] == 1 and cfg["data"]["val_batch_size"] == 8


def test_full_run_rejects_failed_or_different_smoke(tmp_path):
    from train_control import check_smoke

    expected = dict(
        arm="N", seed=0, manifest_sha256="hash", source_sha256={"source": "hash"}
    )
    (tmp_path / "experiment.json").write_text(
        json.dumps(dict(expected, profile="smoke"))
    )
    (tmp_path / "exitcode").write_text("0")
    (tmp_path / "launch.json").write_text(
        json.dumps(dict(same_initial_model_as_archived_run=True))
    )
    for stage in (1, 2):
        p = tmp_path / "run/u_evidence" / f"stage_{stage:06d}"
        p.mkdir(parents=True)
        (p / "complete.json").write_text(json.dumps(dict(actor_update_completed=True)))
        (p / "logged_metrics.json").write_text(
            json.dumps(
                {
                    "actor/optimizer_step_attempts": 2,
                    "actor/optimizer_steps_completed": 2,
                    "val/success_rate": 0,
                    "u/probes": 0,
                    "u/query_requests": 0,
                    "u/extra_physical_env_calls": 0,
                }
            )
        )
    check_smoke(tmp_path, expected)
    with pytest.raises(ValueError, match="differs"):
        check_smoke(tmp_path, dict(expected, arm="R"))
    metrics = p / "logged_metrics.json"
    d = json.loads(metrics.read_text())
    d["actor/optimizer_steps_completed"] = 1
    metrics.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="skipped"):
        check_smoke(tmp_path, expected)
