"""Release-level checks: archived algorithm identity, launch semantics and results."""

import ast
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/irgo"))
from common import training_config
from evaluate import evaluation_config, parse_result
from prepare_data import metadata_rows


def canonical_ast(node):
    if isinstance(node, ast.AST):
        result = {"node": type(node).__name__}
        for key, value in ast.iter_fields(node):
            if key == "type_params" and not value:
                continue
            if key == "body" and isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                if (
                    value
                    and isinstance(value[0], ast.Expr)
                    and isinstance(value[0].value, ast.Constant)
                    and isinstance(value[0].value.value, str)
                ):
                    value = value[1:]
            result[key] = canonical_ast(value)
        return result
    if isinstance(node, list):
        return [canonical_ast(value) for value in node]
    if isinstance(node, bytes):
        return {"bytes": node.hex()}
    if node is Ellipsis:
        return {"constant": "Ellipsis"}
    return node


def test_versioned_algorithm_ast_changes_are_explicit():
    reference = json.loads((ROOT / "reproducibility/algorithm_semantics.json").read_text())
    changes = json.loads((ROOT / "reproducibility/experimental_changes.json").read_text())
    assert changes['historical_result_applies'] is False
    assert set(changes['files']) <= set(reference['sha256'])
    for name, expected in reference["sha256"].items():
        if name in changes['files']:
            change = changes['files'][name]
            assert change['baseline_ast_sha256'] == expected
            assert change['reason'] and change['new_ast_sha256'] != expected
            expected = change['new_ast_sha256']
        tree = canonical_ast(ast.parse((ROOT / name).read_text()))
        value = json.dumps(tree, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        assert hashlib.sha256(value).hexdigest() == expected, name


def test_full_training_config_retains_scientific_settings(tmp_path):
    # Relocation handles ordinary shell/YAML special characters as path data.
    model = tmp_path / "model #1: a'b"
    cfg = training_config(model, tmp_path / "data", tmp_path / "out")
    assert cfg["actor_rollout_ref"]["model"]["path"] == str(model)
    assert cfg["data"]["train_batch_size"] == 16
    assert cfg["env"]["rollout"]["n"] == 8
    assert cfg["env"]["max_steps"] == 15
    assert cfg["trainer"]["total_training_steps"] == 150
    assert cfg["trainer"]["resume_mode"] == "disable"
    assert cfg["actor_rollout_ref"]["model"]["lora_rank"] == 0
    assert cfg["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] == 1
    u = cfg["algorithm"]["u"]
    assert (u["max_probes"], u["max_probes_per_task"], u["max_extra_env_calls"]) == (32, 4, 480)
    assert u["credit_state_mode"] == "page" and u["auxiliary_coefficient"] == 0.1
    # A valid Linux checkout can live under /home/runner. Check the relocated
    # fields themselves instead of rejecting a platform's entire home prefix.
    assert cfg["data"]["train_files"] == str(tmp_path / "data/train.parquet")
    assert cfg["data"]["val_files"] == str(tmp_path / "data/test.parquet")
    assert cfg["trainer"]["default_local_dir"] == str(tmp_path / "out/run")
    assert cfg["trainer"]["validation_data_dir"] == str(tmp_path / "out/run/validation")
    assert cfg["ray_init"]["runtime_env"]["env_vars"]["PYTHONPATH"] == str(ROOT)
    assert "${oc.env:" not in json.dumps(cfg)


def test_smoke_is_separate_and_keeps_auxiliary_path(tmp_path):
    cfg = training_config(tmp_path / "model", tmp_path / "data", tmp_path / "smoke", "smoke")
    assert cfg["trainer"]["total_training_steps"] == 2
    assert cfg["data"]["train_batch_size"] == 4
    assert cfg["env"]["rollout"]["n"] == 2
    assert cfg["algorithm"]["u"]["auxiliary_enabled"]
    assert cfg["algorithm"]["u"]["max_probes"] == 4


@pytest.mark.parametrize("seed", [123, 456, 789])
def test_evaluation_never_trains_or_intervenes(tmp_path, seed):
    cfg = evaluation_config(tmp_path / "final-model", tmp_path / "eval", seed)
    assert cfg["trainer"]["val_only"] and cfg["trainer"]["val_before_train"]
    assert cfg["trainer"]["save_freq"] == -1 and cfg["trainer"]["resume_mode"] == "disable"
    assert not cfg["algorithm"]["u"]["enabled"]
    assert not cfg["algorithm"]["filter_groups"]["enable"]
    assert cfg["env"]["seed"] == seed and cfg["env"]["max_steps"] == 30
    assert cfg["env"]["webshop"]["observation_mode"] == "text_rich"
    assert cfg["data"]["val_batch_size"] == 128
    assert cfg["data"]["max_prompt_length"] == 4096
    assert cfg["actor_rollout_ref"]["rollout"]["prompt_length"] == 4096
    assert cfg["actor_rollout_ref"]["rollout"]["val_kwargs"]["n"] == 1


def test_partial_or_failed_evaluation_cannot_be_reported_as_complete():
    text = "Validation: 100% 4/4\n 'val/success_rate': 0.798828125"
    assert parse_result(text, 789, 0)["successes"] == 409
    for log, code in [(text.replace("4/4", "2/4"), 0), (text, 1), ("4/4", 0)]:
        with pytest.raises((ValueError, RuntimeError)):
            parse_result(log, 789, code)


def test_public_results_use_environment_seeds_and_exact_counts():
    data = json.loads((ROOT / "results/webshop_150.json").read_text())
    assert data["protocol"]["training_seeds_per_method"] == 1
    for model in data["models"].values():
        assert sum(r["successes"] for r in model["runs"]) == model["total_successes"]
        assert sum(r["episodes"] for r in model["runs"]) == 1536
        assert model["success_rate"] == model["total_successes"] / 1536


def test_metadata_is_not_a_source_of_task_solutions():
    rows = metadata_rows("test", 512)
    assert len(rows) == 512 and rows[511]["extra_info"]["index"] == 511
    assert all(r["prompt"] == [{"role": "user", "content": ""}] for r in rows)
    with pytest.raises(ValueError):
        metadata_rows("train", 0)
