import pytest
from collections import Counter
from dataclasses import replace

from recipe.GraphGPO.prob_u import UConfig
from recipe.GraphGPO.full_intervention import execute_interventions, make_candidates


def evidence():
    traces = []
    for task in ("a", "b"):
        steps = []
        for i in range(5):
            steps.append(
                dict(
                    key=f"{task}-{i}",
                    anchor=f"page-{i}",
                    next_anchor="next",
                    step=i + 1,
                    action="search[original]",
                    valid=True,
                    checked_before={},
                    checked_after={"visible_static_feasible": 0 if i == 0 else 1},
                    clickables_before=["X", "Y"],
                )
            )
        traces.append(
            dict(id=task, task_id=task, steps=steps, max_steps=15, success=False)
        )
    return traces


def test_zero_budget_never_generates_queries_or_executes_branches(tmp_path):
    class ForbiddenRunner:
        def propose_queries(self, *args):
            raise AssertionError("A pure Prob control cannot generate search proposals")

        def __call__(self, *args):
            raise AssertionError("A pure Prob control cannot intervene")

    for cfg in [
        replace(UConfig(), max_probes=0),
        replace(UConfig(), max_extra_env_calls=0),
    ]:
        result = execute_interventions(
            evidence(), ForbiddenRunner(), cfg, 1, [], tmp_path, lambda path, obj: None
        )
        assert result[:4] == ([], [], 0, Counter())
        assert result[4]["u/query_requests"] == 0


def test_random_changes_sites_but_keeps_action_order_and_task_priority():
    traces = evidence()
    heuristic, order = make_candidates(traces, [], {}, 1, 42)
    random, random_order = make_candidates(
        traces, [], {}, 1, 42, site_selection="random"
    )
    assert random_order == order
    assert random == make_candidates(traces, [], {}, 1, 42, site_selection="random")[0]
    assert {c["step"]["key"] for c in random["a"]} != {
        c["step"]["key"] for c in heuristic["a"]
    }
    for rows in random.values():
        assert len(rows) <= 4
        assert len({c["step"]["key"] for c in rows}) <= 2
        assert all(c["action"] in {"click[x]", "click[y]"} for c in rows)
        # A site keeps exactly its existing action ordering: the intervention
        # baseline randomizes the context, not candidate quality or actions.
        for c in rows:
            singleton = [dict(c["parent"], steps=[c["step"]])]
            original, _ = make_candidates(singleton, [], {}, 1, 42)
            assert c["actions"] == original[c["task"]][0]["actions"]


@pytest.mark.parametrize("reference_count", [6, 8])
def test_reference_failure_is_not_changed_by_success_after_reference(
    tmp_path, reference_count
):
    from recipe.GraphGPO.prob_u import UController, failed_trajectory_ids

    traces = []
    for i in range(10):
        success = i == reference_count
        traces.append(
            dict(
                id=str(i),
                task_id="one",
                goal_id="g",
                source="natural",
                stage=1,
                policy_version="p",
                split="train",
                max_steps=15,
                success=success,
                score=int(success),
                stop_reason="terminal",
                credit_start=0,
                steps=[
                    dict(
                        step=1,
                        key="context",
                        anchor="page",
                        action="click[x]",
                        next_anchor="end",
                        valid=True,
                        won=success,
                        done=True,
                        prompt_ids=[1],
                        remaining=15,
                        forced=False,
                    )
                ],
            )
        )
    cfg = replace(
        UConfig(reference_natural_count=reference_count),
        max_probes=0,
        max_extra_env_calls=0,
        auxiliary_enabled=False,
    )

    def forbidden(*args):
        raise AssertionError("No branches in natural-only control")

    before, after, metrics = UController(cfg, tmp_path).process(traces, forbidden, 1)
    assert failed_trajectory_ids(traces, limit=reference_count) == {
        str(i) for i in range(reference_count)
    }
    assert failed_trajectory_ids(traces) == set()
    assert before == after
    assert metrics[f"tasks/reference_first{reference_count}_eligible"] == 1
    assert metrics[f"tasks/reference_first{reference_count}_all_failed"] == 1
    assert metrics["tasks/initial_all_failed_count"] == 1
    assert metrics["tasks/initial_all_failed_ratio"] == 1
    assert metrics["tasks/all_natural_all_failed_count"] == 0
    assert metrics["u/extra_input_tokens"] == metrics["u/extra_generated_tokens"] == 0
