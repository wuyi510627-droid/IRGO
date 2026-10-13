import copy, importlib, sys, types, unittest, json, collections, tempfile
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT
pkg = types.ModuleType("full_test")
pkg.__path__ = [str(SRC / "recipe/GraphGPO")]
sys.modules["full_test"] = pkg
u = importlib.import_module("full_test.prob_u")
branch = importlib.import_module("full_test.full_intervention")
aux = importlib.import_module("full_test.success_branch_loss")
hook = importlib.import_module("full_test.prob_trainer_hook")


def trace(ident, action="a", success=False):
    s = dict(
        step=1,
        key="exact-" + ident,
        anchor="page",
        remaining=15,
        action=action,
        next_anchor="end",
        valid=True,
        forced=False,
        won=success,
        done=True,
        prompt_ids=[0],
        response_ids=[1],
        checked_before={},
        checked_after={},
        clickables_before=["a", "b", "c"],
    )
    return dict(
        id=ident,
        task_id="task",
        goal_id="goal",
        goal_index=500,
        source="natural",
        split="train",
        policy_version="p",
        stage=1,
        max_steps=15,
        success=success,
        score=float(success),
        stop_reason="done",
        credit_start=0,
        steps=[s],
    )


class Contracts(unittest.TestCase):
    def test_page_credit_agrees_with_pure_prob_without_branches(self):
        ts = [trace("a1", "a"), trace("a2", "a", True), trace("b", "b", True)]
        vals, _, _ = u.estimate(ts, [], 8, "page")
        nb = dict(
            uid=["task"] * 3,
            anchor_obs=["page"] * 3,
            text_actions=["a", "a", "b"],
            next_obs=["failure:a1", "success", "success"],
        )
        expected = hook.compute_prob_step_rewards(nb, horizon=8)
        self.assertTrue(torch.allclose(torch.tensor([vals[t["id"], 1]["a"] for t in ts]), expected))

    def test_remaining_and_full_prompt_do_not_split_credit_nodes(self):
        a, b = trace("a1", "a"), trace("a2", "a", True)
        b["steps"][0].update(remaining=4, prompt_ids=[99, 98])
        values, _, _ = u.estimate([a, b], [], 8, "page")
        self.assertEqual(values["a1", 1], values["a2", 1])
        self.assertNotEqual(
            u.context_key("g", [0], "page", 15), u.context_key("g", [99, 98], "page", 4)
        )

    def test_forced_success_changes_q_without_inventing_policy_frequency(self):
        a, b = trace("a1"), trace("a2")
        t = trace("probe", "b", True)
        t["source"] = "probe"
        t["steps"][0]["forced"] = True
        values, q, pi = u.estimate([a, b], [t], 8, "page")
        state = u.graph_state("task", "page")
        self.assertEqual(q[state, "b"], 1.0)
        self.assertEqual(pi[state], {"a": 1.0})
        self.assertEqual(values["a1", 1]["a"], 0.0)

    def test_low_coverage_new_evidence_is_retained(self):
        a, b = trace("a1", "a"), trace("b", "b", True)
        p = trace("probe", "a", True)
        p["source"] = "probe"
        p["steps"][0]["forced"] = True
        old, _, _ = u.estimate([a, b], [], 8, "page")
        new, q, pi = u.estimate([a, b], [p], 8, "page")
        gates, _ = u.gates([a, b], [a, b, p], q, pi, u.UConfig(), 1)
        self.assertEqual(gates[0]["reason"], "low_coverage")
        self.assertNotEqual(old["a1", 1]["q"], new["a1", 1]["q"])

    def test_candidate_pool_never_repeats_observed_exact_context_action(self):
        # Small synthetic fixtures keep private experiment trajectories out of git.
        ts = [trace("a1", "click[a]"), trace("a2", "click[b]"), trace("a3", "click[a]", True)]
        for t in ts:
            t["steps"][0]["key"] = "shared-exact-context"
        _, q, pi = u.estimate(ts, [], 8, "page")
        gates, _ = u.gates(ts, ts, q, pi, u.UConfig(), 1)
        plan, _ = branch.make_candidates(ts, gates, {}, 1, 42)
        known = collections.defaultdict(set)
        for t in ts:
            for s in t["steps"]:
                known[s["key"]].add(s["action"])
        self.assertTrue(plan)
        for candidates in plan.values():
            self.assertLessEqual(len(candidates), 4)
            self.assertEqual(
                len(candidates), len({(x["step"]["key"], x["action"]) for x in candidates})
            )
            for c in candidates:
                self.assertNotIn(c["action"], known[c["step"]["key"]])

    def test_global_budget_and_success_stop(self):
        t = trace("n")
        t["steps"][0].update(action="click[a]", key="x")

        class Runner:
            def propose_queries(self, *args):
                return {}, {}

            def __call__(self, ref, s, action, ident):
                p = copy.deepcopy(ref)
                p.update(id=ident, source="probe", success=True)
                p["steps"][0].update(action=action, won=True, forced=True)
                return p, 1

        cfg = u.UConfig(max_probes=32, max_probes_per_task=4, max_extra_env_calls=480)
        with tempfile.TemporaryDirectory() as d:
            probes, events, calls, counts, _ = branch.execute_interventions(
                [t], Runner(), cfg, 1, [], Path(d), u.UController.write
            )
        self.assertEqual(len(probes), 1)
        self.assertEqual(calls, 1)
        self.assertEqual(counts["task"], 1)
        cfg = u.UConfig(max_probes=0, max_probes_per_task=4, max_extra_env_calls=0)
        with tempfile.TemporaryDirectory() as d:
            probes, events, calls, _, _ = branch.execute_interventions(
                [t], Runner(), cfg, 1, [], Path(d), u.UController.write
            )
        self.assertEqual(calls, 0)
        self.assertEqual(probes, [])

    def test_balanced_critical_coverage_even_with_only_two_optimizer_steps(self):
        ordinary = dict(
            prompt_ids=[0],
            response_ids=[1],
            intervention=False,
            forced=False,
            replayed_prefix=False,
        )
        repair = {
            str(i): (dict(ordinary, intervention=True, forced=True), [ordinary]) for i in range(16)
        }
        plan, counts = aux.schedule_auxiliary({"normal": [ordinary]}, repair, 2, 1)
        self.assertEqual(len(counts), 16)
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual([len(x) for x in plan], [64, 64])
        self.assertEqual(sum(r["intervention"] for rs in plan for r in rs), 16)

    def test_zero_mass_missing_normal_and_terminal_suffix(self):
        critical = dict(
            prompt_ids=[0],
            response_ids=[1],
            intervention=True,
            forced=True,
            replayed_prefix=False,
            terminal_success=True,
        )
        plan, counts = aux.schedule_auxiliary({}, {"t": (critical, [])}, 1, 1)
        self.assertEqual(sum(r is not None for r in plan[0]), 1)
        self.assertEqual(counts, {"t": 1})

    def test_auxiliary_learning_exists_when_ppo_advantage_is_zero(self):
        class Actor:
            ulysses_sequence_parallel_size = 1

            def __init__(self):
                self.actor_module = torch.nn.Linear(1, 2, bias=False)
                torch.nn.init.zeros_(self.actor_module.weight)

            def _forward_micro_batch(self, data, **kwargs):
                logits = self.actor_module(torch.ones(1, 1))
                return None, logits.log_softmax(-1)[:, 1:2]

        row = dict(prompt_ids=[0], response_ids=[1], intervention=True)
        actor = Actor()
        opt = torch.optim.SGD(actor.actor_module.parameters(), lr=0.1)
        before = float(actor.actor_module.weight[1].detach())
        aux.backward_auxiliary(actor, dict(by_update=[[row] * 8], coefficient=0.1), 0)
        opt.step()
        self.assertGreater(float(actor.actor_module.weight[1].detach()), before)
        actor = Actor()
        aux.backward_auxiliary(actor, dict(by_update=[[row] * 8], coefficient=0.0), 0)
        self.assertEqual(float(actor.actor_module.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
