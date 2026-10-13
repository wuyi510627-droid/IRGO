"""U controller: current-policy graph credit, bounded probes, history shadow audit.

No historical or forced response enters the PPO batch. Verified current-round
branches can separately supply supervised labels for the same policy model.
The sign statistic is a conditional MC proxy, not a calibrated correctness claim.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .probabilistic_credit import (
    build_posterior,
    policy_qva_values,
    canonical_state,
    scoped_state,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def context_key(goal, prompt_ids, anchor, remaining):
    """Exact decision context for replay validation, independent of graph merging."""
    return digest([goal, list(map(int, prompt_ids)), anchor, int(remaining)])


# Preserve the name used by historical diagnostics and saved trace tooling.
state_key = context_key


def credit_key(step, mode):
    """Raw graph/normalization identity; task scoping is applied separately."""
    if mode == "page":
        return step["anchor"]
    if mode == "context":
        return step["key"]
    raise ValueError(f"Unsupported credit_state_mode: {mode}")


def graph_state(task, key):
    return scoped_state(task, canonical_state(key))


@dataclass(frozen=True)
class UConfig:
    """IRGO 已完成实验的默认设置；历史缓冲区仅用于影子审计。

    posterior_draws 是后验参数抽样次数，不是环境轨迹数。
    max_probes 是全轮上限，max_probes_per_task 是单任务上限。
    """

    policy_evidence: str = "natural_and_unforced_continuations_v2"
    credit_state_mode: str = "page"
    horizon: int = 8
    min_count: int = 2
    sign_threshold: float = 0.95
    practical_epsilon: float = 0.02
    posterior_draws: int = 512
    max_probes: int = 32
    max_probes_per_task: int = 4
    max_extra_env_calls: int = 480
    site_selection: str = "heuristic"
    # Observation cohort only; intervention/auxiliary eligibility uses all natural evidence.
    reference_natural_count: int = 8
    auxiliary_enabled: bool = True
    auxiliary_coefficient: float = 0.1
    seed: int = 20260927
    history_rounds: int = 10
    history_max_trajectories: int = 1280

    def __post_init__(self):
        if self.reference_natural_count < 1:
            raise ValueError("Invalid reference trajectory count")
        if self.site_selection not in ("heuristic", "random"):
            raise ValueError("Unsupported site selection control")
        if not 0 <= self.auxiliary_coefficient <= 1:
            raise ValueError("Invalid auxiliary coefficient")
        if self.credit_state_mode not in ("page", "context"):
            raise ValueError("Unsupported credit_state_mode")
        if self.policy_evidence != "natural_and_unforced_continuations_v2":
            raise ValueError("Unsupported U policy evidence semantics")
        if self.horizon < 1 or self.min_count < 1 or self.posterior_draws < 32:
            raise ValueError("Invalid estimation parameters")
        if not 0.5 < self.sign_threshold < 1 or self.practical_epsilon < 0:
            raise ValueError("Invalid confidence parameters")
        if (
            min(
                self.max_probes,
                self.max_probes_per_task,
                self.max_extra_env_calls,
                self.history_rounds,
                self.history_max_trajectories,
            )
            < 0
        ):
            raise ValueError("Negative budget")


def outcomes(traces, horizon, credit_state_mode="context"):
    """Suffix outcomes; history diagnostics retain exact-context matching by default."""
    rows = []
    for t in traces:
        if t["split"] != "train":
            raise ValueError("Non-training evidence rejected")
        steps = t["steps"]
        for i in range(t.get("credit_start", 0), len(steps)):
            s = steps[i]
            rows.append(
                dict(
                    trajectory_id=t["id"],
                    task_id=t["task_id"],
                    goal_id=t["goal_id"],
                    key=credit_key(s, credit_state_mode),
                    context_key=s["key"],
                    action=s["action"],
                    valid=s["valid"],
                    y=int(any(x["won"] for x in steps[i : i + horizon])),
                    stage=t["stage"],
                    policy_version=t["policy_version"],
                )
            )
    return rows


def estimate(natural, probes, horizon, credit_state_mode="context"):
    """Finite-horizon policy DP with configurable credit-node identity.

    The controller passes its explicit mode; the default preserves historical
    offline callers. Page mode pools contexts within each task, as pure Prob does.
    Natural actions and current-policy probe continuations estimate pi at each
    credit node. Replayed prefixes and forced decisions never count as policy
    samples. A sampled continuation remains eligible at an already observed state;
    discarding it would assign unseen successful actions zero probability.
    """
    if not natural:
        raise ValueError("Empty natural batch")
    version = natural[0]["policy_version"]
    ids = [t["id"] for t in natural + probes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evidence trajectory")
    nb = {k: [] for k in ("uid", "anchor_obs", "next_obs", "text_actions")}
    base_counts, continuation_counts = defaultdict(Counter), defaultdict(Counter)
    for kind, traces in [("natural", natural), ("probe", probes)]:
        for t in traces:
            if t["policy_version"] != version or t["split"] != "train":
                raise ValueError("Stale-policy or non-training evidence in live credit")
            if t["source"] != kind:
                raise ValueError("Evidence source mismatch")
            steps = t["steps"]
            for i in range(t.get("credit_start", 0), len(steps)):
                s = steps[i]
                key = credit_key(s, credit_state_mode)
                successor = (
                    "success"
                    if s["won"]
                    else (
                        credit_key(steps[i + 1], credit_state_mode)
                        if i + 1 < len(steps) and not s["done"]
                        else "failure:" + t["id"]
                    )
                )
                for k, val in [
                    ("uid", t["task_id"]),
                    ("anchor_obs", key),
                    ("next_obs", successor),
                    ("text_actions", s["action"]),
                ]:
                    nb[k].append(val)
                if not s.get("forced", False):
                    counts = base_counts if kind == "natural" else continuation_counts
                    counts[graph_state(t["task_id"], key)][s["action"]] += 1
    policy = {}
    for state in base_counts.keys() | continuation_counts.keys():
        c = base_counts[state] + continuation_counts[state]
        policy[state] = {a: n / sum(c.values()) for a, n in c.items()}
    # 强制转移可更新 P(s'|s,a)；上面的经验策略计数已排除强制动作。
    post = build_posterior(nb, scope_by_uid=True)
    # All forced states must already have natural-policy support; never rely on
    # the legacy uniform fallback to invent a baseline at a forced-only state.
    if any(s not in policy for s, _ in post.counts()):
        raise ValueError("Missing natural policy support at an evidence state")
    q, v, a = policy_qva_values(post, policy=policy, horizon=horizon)
    values = {}
    for t in natural:
        for s in t["steps"]:
            g = graph_state(t["task_id"], credit_key(s, credit_state_mode))
            values[(t["id"], s["step"])] = dict(
                q=q[(g, s["action"])], v=v[g], a=a[(g, s["action"])]
            )
    return values, q, policy


def gates(natural, evidence, graph_q, policy, cfg, stage):
    """用有限视野终局标签计算条件优势符号稳定性代理。

    Beta 抽样基于动作出现次数；同轨迹重复访问可能相关。因此 .95
    不代表动作有 95% 概率正确。此函数提供选点和停止信号，不决定
    新证据能否更新最终优势，也不生成新的环境样本。
    """
    groups = defaultdict(lambda: defaultdict(list))
    representatives = {}
    for t in natural:
        for s in t["steps"]:
            k = (t["task_id"], credit_key(s, cfg.credit_state_mode))
            if s["valid"]:
                representatives.setdefault((k, s["action"]), (t, s))
    natural_states = {state for state, _ in representatives}
    for row in outcomes(evidence, cfg.horizon, cfg.credit_state_mode):
        k = (row["task_id"], row["key"])
        if k in natural_states and row["valid"]:
            groups[k][row["action"]].append(row["y"])
    task_success = {
        task: any(t["success"] for t in evidence if t["task_id"] == task)
        for task in {t["task_id"] for t in natural}
    }
    result = []
    for k, by_action in sorted(groups.items()):
        actions = sorted(by_action)
        entry = dict(
            task_id=k[0],
            key=k[1],
            credit_state_mode=cfg.credit_state_mode,
            actions=actions,
            counts={a: len(by_action[a]) for a in actions},
            trigger=False,
            confidence=None,
            confidence_kind="conditional_mc_advantage_sign_proxy",
            source_trajectory_id=None,
        )
        if not task_success[k[0]]:
            entry["reason"] = "no_success_support"
        elif len(actions) < 2:
            entry["reason"] = "missing_candidate"
        else:
            g = graph_state(*k)
            weights = np.array([policy[g].get(a, 0.0) for a in actions])
            # Invalid or missing action mass may exist. Do not renormalize it
            # away and report a false complete-policy advantage.
            if abs(weights.sum() - 1.0) > 1e-8:
                entry["reason"] = "missing_policy_mass"
                result.append(entry)
                continue
            rng = np.random.default_rng(int(digest([cfg.seed, stage, k])[:16], 16))
            draws = np.stack(
                [
                    rng.beta(
                        sum(by_action[a]) + 0.5,
                        len(by_action[a]) - sum(by_action[a]) + 0.5,
                        cfg.posterior_draws,
                    )
                    for a in actions
                ],
                axis=1,
            )
            adraw = draws - (draws @ weights)[:, None]
            point = np.array([graph_q[(g, a)] for a in actions])
            adv = point - point @ weights
            signs = [
                (
                    float(np.mean(adraw[:, i] > cfg.practical_epsilon))
                    if adv[i] > cfg.practical_epsilon
                    else (
                        float(np.mean(adraw[:, i] < -cfg.practical_epsilon))
                        if adv[i] < -cfg.practical_epsilon
                        else float(
                            np.mean(np.abs(adraw[:, i]) <= cfg.practical_epsilon)
                        )
                    )
                )
                for i in range(len(actions))
            ]
            entry.update(
                confidence=min(signs),
                point_q=point.tolist(),
                point_advantage=adv.tolist(),
            )
            entry["reason"] = (
                "low_coverage"
                if min(entry["counts"].values()) < cfg.min_count
                else (
                    "uncertain_advantage"
                    if min(signs) < cfg.sign_threshold
                    else "stable_proxy"
                )
            )
            entry["trigger"] = entry["reason"] != "stable_proxy"
        result.append(entry)
    return result, representatives


def history_shadow(natural, past, horizon):
    """只观测历史匹配与潜在价值；返回统计量，不把历史写入训练证据。

        LOTO prediction audit; no old data is fed to the live estimator or loss.

    Comparison is a fixed Beta(.5,.5) MC outcome forecast, not a claim that stale
    policy outcomes are unbiased estimates of the current policy's graph Q.
    """
    start = time.perf_counter()
    current = outcomes(natural, horizon)
    ids = [t["id"] for t in natural + past]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate history/current trajectory identity")
    if past and any(
        t["stage"] >= min(n["stage"] for n in natural) or t["source"] != "natural"
        for t in past
    ):
        raise ValueError("History must contain older natural trajectories")
    old = outcomes(past, horizon)
    old_by = defaultdict(list)
    cur_by = defaultdict(list)
    for r in old:
        old_by[(r["goal_id"], r["key"], r["action"])].append(r)
    for r in current:
        cur_by[(r["goal_id"], r["key"], r["action"])].append(r)
    comparisons = []
    for target in current:
        key = (target["goal_id"], target["key"], target["action"])
        h = old_by.get(key, [])
        if not h:
            continue
        c = [r for r in cur_by[key] if r["trajectory_id"] != target["trajectory_id"]]
        p0 = (sum(r["y"] for r in c) + 0.5) / (len(c) + 1)
        p1 = (sum(r["y"] for r in c + h) + 0.5) / (len(c + h) + 1)
        y = target["y"]
        comparisons.append(
            dict(
                **target,
                current_support=len(c),
                history_support=len(h),
                current_prediction=p0,
                history_prediction=p1,
                current_brier=(p0 - y) ** 2,
                history_brier=(p1 - y) ** 2,
                gain=(p0 - y) ** 2 - (p1 - y) ** 2,
                historical_ids=sorted({r["trajectory_id"] for r in h}),
                stale_policy_count=sum(
                    r["policy_version"] != target["policy_version"] for r in h
                ),
            )
        )
    grouped = defaultdict(list)
    for r in comparisons:
        grouped[r["goal_id"]].append(r["gain"])
    cluster_metrics = {}
    if len(grouped) >= 2:
        gains = np.array([np.mean(v) for v in grouped.values()])
        rng = np.random.default_rng(20260927)
        boot = rng.choice(gains, size=(1000, len(gains)), replace=True).mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        cluster_metrics = {
            "history/descriptive_gain_ci_low": float(lo),
            "history/descriptive_gain_ci_high": float(hi),
            "history/improved_goals": int((gains > 0).sum()),
            "history/worsened_goals": int((gains < 0).sum()),
        }
    return comparisons, dict(
        **{
            "history/shadow_available": int(bool(comparisons)),
            "history/matched_decisions": len(comparisons),
            "history/matched_goals": len(grouped),
            "history/coverage": len(comparisons) / max(1, len(current)),
            "history/current_policy_loss_reuse": 0,
            "history/live_credit_reuse": 0,
            "history/shadow_seconds": time.perf_counter() - start,
        },
        **cluster_metrics,
        **(
            {
                "history/task_mean_brier_gain": float(
                    np.mean([np.mean(v) for v in grouped.values()])
                )
            }
            if grouped
            else {}
        ),
    )


def failed_trajectory_ids(natural, limit=None):
    """Diagnostic cohort only; never changes rewards, gates or auxiliary eligibility."""
    groups = defaultdict(list)
    for trace in natural:
        groups[trace["task_id"]].append(trace)
    return {
        t["id"]
        for group in groups.values()
        if not any(t["success"] for t in group[:limit])
        for t in group[:limit]
    }


class UController:
    """编排自然采样→分支干预→信用重算→审计的单轮数据生命周期。"""

    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.pending = None

    @staticmethod
    def write(path, obj):
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))

    def history(self, stage):
        traces = []
        for file in sorted(self.output.glob("stage_*/natural.json")):
            n = int(file.parent.name.split("_")[1])
            if (
                max(0, stage - self.cfg.history_rounds) <= n < stage
                and (file.parent / "complete.json").exists()
            ):
                traces.extend(json.loads(file.read_text()))
        return (
            traces[-self.cfg.history_max_trajectories :]
            if self.cfg.history_max_trajectories
            else []
        )

    def process(self, natural, probe_fn, stage):
        """Probe callback returns (trace, physical_calls); never touches actor batch."""
        if self.pending is not None:
            raise RuntimeError("Previous stage has not completed its actor update")
        folder = self.output / f"stage_{stage:06d}"
        folder.mkdir(exist_ok=False)  # refuse ambiguous replay/overwrite on resume
        self.write(folder / "config.json", asdict(self.cfg))
        self.write(folder / "natural.json", natural)
        if any(t["stage"] != stage or t["source"] != "natural" for t in natural):
            raise ValueError("Stage/source mismatch")
        initial_start = time.perf_counter()
        before, q, pi = estimate(
            natural, [], self.cfg.horizon, self.cfg.credit_state_mode
        )
        pre, refs = gates(natural, natural, q, pi, self.cfg, stage)
        initial_seconds = time.perf_counter() - initial_start
        history_start = time.perf_counter()
        past = self.history(stage)
        shadow, metrics = history_shadow(natural, past, self.cfg.horizon)
        metrics["history/total_shadow_seconds"] = time.perf_counter() - history_start
        metrics["history/archive_trajectories_read"] = len(past)
        matched_ids = {tid for r in shadow for tid in r["historical_ids"]}
        metrics.update(
            {
                "history/unique_trajectories_used": len(matched_ids),
                "history/any_cross_round_use": int(bool(matched_ids)),
                "history/any_cross_run_use": 0,
                "history/loss_tokens": 0,
                "history/loss_token_fraction": 0.0,
                "history/value_rows": 0,
                "history/shadow_value_rows": sum(r["history_support"] for r in shadow),
                "history/age_rounds_available": int(bool(matched_ids)),
            }
        )
        if matched_ids:
            metrics["history/age_rounds_mean"] = float(
                np.mean([stage - t["stage"] for t in past if t["id"] in matched_ids])
            )
        self.write(folder / "history_shadow.json", shadow)
        probe_start = time.perf_counter()
        from .full_intervention import execute_interventions

        probes, events, calls, by_task, query_metrics = execute_interventions(
            natural, probe_fn, self.cfg, stage, pre, folder, self.write
        )
        metrics.update(query_metrics)
        metrics["u/extra_seconds"] = time.perf_counter() - probe_start
        recompute_start = time.perf_counter()
        after, q_after, pi_after = estimate(
            natural, probes, self.cfg.horizon, self.cfg.credit_state_mode
        )
        post, _ = gates(natural, natural + probes, q_after, pi_after, self.cfg, stage)
        metrics["prob/initial_estimate_gate_seconds"] = initial_seconds
        metrics["prob/recompute_gate_seconds"] = time.perf_counter() - recompute_start
        tasks = []
        for task in sorted({t["task_id"] for t in natural}):
            ts = [t for t in natural if t["task_id"] == task]
            tasks.append(
                dict(
                    task_id=task,
                    goal_id=ts[0]["goal_id"],
                    trajectories=len(ts),
                    successes=sum(t["success"] for t in ts),
                    reference_trajectories=min(
                        self.cfg.reference_natural_count, len(ts)
                    ),
                    reference_successes=sum(
                        t["success"] for t in ts[: self.cfg.reference_natural_count]
                    ),
                    partial_scores=[t["score"] for t in ts],
                    stop_reasons=[t["stop_reason"] for t in ts],
                    probes=by_task[task],
                )
            )
        metrics.update(
            {
                "u/extra_physical_env_calls": calls,
                "u/probes": len(probes),
                "u/extra_effective_steps": sum(len(t["steps"]) for t in probes),
                "u/extra_prefix_steps": sum(t["credit_start"] for t in probes),
                "u/extra_generated_tokens": sum(
                    t.get("generated_tokens", 0) for t in probes
                )
                + query_metrics.get("u/query_generated_tokens", 0),
                "u/extra_input_tokens": sum(
                    t.get("generated_input_tokens", 0) for t in probes
                )
                + query_metrics.get("u/query_input_tokens", 0),
                "u/extra_generated_requests": sum(
                    t.get("generated_requests", 0) for t in probes
                )
                + query_metrics.get("u/query_requests", 0),
                "u/triggered_tasks": len(
                    {g["task_id"] for g in pre if g["reason"] != "stable_proxy"}
                ),
                "u/served_tasks": sum(n > 0 for n in by_task.values()),
                "u/triggered_task_ratio": len(
                    {g["task_id"] for g in pre if g["reason"] != "stable_proxy"}
                )
                / len(tasks),
                "u/served_task_ratio": sum(n > 0 for n in by_task.values())
                / len(tasks),
                "u/still_uncertain_states": sum(
                    g["reason"] != "stable_proxy" for g in post
                ),
                "u/stable_proxy_states_after": sum(
                    g["reason"] == "stable_proxy" for g in post
                ),
                "u/na_states_after": sum(g["confidence"] is None for g in post),
                "u/probes_per_task_mean": float(np.mean([t["probes"] for t in tasks])),
                "u/probes_per_task_p90": float(
                    np.quantile([t["probes"] for t in tasks], 0.9)
                ),
                "u/probes_per_task_max": max(t["probes"] for t in tasks),
                "u/triggered_states": sum(g["reason"] != "stable_proxy" for g in pre),
                "u/budget_unserved_candidates": sum(
                    e["status"] == "global_budget_unserved" for e in events
                ),
                "tasks/n_instances": len(tasks),
                "tasks/reference_natural_count": self.cfg.reference_natural_count,
                f"tasks/reference_first{self.cfg.reference_natural_count}_eligible": sum(
                    t["reference_trajectories"] == self.cfg.reference_natural_count
                    for t in tasks
                ),
                f"tasks/reference_first{self.cfg.reference_natural_count}_all_failed": sum(
                    t["reference_trajectories"] == self.cfg.reference_natural_count
                    and t["reference_successes"] == 0
                    for t in tasks
                ),
                "tasks/initial_all_failed_count": sum(
                    t["reference_successes"] == 0 for t in tasks
                ),
                "tasks/initial_all_failed_ratio": sum(
                    t["reference_successes"] == 0 for t in tasks
                )
                / len(tasks),
                "tasks/all_natural_all_failed_count": sum(
                    t["successes"] == 0 for t in tasks
                ),
                "tasks/all_natural_all_failed_ratio": sum(
                    t["successes"] == 0 for t in tasks
                )
                / len(tasks),
                "prob/raw_q_changed_ratio": np.mean(
                    [abs(after[k]["q"] - before[k]["q"]) > 1e-8 for k in before]
                ).item(),
                "prob/raw_v_changed_ratio": np.mean(
                    [abs(after[k]["v"] - before[k]["v"]) > 1e-8 for k in before]
                ).item(),
                "prob/raw_advantage_changed_ratio": np.mean(
                    [abs(after[k]["a"] - before[k]["a"]) > 1e-8 for k in before]
                ).item(),
                "u/policy_updated_natural_states": sum(
                    pi[s] != pi_after[s] for s in pi
                ),
                "u/new_continuation_actions_at_natural_states": sum(
                    len(set(pi_after[s]) - set(pi[s])) for s in pi
                ),
                "u/policy_continuation_samples": sum(
                    not s.get("forced", False)
                    for t in probes
                    for s in t["steps"][t.get("credit_start", 0) :]
                ),
                "u/policy_forced_samples": 0,
                "prob/observer_errors": 0,
            }
        )
        for k, n in Counter(g["reason"] for g in pre).items():
            metrics["u/reason/" + k] = n
        for k in range(max(t["trajectories"] for t in tasks) + 1):
            metrics[f"tasks/initial_success_count_{k}"] = sum(
                t["successes"] == k for t in tasks
            )
        rows = [
            dict(trajectory_id=k[0], step=k[1], before=before[k], after=after[k])
            for k in before
        ]
        for name, obj in [
            ("credit.json", rows),
            ("tasks.json", tasks),
            ("gate_before.json", pre),
            ("gate_after.json", post),
            ("probes.json", probes),
            ("interventions.json", events),
            ("metrics.json", metrics),
        ]:
            self.write(folder / name, obj)
        failed = {t["task_id"] for t in tasks if t["successes"] == 0}
        rescued = {
            t["task_id"] for t in probes if t["success"] and t["task_id"] in failed
        }
        metrics.update(
            {
                "u/all_failed_rescued_tasks": len(rescued),
                "u/all_failed_remaining_tasks": len(failed - rescued),
                "graph/natural_credit_nodes": len(
                    {(t["task_id"], s["anchor"]) for t in natural for s in t["steps"]}
                ),
                "graph/pooled_credit_nodes": len(
                    {
                        (t["task_id"], s["anchor"])
                        for t in natural + probes
                        for s in t["steps"][t.get("credit_start", 0) :]
                    }
                ),
            }
        )
        self.write(folder / "metrics.json", metrics)
        self.pending = folder
        return before, after, metrics

    def finish(self, final_advantage_metrics):
        if self.pending is None:
            raise RuntimeError("No pending U stage")
        self.write(self.pending / "complete.json", final_advantage_metrics)
        self.pending = None
