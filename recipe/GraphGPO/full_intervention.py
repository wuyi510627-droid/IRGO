"""Bounded alternative branches with an explicit, auditable site heuristic."""

from collections import Counter, defaultdict
from .prob_u import digest, credit_key, estimate, gates


def make_candidates(
    natural, before_gates, query_map, stage, seed, site_selection="heuristic"
):
    """选点而非判定动作正确性：条件检查优先，其次证据稀缺程度。

    已有成功的组仅考虑信用代理尚不稳定的位置；全失败组不受此筛选。
    同一任务最多保留两个完整上下文、四个替代动作，排除该完整上下文
    已出现的动作。动作顺序由确定性哈希给出，不是按真实成功率排序。
    """
    if site_selection not in ("heuristic", "random"):
        raise ValueError("Unsupported site selection control")
    known = defaultdict(set)
    count = defaultdict(Counter)
    for t in natural:
        for s in t["steps"]:
            known[s["key"]].add(s["action"])
            if s["valid"]:
                count[t["task_id"], s["anchor"]][s["action"]] += 1
    success = {t["task_id"]: False for t in natural}
    for t in natural:
        success[t["task_id"]] |= t["success"]
    uncertain = {
        (g["task_id"], g["key"]) for g in before_gates if g["reason"] != "stable_proxy"
    }
    sites = {}
    for t in natural:
        for s in t["steps"]:
            if s["step"] >= t["max_steps"]:
                continue
            if success[t["task_id"]] and (t["task_id"], s["anchor"]) not in uncertain:
                continue
            choices = {
                "click[" + a.lower() + "]"
                for a in s["clickables_before"]
                if a.lower() != "search"
            }
            choices.update(query_map.get(s["key"], []))
            choices -= known[s["key"]]
            if not choices:
                continue
            pre, aft = s.get("checked_before", {}), s.get("checked_after", {})
            if (
                s["action"].startswith("search[")
                and aft.get("visible_static_feasible") == 0
            ):
                priority, reason = 4, "search_no_verified_feasible_visible_product"
            elif (
                pre.get("asin") is None
                and aft.get("asin") is not None
                and aft.get("static_ok") is False
            ):
                priority, reason = 3, "entered_static_mismatch"
            elif pre.get("static_ok") and pre.get("option_fraction", 1) < 1:
                priority, reason = 2, "feasible_product_missing_options"
            elif not s["valid"] or s["anchor"] == s["next_anchor"]:
                priority, reason = 1, "invalid_or_no_observation_change"
            else:
                priority, reason = 0, "uncertain_credit_without_verified_progress"
            counts = count[t["task_id"], s["anchor"]]
            row = dict(
                task=t["task_id"],
                parent=t,
                step=s,
                priority=priority,
                reason=reason,
                scarcity=min(counts.values()) if counts else 0,
                actions=sorted(
                    choices, key=lambda a: digest([seed, stage, "action", s["key"], a])
                ),
                all_failed=not success[t["task_id"]],
            )
            key = (t["task_id"], s["key"])
            old = sites.get(key)
            if old is None or (priority, -s["step"], t["id"]) > (
                old["priority"],
                -old["step"]["step"],
                old["parent"]["id"],
            ):
                sites[key] = row
    by_task = defaultdict(list)
    for row in sites.values():
        by_task[row["task"]].append(row)
    plan = {}
    task_priority = {}
    for task, choices in by_task.items():
        # Freeze task scheduling from the whole eligible pool. Randomizing the
        # two selected sites must not also change which tasks receive budget.
        task_priority[task] = max(x["priority"] for x in choices)
        choices.sort(
            key=lambda x: (
                -x["priority"],
                x["scarcity"],
                x["step"]["step"],
                digest([seed, stage, x["step"]["key"]]),
            )
        )
        if site_selection == "random":
            choices.sort(
                key=lambda x: digest([seed, stage, "random_site", x["step"]["key"]])
            )
        # Two contexts; visit each before a second alternative. The default
        # retains the historical heuristic order; R only changes this order.
        choices = choices[:2]
        candidates = []
        for action_index in range(4):
            for site in choices:
                if action_index < len(site["actions"]):
                    candidates.append(
                        dict(**site, action=site["actions"][action_index])
                    )
        plan[task] = candidates[:4]
    order = sorted(
        plan,
        key=lambda task: (
            -task_priority[task],
            not plan[task][0]["all_failed"],
            digest([seed, stage, "task", task]),
        ),
    )
    return plan, order


def execute_interventions(natural, runner, cfg, stage, pre, folder, write):
    """按任务轮流执行有界探测；前缀重放也计入物理环境步预算。

    保存选择计划后才读取分支结果，避免事后选点。全失败组找到真实
    成功即停止；已执行两条时可按选中节点的条件稳定代理提前停止。
    所有有效证据仍进入后续信用重算，不按第二次 gate 回退原优势。
    """
    if min(cfg.max_probes, cfg.max_probes_per_task, cfg.max_extra_env_calls) == 0:
        # Pure Prob controls still collect audit traces. A zero intervention
        # budget must also suppress proposal generation, not just branch calls.
        write(folder / "selection_before_outcomes.json", [])
        write(folder / "intervention_stops.json", {})
        return (
            [],
            [],
            0,
            Counter(),
            {
                "u/query_requests": 0,
                "u/query_generated_tokens": 0,
                "u/query_input_tokens": 0,
            },
        )
    query_map, proposal_metrics = runner.propose_queries(natural, pre, cfg, stage)
    plan, task_order = make_candidates(
        natural, pre, query_map, stage, cfg.seed, cfg.site_selection
    )
    frozen_plan = [
        {k: v for k, v in row.items() if k not in ("parent", "step", "actions")}
        | dict(
            parent_id=row["parent"]["id"],
            context_key=row["step"]["key"],
            step=row["step"]["step"],
        )
        for task in task_order
        for row in plan[task]
    ]
    write(folder / "selection_before_outcomes.json", frozen_plan)
    probes = []
    events = []
    calls = 0
    counts = Counter()
    stopped = {}
    seen = set()
    for pass_index in range(cfg.max_probes_per_task):
        for task in task_order:
            if task in stopped or pass_index >= len(plan[task]):
                continue
            row = plan[task][pass_index]
            ref, s = row["parent"], row["step"]
            reserve = ref["max_steps"]
            if (
                len(probes) >= cfg.max_probes
                or calls + reserve > cfg.max_extra_env_calls
            ):
                events.append(
                    dict(
                        task_id=task,
                        status="global_budget_unserved",
                        context_key=s["key"],
                        action=row["action"],
                    )
                )
                continue
            identity = (task, s["key"], row["action"])
            assert identity not in seen
            seen.add(identity)
            ident = digest([stage, identity, "probe"])
            t, cost = runner(ref, s, row["action"], ident)
            assert (
                0 < cost <= reserve
                and t["id"] == ident
                and t["policy_version"] == ref["policy_version"]
            )
            branch = t["steps"][t["credit_start"]]
            assert (
                branch["key"] == s["key"]
                and branch["prompt_ids"] == s["prompt_ids"]
                and branch["action"] == row["action"]
                and branch["forced"]
            )
            probes.append(t)
            calls += cost
            counts[task] += 1
            events.append(
                dict(
                    task_id=task,
                    context_key=s["key"],
                    action=row["action"],
                    status="executed",
                    reason=row["reason"],
                    trajectory_id=t["id"],
                    physical_calls=cost,
                    success=t["success"],
                )
            )
            write(folder / "probes.json", probes)
            write(folder / "interventions.json", events)
            if row["all_failed"] and t["success"]:
                stopped[task] = "verified_success_found"
            elif counts[task] == 2:
                _, q, pi = estimate(natural, probes, cfg.horizon, cfg.credit_state_mode)
                post, _ = gates(natural, natural + probes, q, pi, cfg, stage)
                relevant = [
                    g
                    for g in post
                    if g["task_id"] == task
                    and g["key"] in {x["step"]["anchor"] for x in plan[task]}
                ]
                if relevant and all(g["reason"] == "stable_proxy" for g in relevant):
                    stopped[task] = "selected_nodes_stable_proxy"
    write(folder / "intervention_stops.json", stopped)
    return probes, events, calls, counts, proposal_metrics
