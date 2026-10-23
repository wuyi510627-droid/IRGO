"""Adapters for current-round alternatives and verified-success labels."""

import collections
import json
import math
import numpy as np
import torch
from pathlib import Path
from .prob_u import digest
from .success_branch_loss import schedule_auxiliary
from .success_data_tools import trim_unexecuted_tail


def propose_queries(runner, natural, pre, cfg, stage):
    """用同一策略一次性提出搜索备选；每任务至多一次请求，不循环重试。

    这部分生成开销计入干预成本，查询提议本身不算成功轨迹或 PPO 样本。
    """
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from .full_search import parse_search_proposals

    groups = collections.defaultdict(list)
    for t in natural:
        groups[t["task_id"]].append(t)
    candidates = {}
    seen = collections.defaultdict(set)
    for t in natural:
        for s in t["steps"]:
            seen[s["key"]].add(s["action"])
            if (
                s["valid"]
                and s["action"].startswith("search[")
                and s["step"] < t["max_steps"]
            ):
                candidates.setdefault(s["key"], (t, s))
    chosen = sorted(
        candidates.values(),
        key=lambda pair: (
            any(t["success"] for t in groups[pair[0]["task_id"]]),
            pair[1]["step"],
            digest([cfg.seed, stage, pair[1]["key"]]),
        ),
    )
    # At most one request per task, one pass, no regeneration after invalid JSON.
    selected = []
    tasks = set()
    for t, s in chosen:
        if t["task_id"] in tasks:
            continue
        tasks.add(t["task_id"])
        selected.append((t, s))
        if len(selected) >= 16:
            break
    if not selected:
        return {}, {
            "u/query_requests": 0,
            "u/query_generated_tokens": 0,
            "u/query_input_tokens": 0,
        }
    tokenizer = runner.collector.tokenizer
    prompts = []
    for t, s in selected:
        content = (
            "Suggest two different alternative search queries for the shopping task below. "
            "Keep the requested product and its constraints. Do not supply product IDs or answers. "
            "Return only a JSON array of two English query strings, without action tags.\n"
            "Task: " + t["task_text"] + "\nPrevious query: " + s["action"]
        )
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
        )
        assert len(ids) <= 5120
        prompts.append(ids)
    width = max(map(len, prompts))
    n = len(prompts)
    pad = tokenizer.pad_token_id
    inputs = torch.full((n, width), pad, dtype=torch.long)
    mask = torch.zeros_like(inputs)
    raw = np.empty(n, dtype=object)
    for i, ids in enumerate(prompts):
        inputs[i, -len(ids) :] = torch.tensor(ids)
        mask[i, -len(ids) :] = 1
        raw[i] = ids
    batch = DataProto.from_dict(
        tensors=dict(
            input_ids=inputs,
            attention_mask=mask,
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
        ),
        non_tensors=dict(raw_prompt_ids=raw),
        meta_info=dict(
            validate=False,
            do_sample=True,
            u_sampling_seed=int(digest([cfg.seed, stage, "queries"])[:8], 16)
            % (2**31 - 1),
        ),
    )
    padded, extra = pad_dataproto_to_divisor(batch, runner.actor.world_size)
    input_tokens = int(padded.batch["attention_mask"].sum().item())
    all_result = runner.actor.generate_sequences(padded)
    response_width = all_result.batch["responses"].shape[1]
    generated_tokens = int(
        all_result.batch["attention_mask"][:, -response_width:].sum()
    )
    result = unpad_dataproto(all_result, pad_size=extra)
    outputs = tokenizer.batch_decode(
        result.batch["responses"], skip_special_tokens=True
    )
    query_map = {}
    audit = []
    for (t, s), ids, text in zip(selected, prompts, outputs):
        actions = parse_search_proposals(text, seen[s["key"]])
        query_map[s["key"]] = actions
        audit.append(
            dict(
                task_id=t["task_id"],
                context_key=s["key"],
                proposal_prompt_ids=ids,
                raw_response=text,
                actions=actions,
            )
        )
    folder = (
        Path(runner.collector.config.trainer.default_local_dir)
        / "u_evidence"
        / f"stage_{stage:06d}"
    )
    (folder / "query_proposals.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2)
    )
    return query_map, dict(
        {
            "u/query_requests": len(padded),
            "u/query_unique_requests": n,
            "u/query_generated_tokens": generated_tokens,
            "u/query_input_tokens": input_tokens,
            "u/query_valid_alternatives": sum(map(len, query_map.values())),
        }
    )


def attach_auxiliary(bridge, batch):
    """仅为本轮原始全失败、且被真实成功分支救回的组建立学习标签。

    每任务保留首个成功分支的强制关键响应及自然后续响应。辅助批次
    按 6:1:1 混合普通成功响应、关键响应、后续响应；单位是响应步骤。
    仍然只有同一个策略模型，在已有优化器更新前累计辅助梯度。
    """
    from agent_system.environments.env_package.webshop import webshop_projection

    folder = bridge.controller.pending
    cfg = bridge.config.algorithm.u
    if not cfg.get("auxiliary_enabled", False):
        return
    tokenizer = bridge.collector.tokenizer
    natural = bridge.collector.u_capture.traces
    probes = json.loads((folder / "probes.json").read_text())
    normal = collections.defaultdict(list)
    repairs = {}
    rows = []
    drops = collections.Counter()
    all_failed = {
        task: not any(t["success"] for t in natural if t["task_id"] == task)
        for task in {t["task_id"] for t in natural}
    }

    def row(t, s, intervention):
        if not s["valid"] or not s["response_ids"]:
            drops["invalid_or_empty"] += 1
            return None
        try:
            response, _ = trim_unexecuted_tail(
                tokenizer, s["response_ids"], s["action"]
            )
            if response[-1] != tokenizer.eos_token_id:
                response.append(tokenizer.eos_token_id)
            assert len(response) <= 512 and len(s["prompt_ids"]) <= 5120
            text = tokenizer.decode(
                response, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            actions, valid = webshop_projection([text])
            assert valid[0] and actions[0] == s["action"]
        except (AssertionError, ValueError):
            drops["unusable_executed_response"] += 1
            return None
        r = dict(
            index=len(rows),
            task_id=t["task_id"],
            goal_id=t["goal_id"],
            trajectory_id=t["id"],
            step=s["step"],
            prompt_ids=s["prompt_ids"],
            response_ids=response,
            action=s["action"],
            intervention=intervention,
            forced=s["forced"],
            replayed_prefix=s.get("mode") == "prefix",
            stage=bridge.stage,
            policy_version=t["policy_version"],
            verified_terminal_success=True,
            terminal_success=s["won"],
            historical=False,
        )
        rows.append(r)
        return r

    for t in natural:
        assert t["stage"] == bridge.stage
        if t["success"]:
            for s in t["steps"]:
                r = row(t, s, False)
                if r is not None:
                    normal[t["task_id"]].append(r)
    for t in probes:
        assert (
            t["stage"] == bridge.stage
            and t["policy_version"] == natural[0]["policy_version"]
        )
        if not t["success"] or not all_failed[t["task_id"]] or t["task_id"] in repairs:
            continue
        start = t["credit_start"]
        critical = row(t, t["steps"][start], True)
        suffix = [
            r
            for s in t["steps"][start + 1 :]
            if not s["forced"] and (r := row(t, s, False)) is not None
        ]
        if critical is not None:
            if not suffix:
                if not t["steps"][start]["won"]:
                    drops["no_valid_suffix"] += 1
                    continue
            repairs[t["task_id"]] = (critical, suffix)
    updates = int(bridge.config.actor_rollout_ref.actor.ppo_epochs) * math.ceil(
        len(batch) / int(bridge.config.actor_rollout_ref.actor.ppo_mini_batch_size)
    )
    plan, counts = schedule_auxiliary(
        normal, repairs, updates, int(digest([cfg.seed, bridge.stage, "aux"])[:8], 16)
    )
    bridge.controller.write(
        folder / "auxiliary_data.json",
        dict(
            rows=rows,
            by_update=[[None if r is None else r["index"] for r in rs] for rs in plan],
            critical_exposures=counts,
            dropped=dict(drops),
            historical_rows=0,
            coefficient=cfg.auxiliary_coefficient,
            ppo_optimizer_updates=updates,
            terminal_critical_only_tasks=sum(not pair[1] for pair in repairs.values()),
        ),
    )
    if plan:
        batch.meta_info["u_aux_payload"] = dict(
            by_update=plan,
            coefficient=float(cfg.auxiliary_coefficient),
            critical_exposures=counts,
        )
    bridge.last_metrics.update(
        {
            "aux/enabled": 1,
            "aux/tasks": len(repairs),
            "aux/planned_critical_exposures": sum(counts.values()),
            "aux/coefficient": float(cfg.auxiliary_coefficient),
            "aux/planned_response_rows": sum(r is not None for rs in plan for r in rs),
            "cost/aux_planned_input_tokens": sum(
                len(r["prompt_ids"]) for rs in plan for r in rs if r is not None
            ),
            "cost/aux_planned_response_tokens": sum(
                len(r["response_ids"]) for rs in plan for r in rs if r is not None
            ),
            "aux/historical_rows": 0,
            "aux/dropped_responses": sum(drops.values()),
            "aux/normal_tasks": len(normal),
        }
    )


def compact_stage_evidence(folder):
    """训练后无损压缩大张量审计文件；核验往返哈希后才移除原始副本。"""
    import gzip, hashlib

    for name in ["actor_input.pt", "final_advantages.pt"]:
        source = folder / name
        if not source.exists():
            continue
        target = folder / (name + ".gz")
        tmp = folder / (name + ".gz.partial")
        h = hashlib.sha256()
        with source.open("rb") as inp, gzip.open(tmp, "wb", compresslevel=1) as out:
            for chunk in iter(lambda: inp.read(8 * 1024**2), b""):
                h.update(chunk)
                out.write(chunk)
        check = hashlib.sha256()
        with gzip.open(tmp, "rb") as inp:
            for chunk in iter(lambda: inp.read(8 * 1024**2), b""):
                check.update(chunk)
        assert h.hexdigest() == check.hexdigest()
        tmp.replace(target)
        (folder / (name + ".compression.json")).write_text(
            json.dumps(
                dict(
                    original_sha256=h.hexdigest(),
                    original_bytes=source.stat().st_size,
                    compressed_bytes=target.stat().st_size,
                    roundtrip_verified=True,
                )
            )
        )
        source.unlink()
