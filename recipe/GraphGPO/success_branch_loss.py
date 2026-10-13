"""Separate supervised labels; never feed forced responses to PPO ratios."""

import collections
import math
import random
import torch


def audit_critical_nll(actor, payload):
    """只观测关键响应在更新前后的 NLL；不参与反向传播。"""
    if not payload:
        return None
    selected = {}
    for batch in payload["by_update"]:
        for row in batch:
            if row is not None and row["intervention"]:
                key = (tuple(row["prompt_ids"]), tuple(row["response_ids"]))
                selected.setdefault(key, row)
    if not selected:
        return None
    device = next(actor.actor_module.parameters()).device
    actor.actor_module.eval()
    values = []
    with torch.no_grad():
        for row in list(selected.values())[:8]:
            ids = torch.tensor(
                [row["prompt_ids"] + row["response_ids"]], device=device, dtype=torch.long
            )
            data = dict(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(ids.shape[1], device=device)[None],
                responses=torch.tensor([row["response_ids"]], device=device),
            )
            _, logp = actor._forward_micro_batch(data, temperature=1.0, calculate_entropy=False)
            values.append(float(-logp.float().mean()))
    actor.actor_module.train()
    return sum(values) / len(values)


def schedule_auxiliary(normal, repairs, optimizer_steps, seed):
    """Every rescued task receives a critical-step label before repeats dominate.

    Each unit is six ordinary successful responses plus one critical response
    and one unforced suffix response. An absent ordinary pool retains its zero
    mass instead of increasing the repair coefficient.
    """
    if not repairs:
        return [], {}
    assert optimizer_steps > 0
    rng = random.Random(seed)
    tasks = sorted(repairs)
    units = max(1, math.ceil(len(tasks) / optimizer_steps))
    pending = []
    plan = []
    counts = collections.Counter()
    ordinary_tasks = sorted(normal)
    for _ in range(optimizer_steps):
        batch = []
        for _ in range(units):
            if not pending:
                pending = rng.sample(tasks, len(tasks))
            goal = pending.pop()
            critical, suffix = repairs[goal]
            assert critical["intervention"] and not critical["replayed_prefix"]
            assert all(not r["forced"] and not r["replayed_prefix"] for r in suffix)
            assert suffix or critical["terminal_success"]
            counts[goal] += 1
            for _ in range(6):
                if ordinary_tasks:
                    batch.append(rng.choice(normal[rng.choice(ordinary_tasks)]))
                else:
                    batch.append(None)
            batch.extend([critical, rng.choice(suffix) if suffix else None])
        plan.append(batch)
    assert set(counts) == set(tasks)
    assert max(counts.values()) - min(counts.values()) <= 1
    return plan, dict(counts)


def backward_auxiliary(actor, payload, update_index):
    """Accumulate auxiliary gradients before the existing optimizer step."""
    if not payload or not payload["by_update"]:
        return {}
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if initialized else 1
    rank = torch.distributed.get_rank() if initialized else 0
    assert actor.ulysses_sequence_parallel_size == 1
    rows = payload["by_update"][update_index]
    assert len(rows) % world == 0
    local = rows[rank::world]
    device = next(actor.actor_module.parameters()).device
    metrics = {}
    for row in local:
        # All ranks must perform the same number of FSDP forwards/backwards.
        # Missing normal data is represented by a zero-weight repair example.
        weight = 1.0
        if row is None:
            row = next(r for r in rows if r is not None)
            weight = 0.0
        prompt, response = row["prompt_ids"], row["response_ids"]
        ids = torch.tensor([prompt + response], dtype=torch.long, device=device)
        batch = dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(ids.shape[1], device=device).unsqueeze(0),
            responses=torch.tensor([response], dtype=torch.long, device=device),
        )
        _, logp = actor._forward_micro_batch(batch, temperature=1.0, calculate_entropy=False)
        nll = -logp.float().mean()
        # 空槽位保持零权重且保留分母，避免缺少普通成功数据时放大修复信号。
        loss = payload["coefficient"] * weight * nll / len(local)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite success-branch supervised loss")
        loss.backward()
        values = {
            "aux/response_nll": float(nll.detach()),
            "aux/weighted_loss": float(loss.detach()),
            "aux/active_response": weight,
            "aux/critical_response": weight * int(row["intervention"]),
            "aux/historical_response": 0.0,
        }
        for key, value in values.items():
            metrics.setdefault(key, []).append(value)
    return metrics
