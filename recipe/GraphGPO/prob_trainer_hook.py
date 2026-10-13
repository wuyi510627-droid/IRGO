"""Pure torch hook for wiring Prob step credits into the existing GRPO batch."""

from __future__ import annotations
from collections import Counter, defaultdict
from typing import Mapping, Any
import numpy as np
import torch

try:  # package import inside recipe.GraphGPO
    from .probabilistic_credit import (
        build_posterior,
        canonical_state,
        scoped_state,
        policy_qva_values,
        transition_rows,
    )
except ImportError:  # standalone tests from the integration directory
    from probabilistic_credit import (
        build_posterior,
        canonical_state,
        scoped_state,
        policy_qva_values,
        transition_rows,
    )


# [PROB-ALIGN-1] 修改边界：只返回每条环境转移的原始 A_prob（Q^pi - V^pi）。
# 已移除 Prob 自己的按 uid 归一化和 token 广播，交由 GraphGPO 原函数统一处理。
def compute_prob_step_rewards(non_tensor_batch: Mapping[str, Any], device=None, horizon=8):
    """Return raw per-transition A^pi, before the shared penalty/composition.

    Normalization, episode credit, weights and token masking belong to the
    unchanged GraphGPO composer. The trainer applies its existing invalid-action
    penalty to these rewards before calling that composer.
    """
    rows = transition_rows(non_tensor_batch)
    intervention_mask = non_tensor_batch.get("intervention_mask")
    if intervention_mask is not None and len(intervention_mask) != len(rows):
        raise ValueError("intervention_mask length must match rollout rows")
    if intervention_mask is not None and bool(np.asarray(intervention_mask, dtype=bool).any()):
        raise ValueError("intervention rows must not enter the on-policy GRPO batch")
    post = build_posterior(non_tensor_batch, scope_by_uid=True)
    action_counts = defaultdict(Counter)
    for uid, state, url, action, _, _ in rows:
        action_counts[scoped_state(uid, canonical_state(state, url))][str(action)] += 1
    policy = {
        state: {action: count / sum(counts.values()) for action, count in counts.items()}
        for state, counts in action_counts.items()
    }
    _, _, advantage = policy_qva_values(post, policy=policy, horizon=horizon)
    step = np.asarray(
        [
            advantage.get((scoped_state(uid, canonical_state(s, u)), str(a)), 0.0)
            for uid, s, u, a, _, _ in rows
        ],
        dtype=np.float32,
    )
    # [PROB-ALIGN-1] 此处尚未施加 invalid-action penalty，也未合成 episode advantage。
    # 返回一维 step_rewards，由 trainer 按原有顺序完成这些后续处理。
    return torch.as_tensor(step, dtype=torch.float32, device=device)
