"""
两层 best-level 融合的 step 级优势估计。

    L0  节点 = anchor_obs 精确哈希                  <- GraphGPO 原样，一字不改
    L1  节点 = SBERT 余弦贪心聚类，δ=0.85，per-task
    融合  A(i) = A_L0(i)  if |A_L0(i)| > 0
              = A_L1(i)  otherwise                 <- 约 18% 的步走这条

离线诊断结论（WebShop / GiGPO-7B / 每 task 8 条 rollout，见 实验总结与方案现状.md）：
    活图 A^G=0        32.6% -> 23.3%
    近似闭环的纠错步    零信号率 15.1% -> 6.2%（仅活图；死图上无效）
    聚合符号翻转率      1.157% -> 1.170%

四条硬约束，改动前先读，都是实验定出来的不是超参：

1. **δ 不得低于 0.82。** 条件符号翻转率在 δ≈0.81 有陡崖（δ=0.82 时 7.3%，
   δ=0.81 时 25.7%，δ=0.60 时 53.9% 已比抛硬币还差），两个采样温度下崖的
   位置一致。越崖补进来的是噪声梯度。

2. **L0 必须排在第一位且原样保留。** 81.8% 的步在 L0 就有信号，融合对它们
   零改动。单层软合并对 100% 的步施加偏差，其中八成是白付的。

3. **不许跨层拼接分组与 d。** 实测「软分组 + 精确 d」(ρ=0.84) 比纯精确
   (0.99) 和纯软层 (0.92) 都差。每一层的 group_uid 和 step_reward 必须来自
   **同一张图、同一个 canonicalizer**。

4. **两层就够。** 崖以上从 2 层加到 6 层只多赚 1.6pp 的信号，不值 4 次额外
   的聚类 + Dijkstra。

评测注意（踩过的坑）：**不要用聚合的符号翻转率判断这个方法有没有引入偏差。**
软层只在「L0 为 0」的步上开火，而参考答案在这些步上大多也是 0，不进翻转率
分母。实测 L1 层自身的**条件**翻转率是 5.8%（L0 是 1.2%，5 倍），但只贡献
1.7% 的分母，聚合只动 +0.013%。必须逐层做条件统计。
"""
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from gigpo.core_gigpo import build_step_group, episode_norm_reward, step_norm_reward

from .core_graph import ComplexGraph, compute_graph_path_returns, to_hashable

SUCCESS_STATE = "success"
# 约束 1：低于此值直接拒绝，不给"试试更松会不会更好"的机会
DELTA_FLOOR = 0.82


class SBERTCanonicalizer:
    """
    SBERT 余弦贪心聚类的 canonicalizer，接口与 core_graph.StateCanonicalizer 兼容
    （只需要 .canonicalize(state)）。

    与 StateCanonicalizer 的两点不同：
      * 相似度用 SBERT 余弦，不是 SequenceMatcher 的字符匹配率
      * **两趟**：先收齐一个 task 的全部状态、排序、再聚类，而不是边遍历边聚。
        排序是为了可复现——贪心聚类结果依赖处理顺序，不排序的话同一批数据
        两次运行会得到不同的图。
    """

    def __init__(self, mapping: Dict):
        self._m = mapping

    def canonicalize(self, state):
        h = to_hashable(state)
        return self._m.get(h, h)

    @classmethod
    def fit(cls, states: List[str], encoder, threshold: float, cache: Dict):
        """states: 该 task 的全部原始状态字符串（可含重复）。"""
        uniq = sorted({s for s in states if isinstance(s, str) and s != SUCCESS_STATE})
        if not uniq:
            return cls({})
        vecs = encode_cached(uniq, encoder, cache)
        reps: List[str] = []
        rep_vecs: List[np.ndarray] = []
        mapping: Dict = {}
        for s, v in zip(uniq, vecs):
            hit = None
            for k, rv in enumerate(rep_vecs):
                if float(np.dot(v, rv)) >= threshold:
                    hit = k
                    break
            if hit is None:
                reps.append(s)
                rep_vecs.append(v)
                mapping[to_hashable(s)] = to_hashable(s)
            else:
                mapping[to_hashable(s)] = to_hashable(reps[hit])
        return cls(mapping)


_ENCODER = None


def get_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                device: Optional[str] = None):
    """进程内单例。默认放 CPU——编码量小，占 GPU 显存不划算。"""
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer(
            model_name, device=device or os.environ.get("MULTILAYER_SBERT_DEVICE", "cpu"))
    return _ENCODER


def encode_cached(states: List[str], encoder, cache: Dict) -> np.ndarray:
    """
    按状态字符串做全局缓存。WebShop 的状态高度重复（同一批 rollout 里
    22254 步只有 2240 个不同状态），不缓存的话每个 batch 重编码是主要开销。
    """
    todo = [s for s in states if s not in cache]
    if todo:
        vs = encoder.encode(todo, batch_size=64, normalize_embeddings=True,
                            show_progress_bar=False)
        for s, v in zip(todo, vs):
            cache[s] = np.asarray(v, dtype=np.float32)
    return np.stack([cache[s] for s in states])


def build_soft_canonicalizers(data, threshold: float, encoder, cache: Dict) -> Dict:
    """每个 uid（task）独立聚类——对齐 GraphGPO「不跨 task 合并」的原则。"""
    uids = data.non_tensor_batch["uid"]
    anchor = data.non_tensor_batch["anchor_obs"]
    nxt = data.non_tensor_batch["next_obs"]
    out = {}
    for uid in set(uids):
        idx = np.where(uids == uid)[0]
        states = [anchor[i] for i in idx] + [nxt[i] for i in idx]
        out[uid] = SBERTCanonicalizer.fit(states, encoder, threshold, cache)
    return out


def _layer_step_advantage(batch, Gs, canonicalizers, gamma, normalize_distance,
                          index, response_mask, epsilon, remove_std):
    """
    一层的 step 级优势。**分组与 d 必须来自同一个 canonicalizer**（约束 3）：
    group_uid 用 canonicalize 后的 anchor_obs，step_reward 用同一张图的 d。
    """
    ComplexGraph.Calculate_all_shortest_path(Gs)
    step_rewards = compute_graph_path_returns(
        batch=batch, Gs=Gs, gamma=gamma, normalize_distance=normalize_distance)

    anchor = batch.non_tensor_batch["anchor_obs"]
    if canonicalizers is None:
        keys = anchor
    else:
        keys = np.array(
            [canonicalizers[index[i]].canonicalize(anchor[i]) for i in range(len(anchor))],
            dtype=object)
    gids = build_step_group(keys, index)
    return step_norm_reward(step_rewards, response_mask, gids, epsilon, remove_std)


def compute_multilayer_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    batch,
    index: np.ndarray,
    traj_index: np.ndarray,
    gamma: float,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    episode_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
    delta: float = 0.85,
    normalize_distance: bool = False,
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    _cache: Dict = {},
):
    """
    两层 best-level 融合的 GraphGPO 优势。签名向 compute_graphgpo_outcome_advantage
    看齐，额外需要 `batch`（要重建第二张图）和 `gamma`。

    返回 (advantages, returns)，与原接口一致。
    """
    if delta < DELTA_FLOOR:
        raise ValueError(
            f"delta={delta} 低于安全下界 {DELTA_FLOOR}。条件符号翻转率在 δ≈0.81 "
            f"有陡崖（0.82 时 7.3% -> 0.81 时 25.7%），越崖补进来的是噪声梯度，"
            f"而**聚合指标看不出来**。要跨过这条线必须先用逐层条件统计重新验证。")

    remove_std = {"mean_std_norm": False, "mean_norm": True}[mode]

    # ── L0：精确哈希，GraphGPO 原样 ──────────────────────────────
    Gs0 = ComplexGraph.from_data(batch, enable_similarity=False)
    adv0 = _layer_step_advantage(batch, Gs0, None, gamma, normalize_distance,
                                 index, response_mask, epsilon, remove_std)

    # ── L1：SBERT 软合并 ─────────────────────────────────────────
    canon = build_soft_canonicalizers(
        batch, delta, get_encoder(sbert_model), _cache.setdefault("emb", {}))
    Gs1 = ComplexGraph.from_data(batch, enable_similarity=False,
                                 canonicalizers=canon)
    adv1 = _layer_step_advantage(batch, Gs1, canon, gamma, normalize_distance,
                                 index, response_mask, epsilon, remove_std)

    # ── best-level 融合：逐步取第一个非零的层 ─────────────────────
    # 用 response_mask 加权的绝对值判断该步是否有信号——adv 是 (bs, resp_len)
    # 的平铺，同一步所有 token 值相同，取第一个有效 token 即可。
    with torch.no_grad():
        flat0 = (adv0 * response_mask).abs().sum(dim=-1)
        has0 = (flat0 > 1e-8).float().unsqueeze(-1)
        step_advantages = has0 * adv0 + (1.0 - has0) * adv1

    if episode_advantage_w != 0.0:
        episode_advantages = episode_norm_reward(
            token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
        scores = step_advantage_w * step_advantages + episode_advantage_w * episode_advantages
    else:
        scores = step_advantage_w * step_advantages

    return scores, scores
