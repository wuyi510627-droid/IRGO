"""
训练期埋点：统计 GraphGPO 每个训练步里「死图」与「零信号」的占比。

这是复现 GraphGPO 的核心目的——**不是看它涨多少分，是看它浪费了多少数据**。
离线诊断已在一个 7B 收敛快照上测过（死图 38% 的 task / 48% 的步），但那是
一个静态快照；训练过程中成功率从 ~5% 涨到 60%+，死图占比理应大幅变化。
这个曲线是离线数据拿不到的。

埋点位置：advantage 算完之后，不改任何训练逻辑，纯读。

统计口径（与离线诊断严格一致，便于对照）：

    死图        该 task 的 rollout 组里没有任何一条成功轨迹
                -> 图中无 'success' 节点 -> Dijkstra 无源点
                -> 所有节点被兜底成 d=1（core_graph.py:325-332）
                -> R^G 恒为 10·γ¹ -> 组内方差 0 -> A^G ≡ 0，且不报错

    A^G = 0     步级优势为 0，拿不到 step 级梯度
                两个成因分开统计：
                  单例组      该状态在本 task 只出现过一次（组内无人可比）
                  后继 d 全同  组内多人，但后继离目标一样远（比不出高下）

    A^ep = 0    该 task 组内所有轨迹回报相同（全成功 或 全失败）

    总优势 = 0  A^G 和 A^ep 同时为 0 —— **这些步对策略更新完全没有贡献**

用法（在 trainer 里 advantage 算完后调一次）：

    from recipe.GraphGPO.graph_stats import log_graph_stats
    metrics.update(log_graph_stats(batch, Gs, global_step, out_dir))
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np

SUCCESS = 'success'


def _to_np(x):
    return x.detach().float().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def compute_graph_stats(batch, Gs=None, eps=1e-8):
    """
    从一个训练 batch 里算出死图/零信号的各项占比。

    batch 需要有：
      non_tensor_batch: uid, traj_uid, anchor_obs, next_obs, episode_rewards
      batch: advantages（或 step 级优势）—— 有则用，无则按图重算
    Gs: {uid: ComplexGraph}，可选；给了就能直接读 d(s)
    """
    nb = batch.non_tensor_batch
    uid = np.asarray(nb['uid'])
    traj = np.asarray(nb['traj_uid'])
    anchor = np.asarray(nb['anchor_obs'])
    nxt = np.asarray(nb['next_obs'])
    ep_r = _to_np(nb['episode_rewards']).astype(float)

    n = len(uid)
    st = dict(n_steps=n, n_tasks=len(set(uid.tolist())))

    # ── 每个 task：是死图吗？A^ep 退化吗？ ──────────────────────
    dead_task, ep_zero_task = set(), set()
    for u in set(uid.tolist()):
        m = uid == u
        # 死图判据：该组没有任何一步的后继被写成 'success'
        # （rollout_loop.py:288 只在 episode_rewards>0 时写入）
        if not np.any(nxt[m] == SUCCESS):
            dead_task.add(u)
        # A^ep 退化：组内轨迹回报全相同（全成功 或 全失败）
        rr = ep_r[m]
        if rr.size and np.all(np.abs(rr - rr[0]) < eps):
            ep_zero_task.add(u)

    dead_mask = np.array([u in dead_task for u in uid])
    epz_mask = np.array([u in ep_zero_task for u in uid])
    st['dead_graph_task_ratio'] = len(dead_task) / max(st['n_tasks'], 1)
    st['dead_graph_step_ratio'] = float(dead_mask.mean())
    st['ep_adv_zero_step_ratio'] = float(epz_mask.mean())

    # ── 每一步：A^G 是否为 0，以及成因 ─────────────────────────
    # 分组口径与 build_step_group 一致：同 uid 内 anchor_obs 相同的所有步
    grp = defaultdict(list)
    for i in range(n):
        grp[(uid[i], anchor[i])].append(i)

    # ⚠️ 必须拿到 Gs 才能正确判定。A^G=0 的充要条件是「组内后继的 **d** 全同」，
    # 不是「后继状态全同」——不同的后继完全可能有相同的 d（Type C 失效）。
    # 实测在真实数据上：按 d 判 67.8%，按后继状态判只有 28.4%，
    # **少报 39 个百分点**。所以没有 Gs 时宁可报错，不能猜。
    if Gs is None:
        raise ValueError(
            'graph_stats 需要 Gs（{uid: ComplexGraph}）才能判定 A^G=0。'
            '没有它只能比后继状态，会漏掉「不同后继但 d 相同」这一大类失效，'
            '实测少报约 39 个百分点。')

    agz = np.zeros(n, dtype=bool)
    single = np.zeros(n, dtype=bool)
    ndv = np.zeros(n, dtype=float)
    for (u, _), idx in grp.items():
        if len(idx) == 1:
            agz[idx[0]] = single[idx[0]] = True
            ndv[idx[0]] = 1
            continue
        g = Gs.get(u)
        sp = getattr(g, 'shortest_path_to_final', None) if g is not None else None
        if not sp:
            raise ValueError(f'uid={u} 的图没有 shortest_path_to_final，'
                             '需要先调用 Calculate_all_shortest_path')
        dist = sp[0]
        ds = [dist.get(nxt[i]) for i in idx]
        k = len(set(map(str, ds)))
        for i in idx:
            ndv[i] = k
        if k == 1:                      # 组内后继 d 全同 => R^G 全同 => A^G=0
            for i in idx:
                agz[i] = True

    st['agz_step_ratio'] = float(agz.mean())
    st['agz_from_singleton'] = float(single.mean())
    st['agz_from_same_d'] = float((agz & ~single).mean())
    # 按**步**加权，与离线诊断口径一致。按组平均会得到完全不同的数
    # （实测 5.7 vs 23.4），因为热点状态被访问很多次。
    st['group_size_mean'] = float(np.mean([len(grp[(uid[i], anchor[i])])
                                           for i in range(n)]))
    st['succ_d_values_mean'] = float(ndv.mean())

    # ── 总优势为 0：A^G=0 且 A^ep=0，这些步完全没贡献 ──────────
    st['total_adv_zero_ratio'] = float((agz & epz_mask).mean())
    # 活图口径——方法真正能改善的部分
    live = ~dead_mask
    st['agz_step_ratio_live'] = float(agz[live].mean()) if live.any() else 0.0

    # ── 轨迹成功率（对照）─────────────────────────────────────
    seen, succ, tot = set(), 0, 0
    for i in range(n):
        k = (uid[i], traj[i])
        if k in seen:
            continue
        seen.add(k)
        tot += 1
        succ += int(ep_r[i] > 0)
    st['traj_success_rate'] = succ / max(tot, 1)
    return st


def log_graph_stats(batch, Gs=None, step=None, out_dir=None, prefix='graph/'):
    """算统计并落盘一行 jsonl；返回带前缀的 dict 供 wandb 记录。"""
    st = compute_graph_stats(batch, Gs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'graph_stats.jsonl'), 'a') as f:
            f.write(json.dumps(dict(step=step, **st)) + '\n')
    if step is not None and step % 5 == 0:
        print(f"[graph_stats] step={step} "
              f"死图 task {st['dead_graph_task_ratio']:.1%} / 步 {st['dead_graph_step_ratio']:.1%} | "
              f"A^G=0 {st['agz_step_ratio']:.1%}（活图 {st['agz_step_ratio_live']:.1%}）| "
              f"总优势=0 {st['total_adv_zero_ratio']:.1%} | "
              f"成功率 {st['traj_success_rate']:.1%}", flush=True)
    return {prefix + k: v for k, v in st.items()}
