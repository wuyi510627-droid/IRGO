# IRGO：方法与源码对应

本文描述已完成的150轮方案。旧文件名 `prob_u` 为兼容名称，不等于历史的上一版 Prob+U。

## 1. 信用状态与可执行上下文

信用图的节点为 $z=(\mathrm{task},\mathrm{page})$；同任务内完全相同的页面文本可以共享证据。剩余步数与历史不参与此合并。它是一种状态抽象，不保证严格满足 Markov 条件。

恢复分支的上下文标识为 $c=\operatorname{hash}(\mathrm{goal},\mathrm{prompt\_ids},\mathrm{page},\mathrm{remaining})$。这用于确认恢复了同一决策现场，不用于拆分默认信用节点。

源码：[credit_key/context_key](../../recipe/GraphGPO/prob_u.py)、[TraceCapture/ProbeRunner](../../recipe/GraphGPO/prob_u_runtime.py)。

## 2. 概率信用

令 $N(z,a,z')$ 为本轮自然轨迹与有效分支区间的转移计数。前缀重放不重复计数。平滑只作用于已观察后继集合 $\mathcal S_{z,a}$：

$$\widehat P(z'\mid z,a)=\frac{N(z,a,z')+0.5}{\sum_uN(z,a,u)+0.5|\mathcal S_{z,a}|}.$$

没有凭空给未见的成功后继分配先验质量。经验策略只统计自然动作和分支中非强制的后续动作：

$$\widehat\pi(a\mid z)=\frac{N_{\mathrm{natural}}(z,a)+N_{\mathrm{continuation}}(z,a)}{\sum_b[N_{\mathrm{natural}}(z,b)+N_{\mathrm{continuation}}(z,b)]}.$$

强制动作进入转移证据，但不进入上述经验策略分子。成功终止节点初始值为1，其余未有成功支持的节点为0；有限视野迭代：

$$Q_h(z,a)=\sum_{z'}\widehat P(z'\mid z,a)V_{h-1}(z'),\qquad V_h(z)=\sum_a\widehat\pi(a\mid z)Q_h(z,a).$$

终止成功值在迭代中保留为1。使用 $H=8$，原始步骤信用为 $A_H(z,a)=Q_H(z,a)-V_H(z)$。这是策略评估，不是用最大动作价值回传的最优策略估计。

源码：[estimate](../../recipe/GraphGPO/prob_u.py)、[policy_qva_values](../../recipe/GraphGPO/probabilistic_credit.py)。同文件保留的 `q_values/all_q_values` 是旧离线工具，不是当前在线信用入口。

## 3. 条件稳定性检查

对动作 $a$ 的每次出现，取接下来至多8步是否成功的二值标签 $y$。若共有 $n_a$ 次出现、$s_a$ 次成功，则抽样：

$$p_a^{(m)}\sim\operatorname{Beta}(s_a+0.5,n_a-s_a+0.5),\quad m=1,\ldots,512.$$

$$\widetilde A_a^{(m)}=p_a^{(m)}-\sum_b\widehat\pi(b\mid z)p_b^{(m)}.$$

以图DP给出的点优势为参照，分别统计抽样优势落在正区间 $(0.02,\infty)$、负区间 $(-\infty,-0.02)$ 或近零区间的比例，节点分数取候选动作中的最小值。最低动作计数达到2、策略质量完整、且分数不低于0.95时，记录 `stable_proxy`。

这些是动作出现计数，同轨迹重复出现可能相关；512次抽样没有新增任何环境证据。Beta标签与图DP也不是同一个估计器，因此该指标只是条件稳定性代理，不是经校准的正确率。源码：[gates](../../recipe/GraphGPO/prob_u.py)。

## 4. 选点与分配预算

对已有成功轨迹的组，主要候选来自不稳定节点；对全失败组，不要求先通过上述门槛。位置优先级为：搜索后没有检查通过的可见商品 → 进入静态条件不匹配的商品 → 商品静态条件满足但选项不完整 → 非法动作/观测未变化 → 其他不稳定位置。再按动作支持稀缺程度等排序。

每任务先选最多两个完整上下文，生成最多四个不同替代动作。搜索备选由同一策略一次生成，不循环重试无效JSON；点击备选来自环境合法动作。动作排序使用确定性哈希，不预设某动作优于另一动作。

全轮32条、单任务4条、额外物理环境调用480次是上限。轮流访问任务，前缀恢复也收费。原始全失败组发现真实成功后停止；执行两条后可根据选中节点的稳定性代理停止。停止后释放的预算可供后续任务使用。它不是已证明最优的自适应分配器。

源码：[full_intervention.py](../../recipe/GraphGPO/full_intervention.py)、[搜索提议](../../recipe/GraphGPO/full_runtime.py)、[WebShop内部条件检查](../../agent_system/environments/env_package/webshop/progress_checks.py)。

## 5. 信用进入真实训练

干预后总是用有效新增证据重算原始自然步骤的 $Q/V/A$。即使第二次 gate 未通过，也不丢弃新证据或恢复旧优势。新的原始步骤信用随后经过原有无效动作惩罚、同状态归一化、episode项合成与token mask。

代码不把 raw $Q-V$ 直接称为最终训练优势。`compose_observation` 对同一批数据分别合成干预前后的最终优势，记录有效token的差异。

当新成功只出现在自然轨迹未采取的动作上，该动作可能仍有零经验策略质量，原自然轨迹的优势也可能保持为零。因此新增成功不保证通过原 PPO 行获得学习信号。

源码：[UTrainingBridge](../../recipe/GraphGPO/prob_u_runtime.py)、[训练器](../../recipe/GraphGPO/graphgpo_ray_trainer.py)。

## 6. 成功分支辅助学习

只对本轮初始8条自然轨迹全失败、随后真实成功的任务建立修复标签。每任务采用首个成功分支，保留强制关键响应和非强制后续响应，排除前缀与未执行的生成尾部。

以响应为单位，每组辅助槽位是6条普通成功响应、1条关键响应、1条自然后续响应。关键动作直接终止成功时允许没有后续。普通成功池为空时保留零权重槽位，不把其质量重新分给修复样本。

$$\mathcal L_{\mathrm{aux}}=\frac{1}{M}\sum_{j=1}^{M}w_j\left[-\frac{1}{|r_j|}\sum_k\log\pi_\theta(r_{j,k}\mid x_j,r_{j,<k})\right],\quad \mathcal L=\mathcal L_{\mathrm{PPO}}+0.1\mathcal L_{\mathrm{aux}}.$$

其中空槽位 $w_j=0$，有效槽位 $w_j=1$；实现按各卡分配的响应槽位归一化并累计梯度。没有新增子模型，也不把强制动作伪装成自然 PPO 采样。没有救回的任务就没有此类辅助标签。该目标不保证成功率单调提高。

源码：[标签构造](../../recipe/GraphGPO/full_runtime.py)、[调度与辅助反传](../../recipe/GraphGPO/success_branch_loss.py)、[actor梯度接入](../../verl/workers/actor/dp_actor.py)。

## 7. 历史数据与已知限制

最近10轮、最多1280条自然轨迹仅供 `history_shadow` 比较匹配和预测误差，不进入当前策略信用或actor损失。要研究历史复用，需另立实验，不能只修改README就声称已实现。

内部条件检查依赖WebShop目标与奖励分解；全失败干预仍失败时，本方案没有自动创造可靠正信用。当前证据也未隔离采样数量、选点和辅助学习各自的因果贡献。论文与后续实验需如实披露。
