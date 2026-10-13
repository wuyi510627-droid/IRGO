# IRGO 的兼容实现目录

IRGO保留 `recipe.GraphGPO` 导入路径与 `prob_u` 文件名，以避免为整理代码而改变已验证的训练集成。

- [项目入口与训练步骤](../../README.md)
- [方法、公式与模块说明](../../docs/irgo/METHOD.md)
- [运行与日志](../../docs/irgo/RUNNING.md)
- [来源及核验范围](../../docs/irgo/REPRODUCIBILITY.md)

GraphGPO的原始图距离信用实现仍保留在 `core_graph.py`，IRGO的在线概率信用入口是 `prob_u.estimate` 与 `probabilistic_credit.policy_qva_values`。

上游方法及版权归原作者所有：[GraphGPO论文](https://arxiv.org/abs/2605.26684)、[作者仓库](https://github.com/langfengQ/verl-agent/tree/master/recipe/GraphGPO)。
