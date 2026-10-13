# 运行、日志与排查

## 运行前

1. 核对两个可见CUDA设备及NCCL；设置Java 11，安装spaCy语言模型。
2. 商品JSON的SHA256与 `scripts/irgo/common.py` 一致；Lucene `indexes` 含相同1000个ASIN。
3. 模型是历史相同初始权重或明确标记的新实验模型。完整训练从初始模型开始，不接着smoke权重训练。
4. 准备新的输出目录；默认完整训练保留最近一个actor检查点。需要更多阶段时提前调整并记录。
5. 在兼容环境运行CPU测试，随后两轮GPU smoke；核验 `exitcode=0`、损失有限和检查点文件完整。

`train.py` 与 `evaluate.py` 默认只显示计划，带 `--execute` 才真正运行。训练日志写入输出目录下的 `train.log`，另一个终端可用 `tail -f` 查看。

## 输出结构

```text
runs/<experiment>/
  config.yaml           本次实际训练配置
  launch.json           初始模型一致性、数据哈希和元数据行数
  train.log             训练进程输出
  exitcode              子进程结束状态
  wandb/                当前运行的本地W&B文件
  run/
    global_step_*/actor/ FSDP模型、优化器和状态分片
    u_evidence/
      run_manifest.json
      stage_000001/
        natural.json
        selection_before_outcomes.json
        probes.json
        interventions.json
        intervention_stops.json
        auxiliary_data.json
        metrics.json
        actor_input.pt.gz
        final_advantages.pt.gz
        complete.json
```

文件名以实际控制器输出为准；无候选/无成功的阶段可能没有相应数据内容。大张量训练后无损压缩并保存校验信息。不要把这些本地轨迹和模型加入Git。

## 应重点观察的指标

| 指标族 | 解释 |
|---|---|
| `val/success_rate` | 任务成功率；不是连续task score |
| `u/triggered_tasks`、`u/served_tasks` | 有干预需求与实际获得干预的任务覆盖 |
| `u/probes`、每任务分支分布 | 实际采样条数，区分预算上限与用量 |
| `u/extra_physical_env_calls` | 包括前缀重放的额外交互成本 |
| `u/extra_generated_tokens`、`u/query_*` | 分支和搜索提议的模型生成开销 |
| `prob/final_token_advantage_changed_gt_001` | 真实最终优势变化，不等于变化更正确 |
| `u/all_failed_rescued_tasks` | 初始全失败组找到真实成功分支的次数 |
| `aux/tasks`、`aux/completed_optimizer_steps` | 标签覆盖及辅助训练是否实际执行 |
| `aux/critical_nll_before/after` | 关键响应局部学习迹象，不替代无提示任务成功率 |
| `history/*` | 历史影子审计；不表示数据已用于训练 |

## 常见问题

- **显存不足**：本配置使用每卡PPO microbatch=1，已关闭相关torch compile。先检查实际长度、推理缓存和其他GPU进程，不通过缩短任务步数静默改变实验。
- **主机内存不足**：FSDP模型/优化器检查点也消耗主机内存。历史约64GB可运行，不保证任意低内存机器都可行。
- **恢复分支不一致**：程序会对初始页面、完整prompt、上下文哈希和替代动作投影报错。不要关闭断言继续训练。
- **gate通过率低**：这是条件稳定性代理；证据数少、单一候选或策略质量不完整都会失败。未通过不意味着本轮证据被丢弃。
- **成功分支未改原优势**：可能是强制动作的经验策略质量为零；检查辅助标签是否进入同模型NLL，而不是增加伪造策略计数。
- **验证看起来偏高/偏低**：确认同一商品版本、text_rich、步数、温度、数据量及模型来源。训练内128次与独立1536次验证分开比较。
- **更换初始模型**：`--allow-different-model` 会记录新起点；不能沿用原始实验的公平对照声明。

## 独立验证

逐种子的日志与退出码独立保留。`results.json` 在每个种子完成后更新，只有三个种子均正常结束且权重未变时才写 `complete=true`。默认解码设置中种子含义与历史一致：更换的是环境seed，并非三次独立训练。

发布验证检查能核对计数和进程结束，不证明任务清单未参与调参。进行论文实验时还应固定独立测试清单、保存逐任务结果，并明确训练随机种子与环境随机种子。
