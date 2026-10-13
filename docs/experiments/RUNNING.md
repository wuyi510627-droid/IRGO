# 固定任务协议与采样对照

当前优先实验：**F6，每任务6条自然＋原IRGO干预**。N10、R8暂缓，不自动接续。
F6仍使用原全失败救援和成功分支辅助学习规则；全失败资格根据该任务全部6条自然轨迹判断。
没有真实成功分支时，不构造正奖励或成功标签。

此分支是后续实验版本，尚未取得新协议上的模型结果。历史150轮结果属于
`eefaf39004be92f51c7ba316d0c871a913542750`，复现历史版本请检出该提交。
配置目标与比较边界见 [WEEK1_SPEC.md](WEEK1_SPEC.md)。

## 已完成和仍待完成

- 完整150轮证据已审计：2400任务组、19200自然轨迹、4800干预分支。
- 323032次物理环境调用中，39384次来自干预；8499次为重放前缀。
- 自然生成19603845个输出token；干预额外3719246个输出token，**已经包含**
  158154个查询提议token，不得再相加。原记录没有完整生成输入token。
- 150轮步耗时合计174524.31秒；其中干预阶段64784.01秒。两者是包含关系。
  轮耗时合计不包含全部启动/收尾时间，也不等同于纯优化器耗时。
- 训练实际出现2017个结构任务、2213条不同指令。所有2400次目标重建均与
  保存的训练指令（忽略随种子变化的价格）一致。
- 旧在线验证任务未保留逐任务轨迹；按原种子/重置顺序重建，第150轮128次验证
  中35次与此前训练存在结构任务重合。这是**重建且忽略价格**的重合，不能称为
  35条完全相同指令，更不能直接套到最终三个种子512次的独立验证。
- 新目录6910目标按结构身份及标准化指令的连通组分开：训练5539、开发708、
  保留测试663。共享1000商品库；不声称商品泛化。每个种子固定150×16任务，
  在线开发每次使用同一128目标，最终512测试目标仅列清单，尚未评测。
- GPU短程检查和新150轮训练尚未执行；2026-10-13新实例尚未挂载GPU，内存约2GB。

可复算的完整汇总为 [historical_cost_audit.json](../../results/historical_cost_audit.json)。
原始任务指令、私有轨迹、模型与运行清单保存在实验目录，不进入Git。

## 导出与冻结协议

在已有WebShop环境中导出实际目录（使用原1000商品数据及已建Lucene索引）：

```bash
python scripts/experiments/export_catalog.py --repo "$PWD" --output /data/irgo/catalog.json
python scripts/experiments/prepare_protocol.py --catalog /data/irgo/catalog.json --output /data/irgo/protocol
python scripts/irgo/prepare_data.py --output /data/irgo/metadata --train-rows 16 --val-rows 128
python scripts/irgo/prepare_data.py --output /data/irgo/smoke_metadata --train-rows 16 --val-rows 8
```

`prepare_protocol.py`不读取模型结果。三种训练seed共享划分、开发顺序和测试顺序，
训练任务序列分别固定。各方法同seed使用同一清单。运行环境会核对实际目标目录
指纹和每个身份映射，并保存清单副本。禁止恢复中断长训：当前入口明确`resume_mode=disable`，
不能将任务序列从第一轮重新开始却接着旧权重训练。

## 启动顺序

先完成依赖安装、WebShop索引和初始权重核验（见原运行说明）。W&B由已有登录
提供凭据，不把key写进代码；新对照入口默认online，可明确指定offline。

```bash
# 不加--execute，只显示待运行配置。
python scripts/experiments/train_control.py --arm F6 --model /data/base_model --data-dir /data/irgo/smoke_metadata --manifest /data/irgo/protocol/seed0.json --output /data/irgo/F6_smoke --profile smoke --execute
python scripts/experiments/train_control.py --arm F6 --model /data/base_model --data-dir /data/irgo/metadata --manifest /data/irgo/protocol/seed0.json --output /data/irgo/F6_150 --smoke-output /data/irgo/F6_smoke --execute
```

当前只优先执行F6。后续经安排运行P/N/R/F及其他seed时，也必须重新通过
对应短程检查。短程保留完整16×6（F6）、16×8或16×10自然batch、实际分支预算，执行2轮训练
并验证开发重置路径。新入口要求metadata恰好一批，避免验证多批反复重置成同一任务。
运行源码、种子、清单、初始权重不匹配，或短程退出失败/跳过优化器更新时拒绝长训。
旧`train.py`入口不接受继承的`IRGO_TASK_MANIFEST`，避免默默改变历史任务协议。

## 观测和比较

P=8自然；N=10自然；R=8自然+随机合格位置；F=8自然+原启发式位置；
F6=6自然+原启发式位置。F6保持每轮32分支、每任务4分支、额外480调用限制。
P/N仍运行信用与历史复用观察器，但没有查询提议、分支调用、成功分支辅助损失。
历史轨迹继续仅供影子诊断；不会进入当前信用或策略梯度。

R只随机合格上下文的顺序，保留任务优先级、候选动作生成/顺序、每任务两位置、
停止条件和辅助损失。它测的是已有合格池内选点排序的作用；不是所有状态均匀
抽样、也不是完全移除WebShop条件检查。不同位置的结果和提前停止仍会造成实际
执行条数和调用数不同，应如实记录。

每轮记录：自然及干预实际调用、有效步、重放前缀、生成输入/输出token（含补齐
请求）、查询提议、PPO训练行与token、优化器尝试/完成次数、辅助标签计划token、
初始前6条全失败数、N全部10条全失败数、干预救回、最终优势变化、历史只读覆盖。
`tasks/initial_all_failed_*`统一使用前6条，`tasks/all_natural_all_failed_*`报告全部自然样本；
失败轨迹优势诊断只统计前6条全失败组内的前6条轨迹。实际信用估计和辅助标签资格仍以
全部当前自然证据为准；观测口径不改变学习规则。

PPO token是训练batch在各epoch的暴露量；辅助token是计划标签暴露量，不含所有
前向审计/重算，二者都不是总GPU FLOPs。墙钟耗时同时报告。

F6与P每轮调用上限1920；N/R/F每轮调用上限2400，只能称为同上限；历史F实际均值2153.55。论文应同时给
轮数、实际累计调用、生成token和耗时轴。四组新协议结果必须重新训练，不能用
历史P/F填表；不得凭单次训练或同上限直接宣称严格等成本胜出。

## 历史审计复算

```bash
python scripts/experiments/export_history.py --evidence /archive/run/u_evidence --output /data/irgo/history.json.gz
python scripts/experiments/audit_history.py --history /data/irgo/history.json.gz --catalog /data/irgo/catalog.json --output /data/irgo/audit.json
```

汇总脚本要求150轮均有完成记录，核对自然/干预物理调用账目，拒绝用部分阶段
外推完整成本。验证重合重建仍需遵守文件中的证据范围说明。
