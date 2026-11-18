# Harness 实验归档

每条记录对应一次运行。先在下表选运行，打开各轮的 round.json 可看到候选 ID、实际父候选、提交决策、逐算例和种子成绩。candidate.py 保存这次生成的代码，events.jsonl 按实际顺序保留了可见的模型请求、回复、候选筛选和反馈。round.json 的 event_sequences 对应 events.jsonl 的 sequence。

[全部候选轮次（CSV）](rounds.csv) · [逐算例和种子（CSV）](cells.csv) · [运行索引（JSON）](index.json)

| 运行 | 评估范围 | 有账本或回执的候选数 | 完整成绩（按轮次） | 请求 / 回复正文数 |
|---|---|---:|---|---:|
| [10820-branching](10820-branching/README.md) | 正式训练 · 九组 | 5 | 12.10 → 23.08 → 25.35 → 26.93 → 7/9 组有数据 | 10 / 10 |
| [10820-evidence](10820-evidence/README.md) | 正式训练 · 九组 | 4 | 4.55 → 14.11 → 26.49 → 25.01 | 11 / 11 |
| [15332-claude-original](15332-claude-original/README.md) | 正式训练 · 九组 | 10 | 11.94 → 29.43 → 24.87 → 34.75 → 25.12 → 34.75 → 18.69 → 22.39 → 28.41 → 10.55 | 0 / 0 |
| [15332-claude-v3](15332-claude-v3/README.md) | 正式训练 · 九组 | 5 | 24.04 → 0.83 → 24.48 → 29.67 → 3/9 组有数据 | 13 / 13 |
| [27528-previous](27528-previous/README.md) | 正式训练 · 九组 | 5 | 5.48 → 18.22 → 9.96 → 1.12 → 3/9 组有数据 | 14 / 14 |
| [27528-claude-diagnostic](27528-claude-diagnostic/README.md) | 三组开发小测 | 3 | 1.78 → -3.89 → 30.98 | 6 / 6 |
| [12420-harness](12420-harness/README.md) | 早期训练快照 | 1 | 7/9 组有数据 | 0 / 0 |
| [12420-harness-v5-20261116](12420-harness-v5-20261116/README.md) | 早期训练快照 | 1 | 6/9 组有数据 | 0 / 0 |
| [12420-harness-feedback-20261116](12420-harness-feedback-20261116/README.md) | 早期训练快照 | 1 | 6/9 组有数据 | 0 / 0 |
| [12420-harness-opening-20261116](12420-harness-opening-20261116/README.md) | 早期训练快照 | 1 | 3/9 组有数据 | 0 / 0 |
| [14828-harness-v6-20261116](14828-harness-v6-20261116/README.md) | 早期训练快照 | 1 | 8/9 组有数据 | 0 / 0 |
| [14828-harness-next-20261116](14828-harness-next-20261116/README.md) | 早期训练快照 | 1 | 7/9 组有数据 | 0 / 0 |
| [dual-arms-ours](dual-arms-ours/README.md) | 双卡机早期尝试，部分只发生了模型调用 | 0 | — | 0 / 0 |
| [dual-arms-claude](dual-arms-claude/README.md) | 双卡机早期尝试，部分只发生了模型调用 | 0 | — | 0 / 0 |
| [dual-arms-contract-ours](dual-arms-contract-ours/README.md) | 双卡机早期尝试，部分只发生了模型调用 | 0 | — | 0 / 0 |
| [dual-arms-contract-claude](dual-arms-contract-claude/README.md) | 双卡机早期尝试，部分只发生了模型调用 | 0 | — | 0 / 0 |
| [dual-arms-low-ours](dual-arms-low-ours/README.md) | 双卡机早期尝试，部分只发生了模型调用 | 5 | 0/9 组有数据 → 0/9 组有数据 → 0/9 组有数据 → 0/9 组有数据 → 0/9 组有数据 | 0 / 0 |
| [dual-arms-low-claude](dual-arms-low-claude/README.md) | 双卡机早期尝试，部分只发生了模型调用 | 1 | 0/9 组有数据 | 0 / 0 |
| [27528-claude-formal-2r](27528-claude-formal-2r/README.md) | 双轮正式训练 | 2 | 1.79 → -0.22 | 5 / 5 |

## 数据口径

- 九组训练由三个算例各三个种子组成。完整矩阵先取每个算例的种子均分，再等权平均。三组开发小测和六组公开验证分别列出。
- 在途或中断轮次保留已有的算例结果，完整矩阵的分数留空。标记 native_cell_rescored 的分项按官方评分函数从已完成的原生时序数据计算；完整矩阵取评估器回执。早期快照的 corrected_score 来自当时导出的逐组对照。
- 每个有源码的运行保存对应的 Agent 和提示词。复现时将该目录的文件放到官方 Agent 目录，按仓库复现手册从官方任务包恢复起始 program.py。各轮的 candidate.py 用于查看和复核生成结果。
- events.jsonl 含当时实际记录的交互，包括保存下来的假设、理由、被拒方案和模型可见反馈。早期版本记录粒度不同；缺失的交互和分数在各运行页标明。双卡机早期汇总有计量记录和部分候选，但没有逐项时序数据。
- 此次为按运行时间分别采集的静态快照。口令、API 凭据和数据包未收录。
