# 官方算法与评分入口

从[官方赛题页](https://www.aicomp.cn/tracks/tracks-4/5590.html)阅读赛题说明，从[参赛流程](https://www.aicomp.cn/process)获取竞赛材料。将 `contest-source.tar.gz`、`contest-data.tar.gz`、`open3dbench-contest-runtime.tar.gz` 放在一起。任务包本身的 `contest-source/README.md` 有完整环境和参数说明。

| 官方基线 | 初始思路 | 开放修改 | 算法源文件（相对于 `contest-source/AgenticBench/`） |
|---|---|---|---|
| `loss_decoder`（本队主线） | 用正负样本对比训练节点表示，再做面积加权二路聚类 | `compute_tp_gnn_loss` 与 `decode_tp_gnn_embeddings` | `local_tasks/tp_gnn_loss_decoder_timing/initial_program.py` |
| `binary_loss`（参考） | 直接预测分层概率，损失包含割网、面积平衡与二值约束 | 损失函数；解码阈值和约束修复固定 | `local_tasks/tp_gnn_binary_loss_timing/initial_program.py` |

两条基线的起点在本次使用的任务包中分别为以下 SHA-256，方便确认队友拿到的是同一份代码。若官方发了新包，先核对更新再决定是否重新建立对照。

```text
loss_decoder f18b6e303cad3e67b7dd5be6a63a2ca7399e43997fd0c900334adae1a31bc87b
binary_loss  53c784afca8f4d5960b1ac15cb0cc90a251cdfd9d90af890ae30193806f35e7a
```

- 训练：`bp_fe-ng45`、`bp-ng45`、`ariane133-ng45`，每例种子 `1001/1002/1003`。完整九组才能作为训练总分来比较。
- 公开验证：`swerv_wrapper-ng45`、`bp_fe-asap7`，共六组，在候选选定后运行。
- 大会最终评测还加两个隐藏算例。九组训练成绩只能说明选候选时的表现。
- 时序指标看 WNS 和 TNS，分别反映最差路径和累计时序违例。在对比父子候选时，按同算例、同种子看两项，不能直接比较两个不同的部分矩阵均分。

切分之后的布局、合法化、供电网络和时序优化使用官方评估流程。正式候选由 Harness 从官方起点自动生成，并通过官方接口计次、计费。
