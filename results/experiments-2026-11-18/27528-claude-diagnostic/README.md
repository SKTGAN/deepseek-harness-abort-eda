# 27528-claude-diagnostic

三组开发小测。数据来源或快照时间为 2026-11-18 本地归档。

[本次 Harness 源码和提示词](harness/)

[按时间顺序查看交互和决策](events.jsonl) · 记录到 6 个请求正文和 6 个可见回复正文。

[查看模型调用的计量记录](model-calls.json)

| 轮次 | 已归档组数 | 完整成绩 | 生成代码 | 提交时记录的决策 |
|---:|---:|---:|---|---|
| [1](round-01/round.json) | 3/3 | 1.7780 | [候选程序](round-01/candidate.py) | opening_control |
| [2](round-02/round.json) | 3/3 | -3.8944 | [候选程序](round-02/candidate.py) | test_alternative_component_after_opening |
| [3](round-03/round.json) | 3/3 | 30.9836 | [候选程序](round-03/candidate.py) | orientation_only |
