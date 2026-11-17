# 复现手册

## 先跑离线回放

用 Python 3.10+，在仓库根目录执行：

```bash
python3 -m pip install numpy
python3 -B -m unittest discover -s harness/claude/tests -v
```

回放覆盖候选搜索、失败分类与选择。没有 NumPy 时解码探针测试会跳过。这一步不需要 Docker、GPU 或 DeepSeek API，也不代表拿到了相同的物理分数。

## 再在独立目录做真实评估

1. 在 Linux + NVIDIA GPU 主机准备 Docker、官方任务包和 Python `PyYAML`。按来自官方的 `contest-source/README.md` 解压源码与数据、加载镜像。
2. 在一份**新解压**的 `contest-source` 安装本仓库的 Harness。安装脚本从该任务包恢复 `program.py`，不要拿历史候选覆盖它。
3. 检查环境，然后用空的 `--work-root` 开新实验。同一实验恢复时使用原目录和原配置。API key 由运行者经 `DEEPSEEK_API_KEY_FILE` 设置，密钥文件放在代码、数据和工作目录之外。

```bash
python3 scripts/install_harness.py /path/to/contest-source
cd /path/to/contest-source/AgenticBench
python3 -m contest2 check --baseline loss_decoder
python3 -m contest2 run --baseline loss_decoder \
  --work-root /path/to/contest-work/claude-handoff-01 \
  --retain-physical-files
```

`check` 无误后再开始收费的演化。训练每个候选一轮，完整九组；单次运行按当前任务包由外层计量候选次数和 API 费用。两套 Harness 做对照时分开工作目录，保持官方包、GPU 类型和固定种子一致。若要保存完整布局文件，先给工作盘留足空间再打开保留开关。

## 归档赢家，再补验证

从这次运行的 `session-result.json` 读取最终选中的候选 ID。保留同一工作目录里的 `candidates/<id>/candidate.json`、训练矩阵回执、九组结果及日志；单独导出其中的源码。检查 ID 与训练矩阵匹配后，再对同一个候选跑六组公开验证：

```bash
python3 -m contest2 validate --baseline loss_decoder \
  --work-root /path/to/contest-work/claude-handoff-01 \
  --candidate-id "$CANDIDATE_ID" --retain-physical-files
```

验证需要原工作目录中的真实训练记录，不能把另一次运行的候选 ID 直接代进去。验证结果用于报告泛化表现，不反馈进这次正式搜索。整理交付时保存原始回执、布局、指标与时序报告，以及对应的 Harness 版本和候选源文件。
