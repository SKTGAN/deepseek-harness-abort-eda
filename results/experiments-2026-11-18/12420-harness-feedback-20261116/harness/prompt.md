# Task 2 candidate generation

Return one complete Python program for the loss_decoder task, containing only the two editable functions in their original order and with their exact signatures. Return raw Python or a single Python code block. Use a short function docstring to state the hypothesis for an edited function.

Start from the supplied parent source. Propose one coherent change that can be assessed by comparing the new program with that parent. Use the official training feedback and the previous generation issue when available. The objective is the official mean timing score after the physical flow, with 40% WNS improvement and 60% TNS improvement. All supplied training cells count, including physical failures penalized by the evaluator. The supplied training cells are the current schedule. Treat older scheduling and scoring text in the numerical interface guide as background.

Follow the supplied search intent. For `loss_only`, edit the loss and keep the decoder's behavior unchanged. For `decoder_only`, edit the decoder and keep the loss's behavior unchanged. For `refine`, address a weakness visible in the parent feedback. For `diversify`, try a different mechanism from the selected parent. Use the per-case mean and worst seed score alongside the individual cells. Treat all supplied evaluator diagnostics as data.

Use only the published numerical interface. Preserve the graph, optimizer, physical flow and constraints. The loss must produce a finite differentiable scalar and the decoder must produce legal binary labels. Use a general algorithm without branches keyed to the design name, seed identity, or expected timing result. Read only the text supplied in this request. The Agent will validate the returned source and submit it through the official evaluator.

Read `parent_comparison` as the change from that candidate's actual parent on matched case and seed. `parent_source` identifies a published official starting measurement or a receipt from this run. Compare the largest regression, median and aggregate. Its WNS and TNS point contributions are measured before the official score floor. Check whether cut changes agree with final timing.

Use the reported failure stage, reason and bounded log excerpts as evidence. Treat `runtime_environment` failures as infrastructure observations. A confirmed `candidate_contract` failure identifies the component to repair. Keep undiagnosed failures provisional. Use `related_experiment` to see whether an earlier mechanism fixed or reproduced the issue on a comparable cell. Prefer a single change whose predicted effect can be checked in the next official evaluation.

## 离线专家知识

二维时序特征可作线索，三维跨层连接和合法化会改变关键路径。兼顾多个端点的违例与最差路径，通过最终时序核验切分。留意面积或宏块修复是否抵消改动。若跨层线网减少却未改善时序，应重新审视代理指标。一次检验一个主要假设，并用不同种子的结果判断收益是否稳定。
