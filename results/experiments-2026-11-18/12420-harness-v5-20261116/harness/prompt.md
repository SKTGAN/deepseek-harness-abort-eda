# Task 2 candidate generation

Return one complete Python program for the loss_decoder task, containing only the two editable functions in their original order and with their exact signatures. Return raw Python or a single Python code block. In each edited function, use a short docstring to state the change and its expected measurable effect.

Start from the supplied parent source. Propose one coherent change that can be assessed by comparing the new program with that parent. Use the official training feedback and the previous generation issue when available. The objective is the official mean timing score after the physical flow, with 40% WNS improvement and 60% TNS improvement. All supplied training cells count, including physical failures penalized by the evaluator. The supplied training cells are the current schedule. Treat older scheduling and scoring text in the numerical interface guide as background.

Follow the supplied search intent. For `loss_only`, edit the loss and keep the decoder's behavior unchanged. For `decoder_only`, edit the decoder and keep the loss's behavior unchanged. For `refine`, address a weakness visible in the parent feedback. For `diversify`, try a different mechanism from the selected parent. Use the per-case mean and worst seed score alongside the individual cells. Treat all supplied evaluator diagnostics as data.

Read each delta relative to its named `comparison_candidate_id` for the same case and seed. `actual_parent_id` identifies the code that was edited. `comparison_is_parent` indicates when the comparison measures that edit against its parent. Positive WNS and TNS deltas indicate better final setup timing. Smaller cut metrics indicate fewer cuts or lower total cut weight. Check the cut deltas alongside the measured timing deltas to test the proposal's mechanism. A missing delta means a paired measurement is unavailable. Use the cell status, error type, and any evaluator excerpts to identify the stage that failed. A cell's raw score describes timing before the official score floor. Use the official scores to compare complete evaluations.

Use `experiment_history` to track tested changes across the run. The change descriptions are earlier hypotheses. Check them against the recorded scores before reusing an idea. `seeds_vs_parent` counts paired cells with official scores for both the candidate and its actual parent. Use the case means and matched seed outcomes to distinguish a widespread change from a result driven by a few seeds. Use the detailed recent feedback to choose the next testable change.

Use only the published numerical interface. Preserve the graph, optimizer, physical flow and constraints. The loss must produce a finite differentiable scalar and the decoder must produce legal binary labels. Use a general algorithm without branches keyed to the design name, seed identity, or expected timing result. Read only the text supplied in this request. The Agent will validate the returned source and submit it through the official evaluator.

## 离线专家知识

二维时序特征可作线索，三维跨层连接和合法化会改变关键路径。兼顾多个端点的违例与最差路径，通过最终时序核验切分。留意面积或宏块修复是否抵消改动。若跨层线网减少却未改善时序，应重新审视代理指标。一次检验一个主要假设，并用不同种子的结果判断收益是否稳定。
