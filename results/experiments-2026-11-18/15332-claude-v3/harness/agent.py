"""Search the task-2 loss and decoder through the metered contest bridge."""

import ast
import hashlib
import json
from pathlib import Path
from statistics import mean
import time
import traceback
from uuid import uuid4

from candidate_reply import generated_program, model_text
from contest2.agent_api import AgentAPIError
from contest2.model_pricing import MODEL_ID
from contest2.scoring import reference_table, score_metrics
from local_tasks.tp_gnn_local_task_definitions import DECODER_TASK, validate_program
from paired_feedback import (REPAIR_CHAIN, assignment_difference, assignment_snapshot, comparison_for_prompt,
                             failure_evidence, load_attempts, published_partition_changes, related_experiment)


EXPERIENCE_HEADING = "## Offline Expert Knowledge"
UNCHANGED_FUNCTION = {"loss_only": 1, "decoder_only": 0}
# The loss control goes first: it is the opening trial that moves the partition on every cell.
# A decoder control that reproduces the start grouping measures die orientation and this host.
OPENING = ("loss_only", "decoder_only")
SWAP_FRACTION = 0.5          # more than half of the instances changed: the start grouping with dies exchanged
KEPT_UP_TO_SWAP = 1e-4       # changed-instance fraction, after undoing a die exchange, still counted as the start grouping
SINGLE_CELL_SHARE = 0.75     # a gain is narrow when one case/seed cell carries this share of it
REPLAN_AFTER_INVALID = 3     # consecutive rejected programs before the cached plan is discarded
STOP_AFTER_INVALID = 6       # consecutive rejected programs before search stops with a scored candidate
STOP_WITHOUT_CANDIDATE = 12
BRIDGE_ATTEMPTS = 6
ROBUST_MIN_CASES = 2         # final selection: cases, and seeds, whose mean gain over the start must be positive
CONTAINER_PYTHON = (3, 8)    # the agent and evaluator image runs Python 3.8
PLAN_TASK = """Return a short experiment plan, not code. Use at most 90 words.
State the exact parent decision to change, one plausible mechanism that reaches the legal labels,
the measured evidence (or why evidence is missing), the expected WNS or TNS effect, and a falsifier.
For the opening, compare two mechanisms internally and select one. On subsequent rounds, refer to a
specific paired case/seed result if available. Treat timing changes with unchanged labels as confounded.
Use only the published interface. Do not key the algorithm to case or seed identities.
Field notes. `start_reference` counts start cells measured on this run's host; the others are published
values from another host. `die_swap_vs_initial` marks a cell holding the start grouping with dies exchanged.
`decision.reason` classifies the latest paired comparison: `broad_gain`, `narrow_gain` (one cell or one
case carries it), `orientation_only` and `partition_mostly_unchanged` (start grouping kept, dies possibly
exchanged), `broad_regression`, `duplicate_outcome` (every cell repeats an earlier candidate's labels and
timing). For an added loss term, the change in `tp_gnn_final_loss` from the parent approximates the term's
size. Per cell, `raw_cut_net_count_delta` and `balanced_cut_net_count_delta` show what the decoder changed
before repair; `feasible_cut_net_count_delta` and `assignment_vs_parent` show what reached the legal labels.
A raw change with no feasible change was undone by repair."""
TIMING_KEYS = {
    "final_global_wns_setup_ns", "final_global_tns_setup_ns", "cut_net_count", "timing_weighted_cut",
    "wns_deficit_improvement", "tns_deficit_improvement", "tp_gnn_repair_moved_area_fraction",
    "tp_gnn_positive_pair_cosine_median", "tp_gnn_negative_pair_cosine_median",
    "tp_gnn_embedding_total_variance", "tp_gnn_final_loss", *REPAIR_CHAIN,
}


def compact_feedback(row):
    if not row:
        return row
    copy = dict(row)
    copy["cells"] = [
        {**cell, "metrics": {key: value for key, value in (cell.get("metrics") or {}).items()
                             if key in TIMING_KEYS}}
        for cell in row["cells"]
    ]
    return copy


def function_keys(source, compare_structure=False):
    functions = ast.parse(source).body
    for function in functions:
        if ast.get_docstring(function, clean=False) is not None:
            function.body.pop(0)
        if compare_structure:
            nodes = list(ast.walk(function))
            assigned = {node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
            selectors = {id(index) for node in nodes if isinstance(node, ast.Subscript)
                         for index in ast.walk(node.slice) if isinstance(index, ast.Constant)}
            names = {}
            for node in nodes:
                if isinstance(node, ast.Constant) and type(node.value) in (int, float) and id(node) not in selectors:
                    node.value = 0
                elif isinstance(node, ast.Name) and node.id in assigned:
                    node.id = names.setdefault(node.id, f"__local_{len(names)}")
    return tuple(ast.dump(function, include_attributes=False) for function in functions)


def ask_model(context, instructions, request, effort="low", purpose="generation"):
    payload = {
        "model": MODEL_ID,
        "stream": False,
        "reasoning": {"effort": effort},
        "input": [
            {"type": "message", "role": "system", "content": instructions},
            {"type": "message", "role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ],
    }
    folder = Path(__file__).resolve().parent
    call_id = uuid4().hex
    write_event(folder, "model_request", call_id=call_id, purpose=purpose, payload=payload)
    try:
        response = context.model.responses_create(payload)
    except AgentAPIError:
        write_event(folder, "model_response", call_id=call_id, purpose=purpose, status="gateway_error")
        raise
    fields = {"status": "invalid_response"}
    if isinstance(response, dict):
        fields["status"] = response.get("status")
        usage = response.get("usage")
        if isinstance(usage, dict):
            fields["usage"] = {key: usage[key] for key in ("input_tokens", "output_tokens") if key in usage}
        try:
            fields["output_text"] = model_text(response)
        except ValueError as error:
            fields["issue"] = str(error)
    write_event(folder, "model_response", call_id=call_id, purpose=purpose, **fields)
    return response


def shortened(request):
    """A smaller request for a retry after an over-long or empty reply."""
    keep = ("search_intent", "decision", "parent_id", "parent_program", "evaluations_remaining",
            "previous_generation_issue", "start_reference", "experiment_plan", "largest_regression")
    smaller = {key: request[key] for key in keep if key in request}
    smaller["prior_trials"] = (request.get("prior_trials") or [])[-3:]
    return smaller


def proposal_keys(source, parent_key, unchanged, seen):
    key = function_keys(source)
    if key in seen:
        raise ValueError("The response repeats an already considered program")
    if unchanged is not None and key[unchanged] != parent_key[unchanged]:
        raise ValueError("Keep the other function unchanged for this paired search direction")
    return key


def candidate_gate(source):
    """Cheap check before a two-hour evaluation: the program must parse as Python 3.8, the container's
    version. Raises ValueError, which returns the reason to the model as a generation issue."""
    try:
        ast.parse(source, feature_version=CONTAINER_PYTHON)
    except SyntaxError as error:
        raise ValueError("The program uses syntax newer than Python 3.8: %s" % error.msg)


def outcome_fingerprint(result):
    """Per-cell labels-and-timing identity: equal fingerprints mean equal official outcomes."""
    if result["status"] != "succeeded":
        return None
    rows = []
    for cell in result["feedback"]["cells"]:
        metrics = cell.get("metrics") or {}
        values = (metrics.get("partition_assignment_changed_count"), metrics.get("cut_net_count"),
                  metrics.get("final_global_wns_setup_ns"), metrics.get("final_global_tns_setup_ns"))
        if any(not isinstance(value, (int, float)) for value in values):
            return None
        rows.append((cell["case"], str(cell["seed"]), values[0], values[1], round(values[2], 9), round(values[3], 6)))
    return tuple(sorted(rows)) or None


def robust_gain(result, start):
    """Gain over the start reference with the single best cell removed, and how widely it is spread."""
    if result["status"] != "succeeded" or start is None:
        return None
    base = {(cell["case"], str(cell["seed"])): cell.get("score") for cell in start["feedback"]["cells"]}
    deltas = {}
    for cell in result["feedback"]["cells"]:
        key = cell["case"], str(cell["seed"])
        if not isinstance(cell.get("score"), (int, float)) or not isinstance(base.get(key), (int, float)):
            return None
        deltas[key] = cell["score"] - base[key]
    if set(deltas) != set(base):
        return None

    def aggregate(values):  # the official mean over cases of the seed mean
        cases = sorted({case for case, _ in values})
        return mean(mean(value for (case, _), value in values.items() if case == name) for name in cases)

    top = max(deltas, key=deltas.get)
    without_top = {**deltas, top: min(0.0, deltas[top])}
    cases = sorted({case for case, _ in deltas})
    seeds = sorted({seed for _, seed in deltas})
    cases_up = sum(mean(value for (case, _), value in deltas.items() if case == name) > 0 for name in cases)
    seeds_up = sum(mean(value for (_, seed), value in deltas.items() if seed == name) > 0 for name in seeds)
    loo = aggregate(without_top)
    return {"gain": aggregate(deltas), "gain_without_best_cell": loo,
            "best_cell": {"case": top[0], "seed": top[1], "score_delta": deltas[top]},
            "improved_cases": cases_up, "cases": len(cases), "improved_seeds": seeds_up, "seeds": len(seeds),
            "eligible": (loo > 0 and cases_up >= min(ROBUST_MIN_CASES, len(cases))
                         and seeds_up >= min(ROBUST_MIN_CASES, len(seeds)))}


def select_final(history, start, best):
    """Training-only final choice: the eligible candidate with the largest gain without its best cell.

    A candidate is eligible when its gain over the start survives removing its single best cell and is
    positive on most cases and most seeds. Without an eligible candidate, the highest official score.
    """
    table = []
    for result in history:
        stats = robust_gain(result, start)
        if stats is not None:
            table.append((result, stats))
    eligible = [(result, stats) for result, stats in table if stats["eligible"]]
    if eligible:
        chosen, stats = max(eligible, key=lambda row: (row[1]["gain_without_best_cell"], score_key(row[0])))
        rule = "largest_gain_without_best_cell"
    else:
        chosen, stats, rule = best, None, "highest_official_score"
    summary = [{"candidate_id": result["candidate_id"], "score": result["feedback"]["score"], **stats}
               for result, stats in table]
    return chosen, {"rule": rule, "candidates": summary}


def score_key(result):
    feedback = result["feedback"]
    cells = feedback["cells"]
    tns = [cell.get("metrics", {}).get("tns_deficit_improvement") for cell in cells]
    complete_tns = bool(tns) and all(value is not None for value in tns)
    return feedback["score"], mean(tns) if complete_tns else float("-inf")


def inert_evidence(feedback):
    comparison = feedback.get("parent_comparison") or {}
    score_delta = comparison.get("official_score_delta")
    if not comparison.get("complete") or comparison.get("candidate_invalid_cells"):
        return None

    # Against the start, the organizer's assignment diagnostic is exact. Keeping the start grouping,
    # with or without exchanging the dies, says nothing about the edited mechanism, whatever the score did.
    partition = feedback.get("partition_changes_vs_initial", {})
    identical = partition.get("identical_assignments", 0)
    kept = partition.get("kept_up_to_swap", identical)
    if (comparison.get("parent_source") == "published_initial_reference"
            and kept > comparison["expected_cells"] // 2):
        reason = "orientation_only" if partition.get("die_swaps") else "partition_mostly_unchanged"
        return {"reason": reason, "identical_assignments": identical, "kept_up_to_swap": kept,
                "die_swaps": partition.get("die_swaps", 0)}
    if score_delta is None or score_delta > 0:
        return None

    unchanged = sum(
        all(cell.get(metric) is not None and abs(cell[metric]) < 1e-6
            for metric in ("wns_slack_delta_ns", "tns_slack_delta_ns"))
        for cell in comparison["cells"]
    )
    if unchanged > comparison["expected_cells"] // 2:
        return {"reason": "timing_mostly_unchanged", "unchanged_pairs": unchanged}
    return None


def case_deltas(comparison):
    """Mean paired score change per case, from the case rows or, failing that, the cells."""
    rows = {row["case"]: row.get("mean_paired_score_delta") for row in comparison.get("cases") or []}
    if not rows:
        cells = {}
        for cell in comparison.get("cells", []):
            if isinstance(cell.get("score_delta"), (int, float)):
                cells.setdefault(cell["case"], []).append(cell["score_delta"])
        rows = {case: mean(values) for case, values in cells.items()}
    return {case: value for case, value in rows.items() if isinstance(value, (int, float))}


def trial_evidence(feedback):
    """What a complete paired comparison says about the edited mechanism; None when unverified."""
    comparison = feedback.get("parent_comparison") or {}
    if feedback.get("duplicate_of"):
        return {"reason": "duplicate_outcome", "same_as": feedback["duplicate_of"]}
    delta = comparison.get("official_score_delta")
    if delta is None or not comparison.get("complete") or comparison.get("candidate_invalid_cells"):
        return None
    inert = inert_evidence(feedback)
    if inert:
        return inert
    cases = case_deltas(comparison)
    per_case = {}
    for cell in comparison.get("cells", []):
        per_case[cell["case"]] = per_case.get(cell["case"], 0) + 1
    scored = [cell for cell in comparison.get("cells", []) if isinstance(cell.get("score_delta"), (int, float))]
    summary = {"improved_cases": sum(value > 0 for value in cases.values()),
               "regressed_cases": sum(value < 0 for value in cases.values()), "cases": len(cases)}
    if delta > 0 and scored:
        # Official aggregate: mean over cases of the seed mean, so one cell weighs 1 / (cases * seeds).
        top = max(scored, key=lambda cell: cell["score_delta"])
        share = top["score_delta"] / (len(per_case) * per_case[top["case"]]) / delta
        summary["largest_cell"] = {key: top.get(key) for key in ("case", "seed", "score_delta")}
        summary["largest_cell_share"] = round(share, 3)
        narrow = share >= SINGLE_CELL_SHARE or 2 * summary["improved_cases"] <= len(cases)
        return {"reason": "narrow_gain" if narrow else "broad_gain", **summary}
    if delta < 0 and 2 * summary["regressed_cases"] > len(cases):
        return {"reason": "broad_regression", **summary}
    return None


def summarize_trial(feedback):
    attempt = feedback.get("attempt") or {}
    comparison = feedback.get("parent_comparison") or {}
    return {
        "candidate_id": feedback["candidate_id"],
        "parent_id": attempt.get("parent_id"),
        "changed_components": attempt.get("changed_components", {}),
        "status": feedback["status"],
        "score": feedback["score"],
        "paired_score_delta": comparison.get("official_score_delta"),
        "evidence": trial_evidence(feedback),
    }


def lineage_head(history, archive):
    """Best candidate reached from the start through broad paired gains only; None means the start.

    Final selection still uses the official score. The lineage only decides which parent the next
    hypothesis builds on, so a gain carried by one cell or by a die exchange is kept as a candidate
    but is not extended.
    """
    members = {}
    for result, summary in zip(history, archive):
        evidence = trial_evidence(summary) if result["status"] == "succeeded" else None
        comparison = summary.get("parent_comparison") or {}
        parent_id = (summary.get("attempt") or {}).get("parent_id")
        if evidence and evidence["reason"] == "broad_gain" and (
                comparison.get("parent_source") == "published_initial_reference" or parent_id in members):
            members[result["candidate_id"]] = result
    return max(members.values(), key=score_key, default=None)


def select_parent(history, best, archive, initial_reference, opening=OPENING):
    if not history:
        return None, opening[0], {"reason": "opening_control"}

    latest, feedback = history[-1], archive[-1]
    failures = [cell["failure"] for cell in feedback.get("cells", []) if "failure" in cell]
    infrastructure = bool(failures) and all(failure["kind"] == "runtime_environment" for failure in failures)
    attempt = feedback.get("attempt") or {}
    if infrastructure and latest.get("source") and not attempt.get("retry_of") and not attempt.get("retry_refused"):
        comparison = feedback.get("parent_comparison") or {}
        retried = any((row.get("attempt") or {}).get("retry_of") == latest["request_id"] for row in archive)
        median = comparison.get("median_paired_score_delta")
        if not retried and (not comparison.get("scored_pairs") or median is None or median >= 0):
            # The hypothesis was not measured. Resubmit the same source once, without a model call.
            return latest, "retry", {"reason": "runtime_environment_retry", "candidate_id": latest["candidate_id"]}

    if len(history) == 1 and latest["status"] == "succeeded":
        evidence = trial_evidence(feedback)
        ambiguous = feedback.get("timing_changes_with_identical_assignment")
        if evidence and evidence["reason"] == "broad_gain" and evidence["cases"] >= 2 and not ambiguous:
            return latest, opening[0], {"reason": "refine_supported_opening", "evidence": evidence}
        return None, opening[1], {"reason": "test_alternative_component_after_opening", "evidence": evidence,
                                  "largest_regression": (feedback.get("parent_comparison") or {})
                                  .get("largest_regression")}

    contract = next((failure for failure in failures if failure["kind"] == "candidate_contract"), None)
    if contract and contract["component"]:
        # Repair the candidate that broke the contract, so its hypothesis is still measured.
        return latest if latest.get("source") else best, "repair", {
            "reason": "candidate_contract_failure", "candidate_id": latest["candidate_id"],
            "component": contract["component"],
        }
    head = lineage_head(history, archive)
    if infrastructure:
        return head, "refine" if head else "diversify", {
            "reason": "runtime_environment_failure", "candidate_id": latest["candidate_id"],
        }
    if best is None:
        return None, "diversify", {"reason": "no_complete_candidate"}

    # An inconclusive trial (labels kept, dies exchanged, or a gain carried by one cell) does not move
    # the lineage: the next hypothesis builds on the lineage head, not on the trial or on the start.
    evidence = trial_evidence(feedback)
    reason = (evidence or {}).get("reason")
    if reason in ("orientation_only", "partition_mostly_unchanged", "timing_mostly_unchanged", "duplicate_outcome"):
        edited = tuple((feedback.get("attempt") or {}).get("changed_components", {}))
        intent = {("loss",): "decoder_only", ("decoder",): "loss_only"}.get(edited, "diversify")
        if edited == ("loss",) and reason != "duplicate_outcome":
            # The decoder sees only embeddings, so the loss is the lever on which groups share a die; in
            # every recorded run a decoder edit from the start only exchanged the dies. A loss term too weak
            # to move the labels is strengthened once before the decoder is tried.
            inert_losses = 0
            for row in reversed(archive):
                previous = trial_evidence(row) or {}
                if (tuple((row.get("attempt") or {}).get("changed_components", {})) != ("loss",)
                        or previous.get("reason") not in ("partition_mostly_unchanged", "timing_mostly_unchanged")):
                    break
                inert_losses += 1
            if inert_losses < 2:
                return head, "loss_only", {"candidate_id": latest["candidate_id"], **evidence,
                                           "next_step": "strengthen_the_loss_term_so_labels_move"}
        return head, intent, {"candidate_id": latest["candidate_id"], **evidence}
    if reason == "narrow_gain":
        return head, "diversify", {"candidate_id": latest["candidate_id"], **evidence}

    comparison = feedback.get("parent_comparison") or {}
    score_delta = comparison.get("official_score_delta")
    if comparison.get("complete") and score_delta is not None and score_delta < 0:
        regression = comparison["largest_regression"]
        improved, regressed = comparison["improved_pairs"], comparison["regressed_pairs"]
        paired = comparison["scored_pairs"]
        attempt = feedback.get("attempt", {})
        if (regression and improved > paired // 2 and 0 < regressed <= max(1, paired // 3)
                and not comparison["candidate_invalid_cells"] and attempt.get("search_intent") != "repair"):
            decision = {"reason": "concentrated_regression", "largest_regression": regression}
            edited = attempt.get("changed_components", {})
            if len(edited) == 1:
                decision["component"] = next(iter(edited))
            return latest, "repair", decision
        if reason == "broad_regression":
            return head, "diversify", {"candidate_id": latest["candidate_id"], **evidence}
    if head is None:
        return None, "diversify", {"reason": "start_still_ahead"}
    head_index = next(index for index, result in enumerate(history)
                      if result["request_id"] == head["request_id"])
    rounds_without_gain = len(history) - head_index - 1
    if rounds_without_gain and rounds_without_gain % 2 == 0:
        return head, "diversify", {"reason": "search_plateau"}
    return head, "refine", {"reason": "refine_best"}


def save_best(folder, result):
    pending = folder / "program.py.pending"
    pending.write_text(result["source"], encoding="utf-8")
    pending.replace(folder / "program.py")


def claim_once(context, filename):
    workspace = Path(context.files.get("workspaces", "/workspace/workspaces"))
    try:
        with (workspace / filename).open("x", encoding="utf-8"):
            return True
    except FileExistsError:
        return False


def published_initial_reference(context):
    table = reference_table()["cases"]
    cells = []
    for cell in context.evaluation_cells:
        case, seed = cell["case"], str(cell["seed"])
        entry = table.get(case, {})
        sample = entry.get("measurements", {}).get(context.task, {}).get(seed)
        if entry.get("split") != "train" or sample is None:
            return None
        metrics = score_metrics(case, {
            "runs_successfully": 1, "sta_graph_complete": 1,
            "timing_evaluation_boundary": "post-opt-pre-cts", "combined_score": 0,
            "final_global_wns_setup_ns": sample["wns_ns"], "final_global_tns_setup_ns": sample["tns_ns"],
        })
        cells.append({"case": case, "seed": seed, "status": "published_reference",
                      "score": metrics["combined_score"], "metrics": metrics})
    if not cells:
        return None
    cases = {cell["case"] for cell in cells}
    score = mean(mean(cell["score"] for cell in cells if cell["case"] == case) for case in sorted(cases))
    return {"candidate_id": context.baseline_candidate, "status": "published_reference",
            "feedback": {"score": score, "cells": cells}}


def same_host_start(published, history, partitions):
    """The start reference with each cell measured on this host where a receipt kept the start assignment.

    The physical flow is deterministic for a fixed assignment on one host, so an official cell whose
    organizer diagnostic reports zero changed instances is the start program's own timing here. Other
    cells keep the published value, which can come from a different host.
    """
    if published is None:
        return None
    measured = {}
    for result in history:
        if result["status"] != "succeeded":
            continue
        changes = partitions.get(result["request_id"]) or {}
        for cell in result["feedback"]["cells"]:
            key = cell["case"], str(cell["seed"])
            if (changes.get(key, {}).get("partition_assignment_changed_count") == 0
                    and cell.get("status") == "succeeded" and isinstance(cell.get("score"), (int, float))):
                measured.setdefault(key, cell)
    if not measured:
        return published
    cells = []
    for cell in published["feedback"]["cells"]:
        same = measured.get((cell["case"], str(cell["seed"])))
        cells.append({**cell, "score": same["score"], "metrics": same.get("metrics") or {},
                      "measured_on": "this_host"} if same else {**cell, "measured_on": "published"})
    cases = sorted({cell["case"] for cell in cells})
    score = mean(mean(cell["score"] for cell in cells if cell["case"] == case) for case in cases)
    return {**published, "feedback": {"score": score, "cells": cells},
            "summary": {"same_host_cells": len(measured), "published_cells": len(cells) - len(measured),
                        "same_host_score": score, "published_score": published["feedback"]["score"]}}


def bridge(call, *args, attempts=BRIDGE_ATTEMPTS):
    """Repeat a read-only bridge call through transient socket errors."""
    for attempt in range(attempts):
        try:
            return call(*args)
        except AgentAPIError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(60, 5 * 2 ** attempt))


def summarize_cell(cell):
    summary = {key: cell[key] for key in ("case", "seed", "score", "raw_score", "valid", "status", "metrics") if key in cell}
    reason = cell.get("reason") or cell.get("error_message")
    if reason:
        summary["reason"] = str(reason)[:400]
    evidence = failure_evidence(cell)
    if evidence is not None:
        summary["failure"] = evidence
    return summary


def summarize(result):
    feedback = result.get("feedback", {})
    cells = feedback.get("cells", [])
    case_scores = {}
    for cell in cells:
        if isinstance(cell.get("score"), (int, float)):
            case_scores.setdefault(cell["case"], []).append(cell["score"])
    return {
        "request_id": result["request_id"],
        "candidate_id": result["candidate_id"],
        "status": result["status"],
        "reason": str(result.get("reason") or "")[:400],
        "score": feedback.get("score"),
        "case_scores": {case: {"mean": mean(scores), "worst_seed_score": min(scores),
                               "scored_seeds": len(scores)} for case, scores in case_scores.items()},
        "cells": [summarize_cell(cell) for cell in cells],
    }


def write_event(folder, event, **fields):
    with (folder / "search.jsonl").open("a+b") as log:
        log.seek(0, 2)
        separator = b""
        if log.tell():
            log.seek(-1, 2)
            if log.read(1) != b"\n":
                separator = b"\n"
        line = json.dumps({"event": event, **fields}, ensure_ascii=False)
        log.write(separator + line.encode("utf-8") + b"\n")


def submit_and_wait(context, source, request_id):
    """Submit once; raise AgentAPIError only when the bridge has no record of the request."""
    try:
        result = context.submit_evaluation(source, request_id=request_id)
    except AgentAPIError:
        if not any(row["request_id"] == request_id for row in bridge(context.status)["history"]):
            raise
    else:
        if result["status"] in {"succeeded", "failed", "rejected"}:
            return result
    return bridge(context.wait_evaluation, request_id)


class Agent:
    uses_separate_planning = True

    def run(self, context):
        """Search, and on any unexpected error after a complete evaluation, return the best candidate.

        A crash in feedback analysis must not discard the evaluations already spent: the official
        result needs a candidate ID, and an exception from Agent.run fails the whole formal run.
        """
        self.best = None
        try:
            return self._search(context)
        except Exception as error:
            if self.best is None:
                raise
            folder = Path(__file__).resolve().parent
            write_event(folder, "agent_error", error=f"{type(error).__name__}: {error}"[:400],
                        traceback=traceback.format_exc()[-2000:])
            write_event(folder, "selected", selection={"rule": "highest_official_score_after_agent_error"},
                        **summarize(self.best))
            return self.best["candidate_id"]

    def _search(self, context):
        if context.task != "loss_decoder":
            raise ValueError("This Agent implements the loss_decoder baseline")
        folder = Path(__file__).resolve().parent
        opening = OPENING
        opening_file = folder / "opening.txt"
        if opening_file.is_file():
            first = opening_file.read_text(encoding="utf-8").strip()
            if first not in OPENING:
                raise ValueError("Opening must be loss_only or decoder_only")
            opening = (first, next(intent for intent in OPENING if intent != first))
        task = Path(context.files["framework"]) / "local_tasks" / DECODER_TASK
        initial = validate_program((task / "initial_program.py").read_text(encoding="utf-8"), DECODER_TASK)
        instructions, marker, experience = (folder / "prompt.md").read_text(encoding="utf-8").partition(EXPERIENCE_HEADING)
        interface = (task / "prompt.md").read_text(encoding="utf-8")

        state = bridge(context.status)
        history = []
        attempts = load_attempts(folder / "search.jsonl")
        receipts_by_candidate = {}
        partition_cache = {}
        evaluation_root = context.files.get("evaluations", "/workspace/evaluations")
        initial_reference = published_initial_reference(context)
        if initial_reference is not None:
            receipts_by_candidate[context.baseline_candidate] = initial_reference

        def partition_for(result):
            request_id = result["request_id"]
            if request_id not in partition_cache:
                partition_cache[request_id] = published_partition_changes(result, evaluation_root)
            return partition_cache[request_id]

        def refresh_start():
            try:
                start = same_host_start(initial_reference, history,
                                        {result["request_id"]: partition_for(result) for result in history})
            except Exception as error:
                write_event(folder, "analysis_error", stage="start_reference",
                            error=f"{type(error).__name__}: {error}"[:400])
                start = initial_reference
            if start is not None:
                receipts_by_candidate[context.baseline_candidate] = start
            return start

        snapshot_cache, fingerprints, analysis_errors = {}, {}, set()

        def fingerprint(result):
            if result["request_id"] not in fingerprints:
                fingerprints[result["request_id"]] = outcome_fingerprint(result)
            return fingerprints[result["request_id"]]

        def assignments_vs_parent(result, parent_id):
            """Measured label changes from a candidate parent, read from both published label files."""
            parent = receipts_by_candidate.get(parent_id)
            if parent is None or parent.get("status") != "succeeded" or parent_id == result["candidate_id"]:
                return {}
            changes = {}
            for cell in result["feedback"]["cells"]:
                key = cell["case"], str(cell["seed"])
                change = assignment_difference(assignment_snapshot(parent, evaluation_root, key, snapshot_cache),
                                               assignment_snapshot(result, evaluation_root, key, snapshot_cache))
                if change is not None:
                    changes[key] = change
            return changes

        def summarize_for_prompt(result):
            """The analysed receipt; on an analysis error, the plain receipt summary and the error."""
            try:
                return analyse(result)
            except Exception as error:
                if result["request_id"] not in analysis_errors:
                    analysis_errors.add(result["request_id"])
                    write_event(folder, "analysis_error", request_id=result["request_id"],
                                error=f"{type(error).__name__}: {error}"[:400], traceback=traceback.format_exc()[-1500:])
                summary = summarize(result)
                summary["analysis_error"] = type(error).__name__
                attempt = attempts.get(result["request_id"])
                if attempt:
                    summary["attempt"] = attempt
                return summary

        def analyse(result):
            summary = summarize(result)
            own = fingerprint(result)
            if own is not None:
                for earlier in history:
                    if earlier["request_id"] == result["request_id"]:
                        break
                    if earlier["candidate_id"] != result["candidate_id"] and fingerprint(earlier) == own:
                        summary["duplicate_of"] = earlier["candidate_id"]
                        break
            partition = partition_for(result)
            if partition:
                moved, swaps = {}, set()
                for key, row in partition.items():
                    fraction = row.get("partition_assignment_changed_fraction")
                    if row["partition_assignment_changed_count"] == 0:
                        moved[key] = 0.0
                    elif isinstance(fraction, (int, float)):
                        moved[key] = min(fraction, 1 - fraction)
                        if fraction > SWAP_FRACTION:
                            swaps.add(key)
                summary["partition_changes_vs_initial"] = {
                    "reported_cells": len(partition),
                    "identical_assignments": sum(row["partition_assignment_changed_count"] == 0
                                                 for row in partition.values()),
                    "die_swaps": len(swaps),
                    "kept_up_to_swap": sum(value < KEPT_UP_TO_SWAP for value in moved.values()),
                }
                for cell in summary["cells"]:
                    key = cell["case"], str(cell["seed"])
                    change = partition.get(key)
                    if change is not None:
                        cell["partition_change_vs_initial"] = {**change, "die_swap_vs_initial": key in swaps}
                        if key in moved:
                            cell["partition_change_vs_initial"]["moved_fraction_up_to_swap"] = moved[key]
            attempt = attempts.get(result["request_id"], {})
            if attempt:
                summary["attempt"] = attempt
            comparison = comparison_for_prompt(result, attempt.get("parent_id"), receipts_by_candidate,
                                               context.evaluation_cells,
                                               assignments_vs_parent(result, attempt.get("parent_id"))
                                               if result["status"] == "succeeded" else None)
            if comparison is not None:
                summary["parent_comparison"] = comparison
                if comparison.get("parent_source") == "published_initial_reference":
                    discrepancies = [
                        {key: cell[key] for key in ("case", "seed", "score_delta", "wns_slack_delta_ns", "tns_slack_delta_ns")}
                        for cell in comparison["cells"]
                        if partition.get((cell["case"], cell["seed"]), {}).get("partition_assignment_changed_count") == 0
                        and abs(cell["score_delta"]) > 1e-6
                    ]
                    if discrepancies:
                        summary["timing_changes_with_identical_assignment"] = sorted(
                            discrepancies, key=lambda cell: abs(cell["score_delta"]), reverse=True)[:3]
            return summary

        seen = {function_keys(initial)}
        best = None

        def record(result, logged=True):
            nonlocal best
            history.append(result)
            if result.get("source"):
                seen.add(function_keys(result["source"]))
            if result["status"] == "succeeded":
                receipts_by_candidate[result["candidate_id"]] = result
                if best is None or score_key(result) > score_key(best):
                    best = self.best = result
                    if logged:
                        save_best(folder, best)
                        write_event(folder, "incumbent", **summarize(best))

        for row in state["history"]:
            record(bridge(context.wait_evaluation, row["request_id"]), logged=False)
        if best is not None:
            save_best(folder, best)

        issue = None
        initial_experience_pending = bool(marker and not history)
        consecutive_api_failures = 0
        plan_cache = {}
        consecutive_invalid = 0
        shorten_next = False
        while True:
            state = bridge(context.status)
            known = {result["request_id"] for result in history}
            for row in state["history"]:
                if row["request_id"] not in known:  # accepted while its reply was lost
                    result = bridge(context.wait_evaluation, row["request_id"])
                    write_event(folder, "evaluated", **summarize_for_prompt(result))
                    record(result)
            if not state["evaluation_dispatch_allowed"] or not state["model_dispatch_allowed"]:
                write_event(folder, "budget_stop", evaluations_remaining=state["evaluations_remaining"],
                            model_stop_reasons=state["model_stop_reasons"])
                break

            start = refresh_start()
            archive = [summarize_for_prompt(result) for result in history]
            try:
                parent, intent, decision = select_parent(history, best, archive, start, opening)
            except Exception as error:
                # Keep spending the budget from the best candidate rather than stopping the run.
                write_event(folder, "analysis_error", stage="select_parent",
                            error=f"{type(error).__name__}: {error}"[:400], traceback=traceback.format_exc()[-1500:])
                parent, intent, decision = best, ("refine" if best else "diversify"), {"reason": "analysis_error"}
            if intent == "retry":
                request_id = "r-" + parent["request_id"][2:]
                attempt = {**attempts.get(parent["request_id"], {}), "retry_of": parent["request_id"]}
                attempts[request_id] = attempt
                write_event(folder, "submitted", request_id=request_id, decision=decision, **attempt)
                try:
                    result = submit_and_wait(context, parent["source"], request_id)
                except AgentAPIError as error:
                    attempts.pop(request_id, None)
                    attempts[parent["request_id"]] = {**attempts.get(parent["request_id"], {}),
                                                      "retry_refused": str(error)[:200]}
                    write_event(folder, "retry_refused", request_id=parent["request_id"], reason=str(error)[:400])
                    continue
                write_event(folder, "evaluated", parent_id=attempt.get("parent_id"), **summarize_for_prompt(result))
                record(result)
                continue
            parent_id = parent["candidate_id"] if parent else context.baseline_candidate
            parent_source = parent["source"] if parent else initial
            summaries = {row["request_id"]: row for row in archive}
            parent_feedback = summaries[parent["request_id"]] if parent else None
            incumbent_feedback = summaries[best["request_id"]] if best and best is not parent else None
            displayed = {row["request_id"] for row in (parent_feedback, incumbent_feedback) if row is not None}
            previous_results = [row for row in archive if row["request_id"] not in displayed][-2:]
            displayed.update(row["request_id"] for row in previous_results)
            focus = parent_feedback or incumbent_feedback
            if archive and any("failure" in cell for cell in archive[-1]["cells"]):
                focus = archive[-1]
            request = {
                "search_intent": intent,
                "decision": decision,
                "parent_id": parent_id,
                "parent_program": parent_source,
                "training_cells": context.evaluation_cells,
                "evaluations_remaining": state["evaluations_remaining"],
                "parent_feedback": compact_feedback(parent_feedback),
                "incumbent_feedback": compact_feedback(incumbent_feedback),
                "recent_feedback": [compact_feedback(row) for row in previous_results],
                "prior_trials": [summarize_trial(row) for row in archive],
                "related_experiment": related_experiment(focus, archive, displayed),
                "previous_generation_issue": issue,
            }
            if start is not None and start.get("summary"):
                request["start_reference"] = start["summary"]
            system_prompt = instructions.rstrip() + "\n\n" + interface
            plan_key = len(history), parent_id, intent
            try:
                if plan_key not in plan_cache:
                    planner_prompt = PLAN_TASK + "\n\n" + interface
                    if initial_experience_pending:
                        initial_experience_pending = False
                        if claim_once(context, "loss-decoder-initial-prompt-sent"):
                            planner_prompt += "\n\n" + marker + experience
                    plan_cache[plan_key] = ""
                    # An over-long or empty plan gets one shorter, lower-effort attempt before code generation
                    # proceeds without a plan.
                    for effort, planning_request in (("high", request), ("low", shortened(request))):
                        try:
                            plan_cache[plan_key] = model_text(
                                ask_model(context, planner_prompt, planning_request, effort, "plan"))[:1800]
                            break
                        except ValueError as error:
                            write_event(folder, "plan_unavailable", parent_id=parent_id, effort=effort,
                                        reason=str(error)[:250])
                            if not bridge(context.status)["model_dispatch_allowed"]:
                                break
                        except AgentAPIError as error:
                            write_event(folder, "plan_unavailable", parent_id=parent_id, effort=effort,
                                        reason=str(error)[:250])
                            break
                if not bridge(context.status)["model_dispatch_allowed"]:
                    write_event(folder, "stop_before_codegen", parent_id=parent_id)
                    break
                generation = {key: request[key] for key in (
                    "search_intent", "decision", "parent_id", "parent_program", "training_cells",
                    "evaluations_remaining", "previous_generation_issue",
                )}
                generation["experiment_plan"] = plan_cache[plan_key]
                generation["prior_trials"] = request["prior_trials"][-4:]
                focus_comparison = (focus or {}).get("parent_comparison") or {}
                generation["largest_regression"] = focus_comparison.get("largest_regression")
                if not plan_cache[plan_key]:
                    generation["parent_feedback"] = request["parent_feedback"]
                    generation["incumbent_feedback"] = request["incumbent_feedback"]
                if shorten_next:
                    generation = shortened(generation)
                response = ask_model(context, system_prompt, generation)
            except AgentAPIError as error:
                issue = str(error)
                write_event(folder, "model_error", parent_id=parent_id, reason=issue)
                consecutive_api_failures += 1
                if consecutive_api_failures >= 5:
                    break
                time.sleep(min(60, 2 ** consecutive_api_failures))
                continue
            consecutive_api_failures = 0

            try:
                source = generated_program(response)
                parent_key = function_keys(parent_source)
                unchanged = UNCHANGED_FUNCTION.get(intent)
                if intent == "repair" and "component" in decision:
                    unchanged = UNCHANGED_FUNCTION[decision["component"] + "_only"]
                key = proposal_keys(source, parent_key, unchanged, seen)
                # The bridge deduplicates this ID if the submit acknowledgement is lost.
                request_id = "e-" + hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:60]
                if any(row["request_id"] == request_id for row in history):
                    raise ValueError("That program already has an official evaluation record")
                candidate_gate(source)
            except (ValueError, SyntaxError) as error:
                issue = str(error)[:400]
                write_event(folder, "invalid_proposal", parent_id=parent_id, reason=issue)
                if issue.startswith("The model returned an incomplete response"):
                    # The reasoning used the whole output allowance. Ask again with less input.
                    issue = issue.rstrip(".") + ". Reply with the complete program only and keep the edit small."
                    shorten_next = True
                consecutive_invalid += 1
                if consecutive_invalid % REPLAN_AFTER_INVALID == 0 and plan_cache.pop(plan_key, None) is not None:
                    write_event(folder, "replan", parent_id=parent_id, reason=issue)
                if consecutive_invalid >= (STOP_AFTER_INVALID if best is not None else STOP_WITHOUT_CANDIDATE):
                    write_event(folder, "generation_stop", parent_id=parent_id, reason=issue)
                    break
                continue

            consecutive_invalid = 0
            shorten_next = False
            seen.add(key)
            hypotheses = {}
            for component, function, updated, previous in zip(
                    ("loss", "decoder"), ast.parse(source).body, key, parent_key):
                if updated != previous:
                    hypotheses[component] = " ".join((ast.get_docstring(function) or "").split())[:200]
            attempt = {"parent_id": parent_id, "search_intent": intent, "changed_components": hypotheses}
            attempts[request_id] = attempt
            write_event(folder, "submitted", request_id=request_id, decision=decision, **attempt)
            try:
                result = submit_and_wait(context, source, request_id)
            except AgentAPIError as error:
                # The bridge has no record of this request, so no evaluation was spent on it.
                seen.discard(key)
                attempts.pop(request_id, None)
                write_event(folder, "submit_failed", request_id=request_id, reason=str(error)[:400])
                consecutive_api_failures += 1
                if consecutive_api_failures >= 5:
                    break
                time.sleep(min(60, 2 ** consecutive_api_failures))
                continue
            feedback = summarize_for_prompt(result)
            write_event(folder, "evaluated", parent_id=parent_id, **feedback)
            issue = None
            record(result)

        if best is None:
            raise RuntimeError("No candidate completed an official training evaluation" + (f": {issue}" if issue else ""))
        chosen, selection = select_final(history, refresh_start(), best)
        if chosen is not best:
            save_best(folder, chosen)
        write_event(folder, "selected", selection=selection, **summarize(chosen))
        return chosen["candidate_id"]
