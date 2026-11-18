"""Search the task-2 loss and decoder through the metered contest bridge."""

import ast
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from statistics import mean
import time

from contest2.agent_api import AgentAPIError
from contest2.model_pricing import MODEL_ID
from local_tasks.tp_gnn_local_task_definitions import DECODER_TASK, validate_program


EXPERIENCE_HEADING = "## 离线专家知识"
UNCHANGED_FUNCTION = {"loss_only": 1, "decoder_only": 0}
EDITABLE_COMPONENTS = ("loss", "decoder")
OPENING_INTENTS = ("loss_only", "decoder_only")
COMPARISON_METRICS = (
    "final_global_wns_setup_ns",
    "final_global_tns_setup_ns",
    "wns_deficit_improvement",
    "tns_deficit_improvement",
    "cut_net_count",
    "timing_weighted_cut",
)


def function_keys(source):
    functions = ast.parse(source).body
    for function in functions:
        if ast.get_docstring(function, clean=False) is not None:
            function.body.pop(0)
    return tuple(ast.dump(function, include_attributes=False) for function in functions)


def generated_program(response):
    if not isinstance(response, dict) or response.get("status") != "completed":
        raise ValueError("The model returned an incomplete response")
    parts = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
    text = "\n".join(parts).strip()
    fence = re.fullmatch(r"```(?:python|py)?[ \t]*\n(.*?)\n```", text, re.DOTALL)
    source = fence[1] if fence else text
    return validate_program(source, DECODER_TASK)


def score_key(result):
    feedback = result["feedback"]
    cells = feedback["cells"]
    tns = [cell.get("metrics", {}).get("tns_deficit_improvement") for cell in cells]
    complete_tns = bool(tns) and all(is_number(value) for value in tns)
    return feedback["score"], mean(tns) if complete_tns else float("-inf")


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def scores_by_case(cells):
    grouped = {}
    for cell in cells:
        if is_number(cell.get("score")):
            grouped.setdefault(cell["case"], []).append(cell)
    scores = {}
    for case, scored in grouped.items():
        worst = min(scored, key=lambda cell: cell["score"])
        scores[case] = {
            "mean": mean(cell["score"] for cell in scored),
            "worst_seed": worst["seed"],
            "worst_seed_score": worst["score"],
            "scored_seeds": len(scored),
        }
    return scores


def select_parent(history, best, opening=OPENING_INTENTS):
    if len(history) < 2:
        return None, opening[len(history)]
    if best is None:
        return None, "diversify"

    latest_best = next(index for index, result in enumerate(history)
                       if result["request_id"] == best["request_id"])
    rounds_without_gain = len(history) - latest_best - 1
    if rounds_without_gain and rounds_without_gain % 2 == 0:
        alternatives = [result for result in history if result["status"] == "succeeded"
                        and result["candidate_id"] != best["candidate_id"]]
        leader_cases = scores_by_case(best["feedback"]["cells"])
        if leader_cases:
            weak_case = min(leader_cases, key=lambda case: leader_cases[case]["mean"])
            leader = leader_cases[weak_case]
            complementary = []
            for result in alternatives:
                challenger = scores_by_case(result["feedback"]["cells"]).get(weak_case)
                if (challenger and challenger["scored_seeds"] == leader["scored_seeds"]
                        and challenger["mean"] > leader["mean"]):
                    complementary.append(result)
            if complementary:
                return max(complementary, key=score_key), "diversify"
        return max(alternatives, key=score_key, default=None), "diversify"
    return best, "refine"


def claim_initial_prompt(workspace):
    try:
        with (workspace / "loss-decoder-initial-prompt-sent").open("x", encoding="utf-8"):
            return True
    except FileExistsError:
        return False


def summarize_cell(cell, reference):
    fields = ("case", "seed", "score", "raw_score", "valid", "status", "metrics", "error_type")
    summary = {key: cell[key] for key in fields if key in cell}
    reason = cell.get("reason") or cell.get("error_message")
    if reason:
        summary["reason"] = str(reason)[:400]
    diagnostics = cell.get("diagnostics", [])
    if diagnostics:
        summary["diagnostics"] = [
            {"source": item["source"][-120:], "excerpt": item["excerpt"][-600:]}
            for item in diagnostics[:2]
        ]
    if reference is not None:
        score, reference_score = cell.get("score"), reference.get("score")
        if is_number(score) and is_number(reference_score):
            summary["score_delta"] = round(score - reference_score, 6)
        metrics, reference_metrics = cell.get("metrics", {}), reference.get("metrics", {})
        for key in COMPARISON_METRICS:
            value, reference_value = metrics.get(key), reference_metrics.get(key)
            if is_number(value) and is_number(reference_value):
                summary[key + "_delta"] = round(value - reference_value, 6)
    return summary


def summarize(result, comparison=None):
    feedback = result.get("feedback", {})
    cells = feedback.get("cells", [])
    reference_cells = {}
    summary = {
        "request_id": result["request_id"],
        "candidate_id": result["candidate_id"],
        "status": result["status"],
        "reason": str(result.get("reason") or "")[:400],
        "score": feedback.get("score"),
        "case_scores": scores_by_case(cells),
    }
    for key in ("complete", "expected_cells", "succeeded_cells"):
        if key in feedback:
            summary[key] = feedback[key]
    if comparison is not None:
        summary["comparison_candidate_id"] = comparison["candidate_id"]
        reference = comparison["feedback"]
        reference_cells = {(cell["case"], str(cell["seed"])): cell for cell in reference["cells"]}
        if is_number(summary["score"]) and is_number(reference.get("score")):
            summary["score_delta"] = round(summary["score"] - reference["score"], 6)
    summary["cells"] = [summarize_cell(cell, reference_cells.get((cell["case"], str(cell["seed"]))))
                        for cell in cells]
    return summary


def feedback_for(result, edits, evaluated, fallback=None):
    parent_id = edits.get(result["request_id"], {}).get("parent_id")
    comparison = evaluated.get(parent_id) or fallback
    feedback = summarize(result, comparison)
    if parent_id is not None:
        feedback["actual_parent_id"] = parent_id
    if comparison is not None:
        feedback["comparison_is_parent"] = comparison["candidate_id"] == parent_id
    return feedback


def experiment_history(history, edits, evaluated):
    experiments = []
    for result in history:
        edit = edits.get(result["request_id"], {})
        summary = feedback_for(result, edits, evaluated)
        experiment = {
            **edit,
            "candidate_id": result["candidate_id"],
            "status": result["status"],
            "score": summary["score"],
            "case_means": {case: round(values["mean"], 4) for case, values in summary["case_scores"].items()},
        }
        if "score_delta" in summary:
            experiment["parent_score_delta"] = summary["score_delta"]
        deltas = [cell["score_delta"] for cell in summary["cells"] if "score_delta" in cell]
        if deltas:
            experiment["seeds_vs_parent"] = {
                "improved": sum(delta > 0 for delta in deltas),
                "tied": sum(delta == 0 for delta in deltas),
                "regressed": sum(delta < 0 for delta in deltas),
            }
        experiments.append(experiment)
    return experiments


def describe_edit(source, parent_source, key, parent_key):
    functions = ast.parse(source).body
    parents = ast.parse(parent_source).body
    changes = {}
    for component, function, parent, current, prior in zip(EDITABLE_COMPONENTS, functions, parents, key, parent_key):
        if current == prior:
            continue
        hypothesis = ast.get_docstring(function)
        if hypothesis == ast.get_docstring(parent):
            hypothesis = None
        changes[component] = " ".join(hypothesis.split())[:200] if hypothesis else None
    return changes


def save_edit(path, edit):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(edit, ensure_ascii=False), encoding="utf-8")
    pending.replace(path)


def write_event(folder, event, **fields):
    with (folder / "search.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": event, **fields}, ensure_ascii=False) + "\n")


def save_best(folder, result):
    pending = folder / "program.py.pending"
    pending.write_text(result["source"], encoding="utf-8")
    pending.replace(folder / "program.py")


def submit_and_wait(context, source, request_id):
    try:
        result = context.submit_evaluation(source, request_id=request_id)
    except AgentAPIError:
        if not any(row["request_id"] == request_id for row in context.status()["history"]):
            raise
    else:
        if result["status"] in {"succeeded", "failed", "rejected"}:
            return result
    return context.wait_evaluation(request_id)


class Agent:
    def run(self, context):
        if context.task != "loss_decoder":
            raise ValueError("This Agent implements the loss_decoder baseline")
        folder = Path(__file__).resolve().parent
        workspace = Path(context.files.get("workspaces", "/workspace/workspaces"))
        journal = workspace / "loss-decoder-experiments"
        journal.mkdir(exist_ok=True)
        plan = folder / "opening.txt"
        opening = OPENING_INTENTS
        if plan.is_file():
            first = plan.read_text(encoding="utf-8").strip()
            if first not in OPENING_INTENTS:
                raise ValueError("Opening must be loss_only or decoder_only")
            opening = (first, next(intent for intent in OPENING_INTENTS if intent != first))
        task = Path(context.files["framework"]) / "local_tasks" / DECODER_TASK
        initial = validate_program((task / "initial_program.py").read_text(encoding="utf-8"), DECODER_TASK)
        instructions, marker, experience = (folder / "prompt.md").read_text(encoding="utf-8").partition(EXPERIENCE_HEADING)
        interface = (task / "prompt.md").read_text(encoding="utf-8")

        state = context.status()
        best = None
        history = []
        edits = {}
        evaluated = {}
        seen = {function_keys(initial)}
        for row in state["history"]:
            result = context.wait_evaluation(row["request_id"])
            history.append(result)
            path = journal / (row["request_id"] + ".json")
            if path.is_file():
                edits[row["request_id"]] = json.loads(path.read_text(encoding="utf-8"))
            if result["status"] == "succeeded":
                evaluated[result["candidate_id"]] = result
                seen.add(function_keys(result["source"]))
                if best is None or score_key(result) > score_key(best):
                    best = result
        if best is not None:
            save_best(folder, best)

        issue = None
        initial_experience_pending = bool(marker and not history)
        consecutive_api_failures = 0
        while True:
            state = context.status()
            if not state["evaluation_dispatch_allowed"] or not state["model_dispatch_allowed"]:
                write_event(folder, "budget_stop", evaluations_remaining=state["evaluations_remaining"],
                            model_stop_reasons=state["model_stop_reasons"])
                break

            parent, intent = select_parent(history, best, opening)
            parent_id = parent["candidate_id"] if parent else context.baseline_candidate
            parent_source = parent["source"] if parent else initial
            parent_key = function_keys(parent_source)
            comparison = parent or best
            other_results = [result for result in history if result is not parent and result is not best]
            request = {
                "search_intent": intent,
                "parent_id": parent_id,
                "parent_program": parent_source,
                "training_cells": context.evaluation_cells,
                "evaluations_remaining": state["evaluations_remaining"],
                "parent_feedback": feedback_for(parent, edits, evaluated) if parent else None,
                "incumbent_feedback": feedback_for(best, edits, evaluated, parent) if best and best is not parent else None,
                "recent_feedback": [feedback_for(result, edits, evaluated, comparison) for result in other_results[-2:]],
                "experiment_history": experiment_history(history, edits, evaluated),
                "previous_generation_issue": issue,
            }
            system_prompt = instructions.rstrip() + "\n\n" + interface
            if initial_experience_pending:
                initial_experience_pending = False
                if claim_initial_prompt(workspace):
                    system_prompt += "\n\n" + marker + experience
            try:
                response = context.model.responses_create({
                    "model": MODEL_ID,
                    "stream": False,
                    "input": [
                        {"type": "message", "role": "system", "content": system_prompt},
                        {"type": "message", "role": "user", "content": json.dumps(request, ensure_ascii=False)},
                    ],
                })
            except AgentAPIError as error:
                issue = str(error)
                write_event(folder, "model_error", parent_id=parent_id, reason=issue)
                consecutive_api_failures += 1
                if consecutive_api_failures >= 2:
                    break
                time.sleep(2)
                continue
            consecutive_api_failures = 0

            try:
                source = generated_program(response)
                key = function_keys(source)
                if key in seen:
                    raise ValueError("The response repeats an already considered program")
                unchanged = UNCHANGED_FUNCTION.get(intent)
                if unchanged is not None and key[unchanged] != parent_key[unchanged]:
                    raise ValueError("Keep the other function unchanged for this paired search direction")
            except (ValueError, SyntaxError) as error:
                issue = str(error)[:400]
                write_event(folder, "invalid_proposal", parent_id=parent_id, reason=issue)
                continue

            # Keep the request ID stable across comment and docstring edits, including after a restart.
            request_id = "e-" + hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:60]
            if any(row["request_id"] == request_id for row in history):
                issue = "That program already has an official evaluation record"
                continue
            seen.add(key)
            edit = {"parent_id": parent_id, "search_intent": intent,
                    "changed_components": describe_edit(source, parent_source, key, parent_key)}
            save_edit(journal / (request_id + ".json"), edit)
            edits[request_id] = edit
            write_event(folder, "submitted", parent_id=parent_id, request_id=request_id, search_intent=intent)
            result = submit_and_wait(context, source, request_id)
            feedback = feedback_for(result, edits, evaluated, comparison)
            history.append(result)
            write_event(folder, "evaluated", parent_id=parent_id, **feedback)
            issue = None
            if result["status"] == "succeeded":
                evaluated[result["candidate_id"]] = result
                if best is None or score_key(result) > score_key(best):
                    best = result
                    save_best(folder, best)
                    write_event(folder, "incumbent", **summarize(best))

        if best is None:
            raise RuntimeError("No candidate completed an official training evaluation" + (f": {issue}" if issue else ""))
        write_event(folder, "selected", **summarize(best))
        return best["candidate_id"]
