"""Search the task-2 loss and decoder through the metered contest bridge."""

import ast
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import time

from contest2.agent_api import AgentAPIError
from contest2.model_pricing import MODEL_ID
from contest2.scoring import reference_table, score_metrics
from local_tasks.tp_gnn_local_task_definitions import DECODER_TASK, validate_program
from paired_feedback import comparison_for_prompt, failure_evidence, load_attempts, related_experiment


EXPERIENCE_HEADING = "## 离线专家知识"
UNCHANGED_FUNCTION = {"loss_only": 1, "decoder_only": 0}
OPENING = ("loss_only", "decoder_only")


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
    complete_tns = bool(tns) and all(value is not None for value in tns)
    return feedback["score"], mean(tns) if complete_tns else float("-inf")


def select_parent(history, best, archive, initial_reference, opening=OPENING):
    if len(history) < 2:
        return None, opening[len(history)], {"reason": "opening_control"}

    latest, feedback = history[-1], archive[-1]
    failures = [cell["failure"] for cell in feedback["cells"] if "failure" in cell]
    contract = next((failure for failure in failures if failure["kind"] == "candidate_contract"), None)
    if contract and contract["component"]:
        return best, contract["component"] + "_only", {
            "reason": "candidate_contract_failure", "candidate_id": latest["candidate_id"],
            "component": contract["component"],
        }
    if failures and all(failure["kind"] == "runtime_environment" for failure in failures):
        return best, "refine" if best else "diversify", {
            "reason": "runtime_environment_failure", "candidate_id": latest["candidate_id"],
        }
    if best is None:
        return None, "diversify", {"reason": "no_complete_candidate"}

    comparison = feedback.get("parent_comparison") or {}
    below_start = (initial_reference is not None
                   and best["feedback"]["score"] <= initial_reference["feedback"]["score"])
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
        if regressed > paired // 2:
            return None if below_start else best, "diversify", {
                "reason": "broad_regression", "candidate_id": latest["candidate_id"],
                "regressed_pairs": regressed, "scored_pairs": paired,
            }
    if below_start and len(history) % 2 == 0:
        return None, "diversify", {"reason": "published_start_still_ahead"}
    latest_best = next(index for index, result in enumerate(history)
                       if result["request_id"] == best["request_id"])
    rounds_without_gain = len(history) - latest_best - 1
    if rounds_without_gain and rounds_without_gain % 2 == 0:
        alternatives = [result for result in history if result["status"] == "succeeded"
                        and result["candidate_id"] != best["candidate_id"]]
        return max(alternatives, key=score_key, default=None), "diversify", {"reason": "search_plateau"}
    return best, "refine", {"reason": "refine_best"}


def save_best(folder, result):
    pending = folder / "program.py.pending"
    pending.write_text(result["source"], encoding="utf-8")
    pending.replace(folder / "program.py")


def claim_initial_prompt(context):
    workspace = Path(context.files.get("workspaces", "/workspace/workspaces"))
    try:
        with (workspace / "loss-decoder-initial-prompt-sent").open("x", encoding="utf-8"):
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

        state = context.status()
        best = None
        history = []
        attempts = load_attempts(folder / "search.jsonl")
        receipts_by_candidate = {}
        initial_reference = published_initial_reference(context)
        if initial_reference is not None:
            receipts_by_candidate[context.baseline_candidate] = initial_reference

        def summarize_for_prompt(result):
            summary = summarize(result)
            attempt = attempts.get(result["request_id"], {})
            if attempt:
                summary["attempt"] = attempt
            comparison = comparison_for_prompt(result, attempt.get("parent_id"), receipts_by_candidate,
                                               context.evaluation_cells)
            if comparison is not None:
                summary["parent_comparison"] = comparison
            return summary

        seen = {function_keys(initial)}
        for row in state["history"]:
            result = context.wait_evaluation(row["request_id"])
            history.append(result)
            if result["status"] == "succeeded":
                receipts_by_candidate[result["candidate_id"]] = result
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

            archive = [summarize_for_prompt(result) for result in history]
            parent, intent, decision = select_parent(history, best, archive, initial_reference, opening)
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
                "parent_feedback": parent_feedback,
                "incumbent_feedback": incumbent_feedback,
                "recent_feedback": previous_results,
                "related_experiment": related_experiment(focus, archive, displayed),
                "previous_generation_issue": issue,
            }
            system_prompt = instructions.rstrip() + "\n\n" + interface
            if initial_experience_pending:
                initial_experience_pending = False
                if claim_initial_prompt(context):
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
                parent_key = function_keys(parent_source)
                if key in seen:
                    raise ValueError("The response repeats an already considered program")
                unchanged = UNCHANGED_FUNCTION.get(intent)
                if intent == "repair" and "component" in decision:
                    unchanged = UNCHANGED_FUNCTION[decision["component"] + "_only"]
                if unchanged is not None and key[unchanged] != parent_key[unchanged]:
                    raise ValueError("Keep the other function unchanged for this paired search direction")
            except (ValueError, SyntaxError) as error:
                issue = str(error)[:400]
                write_event(folder, "invalid_proposal", parent_id=parent_id, reason=issue)
                continue

            # The bridge deduplicates this ID if the submit acknowledgement is lost.
            request_id = "e-" + hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:60]
            if any(row["request_id"] == request_id for row in history):
                issue = "That program already has an official evaluation record"
                continue
            seen.add(key)
            hypotheses = {}
            for component, function, updated, previous in zip(
                    ("loss", "decoder"), ast.parse(source).body, key, parent_key):
                if updated != previous:
                    hypotheses[component] = " ".join((ast.get_docstring(function) or "").split())[:200]
            attempt = {"parent_id": parent_id, "search_intent": intent, "changed_components": hypotheses}
            attempts[request_id] = attempt
            write_event(folder, "submitted", request_id=request_id, decision=decision, **attempt)
            result = submit_and_wait(context, source, request_id)
            feedback = summarize(result)
            history.append(result)
            write_event(folder, "evaluated", parent_id=parent_id, **feedback)
            issue = None
            if result["status"] == "succeeded":
                receipts_by_candidate[result["candidate_id"]] = result
                if best is None or score_key(result) > score_key(best):
                    best = result
                    save_best(folder, best)
                    write_event(folder, "incumbent", **summarize(best))

        if best is None:
            raise RuntimeError("No candidate completed an official training evaluation" + (f": {issue}" if issue else ""))
        write_event(folder, "selected", **summarize(best))
        return best["candidate_id"]
