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
from local_tasks.tp_gnn_local_task_definitions import DECODER_TASK, validate_program


EXPERIENCE_HEADING = "## 离线专家知识"
UNCHANGED_FUNCTION = {"loss_only": 1, "decoder_only": 0}


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


def select_parent(history, best):
    if len(history) < 2:
        return None, ("loss_only", "decoder_only")[len(history)]
    if best is None:
        return None, "diversify"

    latest_best = next(index for index, result in enumerate(history)
                       if result["request_id"] == best["request_id"])
    rounds_without_gain = len(history) - latest_best - 1
    if rounds_without_gain and rounds_without_gain % 2 == 0:
        alternatives = [result for result in history if result["status"] == "succeeded"
                        and result["candidate_id"] != best["candidate_id"]]
        return max(alternatives, key=score_key, default=None), "diversify"
    return best, "refine"


def claim_initial_prompt(context):
    workspace = Path(context.files.get("workspaces", "/workspace/workspaces"))
    try:
        with (workspace / "loss-decoder-initial-prompt-sent").open("x", encoding="utf-8"):
            return True
    except FileExistsError:
        return False


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
        "cells": [
            {**{key: cell[key] for key in ("case", "seed", "score", "valid", "status", "metrics") if key in cell},
             "reason": str(cell.get("reason") or cell.get("error_message") or "")[:240]}
            for cell in cells
        ],
    }


def write_event(folder, event, **fields):
    with (folder / "search.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": event, **fields}, ensure_ascii=False) + "\n")


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
        task = Path(context.files["framework"]) / "local_tasks" / DECODER_TASK
        initial = validate_program((task / "initial_program.py").read_text(encoding="utf-8"), DECODER_TASK)
        instructions, marker, experience = (folder / "prompt.md").read_text(encoding="utf-8").partition(EXPERIENCE_HEADING)
        interface = (task / "prompt.md").read_text(encoding="utf-8")

        state = context.status()
        best = None
        history = []
        seen = {function_keys(initial)}
        for row in state["history"]:
            result = context.wait_evaluation(row["request_id"])
            history.append(result)
            if result["status"] == "succeeded":
                seen.add(function_keys(result["source"]))
                if best is None or score_key(result) > score_key(best):
                    best = result

        issue = None
        initial_experience_pending = bool(marker and not history)
        consecutive_api_failures = 0
        while True:
            state = context.status()
            if not state["evaluation_dispatch_allowed"] or not state["model_dispatch_allowed"]:
                write_event(folder, "budget_stop", evaluations_remaining=state["evaluations_remaining"],
                            model_stop_reasons=state["model_stop_reasons"])
                break

            parent, intent = select_parent(history, best)
            parent_id = parent["candidate_id"] if parent else context.baseline_candidate
            parent_source = parent["source"] if parent else initial
            previous_results = [summarize(result) for result in history[-2:]
                                if result is not parent and result is not best]
            request = {
                "search_intent": intent,
                "parent_id": parent_id,
                "parent_program": parent_source,
                "training_cells": context.evaluation_cells,
                "evaluations_remaining": state["evaluations_remaining"],
                "parent_feedback": summarize(parent) if parent else None,
                "incumbent_feedback": summarize(best) if best and best is not parent else None,
                "recent_feedback": previous_results,
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
                if key in seen:
                    raise ValueError("The response repeats an already considered program")
                unchanged = UNCHANGED_FUNCTION.get(intent)
                if unchanged is not None and key[unchanged] != function_keys(parent_source)[unchanged]:
                    raise ValueError("Keep the other function unchanged for this paired search direction")
            except (ValueError, SyntaxError) as error:
                issue = str(error)[:400]
                write_event(folder, "invalid_proposal", parent_id=parent_id, reason=issue)
                continue

            # The bridge deduplicates this ID if the submit acknowledgement is lost.
            request_id = "e-" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:60]
            if any(row["request_id"] == request_id for row in history):
                issue = "That program already has an official evaluation record"
                continue
            seen.add(key)
            write_event(folder, "submitted", parent_id=parent_id, request_id=request_id, search_intent=intent)
            result = submit_and_wait(context, source, request_id)
            feedback = summarize(result)
            history.append(result)
            write_event(folder, "evaluated", parent_id=parent_id, **feedback)
            issue = None
            if result["status"] == "succeeded" and (best is None or score_key(result) > score_key(best)):
                best = result

        if best is None:
            raise RuntimeError("No candidate completed an official training evaluation" + (f": {issue}" if issue else ""))
        pending = folder / "program.py.pending"
        pending.write_text(best["source"], encoding="utf-8")
        pending.replace(folder / "program.py")
        write_event(folder, "selected", **summarize(best))
        return best["candidate_id"]
