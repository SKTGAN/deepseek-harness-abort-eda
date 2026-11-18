"""Summarize paired training feedback without changing official scores."""

from collections import defaultdict
import gzip
import json
from math import isfinite
from pathlib import Path
import re
from statistics import fmean, median


PHYSICAL_STAGE = re.compile(r"\b(?:20_partition|30_placement|40_pdn|50_hbt|60_opt)\b")
RESOURCE_ERRORS = (
    "insufficient available memory", "insufficient free disk", "insufficient gpu memory",
    "no space left on device", "cannot connect to the docker daemon", "resource admission timed out",
)
PARTITION_FIELDS = ("partition_assignment_identical", "partition_assignment_changed_count",
                    "partition_assignment_changed_fraction", "tp_gnn_repair_moved_area_fraction")


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
        return value
    return None


def _delta(after, before):
    if after is None or before is None:
        return None
    return after - before


def _metric_delta(after, before, key):
    return _delta(_number(after.get(key)), _number(before.get(key)))


def _cell_key(cell):
    return cell["case"], str(cell["seed"])


def _index(cells):
    index = {}
    for cell in cells:
        key = _cell_key(cell)
        if key in index:
            raise ValueError(f"Duplicate case/seed in evaluator feedback: {key}")
        index[key] = cell
    return index


def _slack_delta(new, old, metric):
    before = _number(old.get(metric))
    after = _number(new.get(metric))
    if before is None or after is None:
        return None
    return min(after, 0) - min(before, 0)


def _reason(cell):
    return str(cell.get("reason") or cell.get("error_message") or "")[:240] or None


def published_partition_changes(result, evaluation_root):
    """Read extra diagnostics from the organizer's published cell result."""
    request_id = result["request_id"]
    if result["status"] != "succeeded" or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id):
        return {}
    output = Path(evaluation_root) / request_id
    if result.get("artifacts_dir") != str(output):
        return {}
    changes = {}
    for cell in result["feedback"]["cells"]:
        case, seed = _cell_key(cell)
        if cell.get("status") != "succeeded" or not re.fullmatch(r"[A-Za-z0-9_-]+", case) or not seed.isdigit():
            continue
        path = output / "cells" / case / seed / "result/evaluation_result.json"
        try:
            if path.is_file():
                with path.open("rb") as stream:
                    native = json.loads(stream.read(2 * 1024 * 1024 + 1))
            else:
                with gzip.open(path.with_suffix(".json.gz"), "rb") as stream:
                    native = json.loads(stream.read(2 * 1024 * 1024 + 1))
            metrics = native.get("metrics", {})
            count = metrics.get("partition_assignment_changed_count")
        except (OSError, ValueError, AttributeError, EOFError):
            continue
        if type(count) is int and count >= 0 and metrics.get("partition_assignment_identical") == int(count == 0):
            changes[case, seed] = {key: metrics[key] for key in PARTITION_FIELDS if _number(metrics.get(key)) is not None}
    return changes


def failure_evidence(cell):
    if cell.get("status") == "succeeded":
        return None
    excerpts = [{"source": str(row.get("source", ""))[-120:], "excerpt": str(row.get("excerpt", ""))[-700:]}
                for row in (cell.get("diagnostics") or [])[:2]]
    reason = _reason(cell)
    if cell.get("status") == "not_completed" and not excerpts and not cell.get("error_type"):
        return None
    evidence = " ".join([reason or "", *(row["source"] + " " + row["excerpt"] for row in excerpts)])
    lowered = evidence.lower()
    stage = PHYSICAL_STAGE.search(lowered)
    kind = "unclassified"
    component = None
    if any(message in lowered for message in RESOURCE_ERRORS):
        kind = "runtime_environment"
    elif "decoder must return one binary integer label per group" in lowered:
        kind, component = "candidate_contract", "decoder"
    return {"kind": kind, "stage": stage[0] if stage else None, "component": component,
            "error_type": cell.get("error_type"), "reason": reason, "diagnostics": excerpts}


def _largest_regression(row):
    if row is None or row["score_delta"] >= 0:
        return None
    result = {key: row[key] for key in ("case", "seed", "score_delta")}
    wns, tns = (row[metric + "_deficit_improvement_delta"] for metric in ("wns", "tns"))
    contributions = {metric: round(delta * weight, 4)
                     for metric, delta, weight in (("wns", wns, 40), ("tns", tns, 60)) if delta is not None}
    if contributions:
        result["raw_score_contribution_deltas"] = contributions
    return result


def related_experiment(focus, history, recent_ids):
    if focus is None:
        return None
    cells = focus["cells"]
    failed = next((cell for cell in cells if "failure" in cell), None)
    regression = (focus.get("parent_comparison") or {}).get("largest_regression")
    scored = [cell for cell in cells if _number(cell.get("score")) is not None]
    target = failed or regression or min(scored, key=lambda cell: cell["score"], default=None)
    if target is None:
        return None
    case, seed = _cell_key(target)
    target_stage = (target.get("failure") or {}).get("stage")
    best_priority, best = None, None
    for trial in history:
        if trial["request_id"] == focus["request_id"]:
            continue
        pairs = _index((trial.get("parent_comparison") or {}).get("cells", []))
        for cell in trial["cells"]:
            failure = cell.get("failure") or {}
            same_stage = bool(target_stage and failure.get("stage") == target_stage)
            same_case = cell["case"] == case
            if not (same_stage or same_case):
                continue
            delta = pairs.get(_cell_key(cell), {}).get("score_delta")
            if _number(cell.get("score")) is None and not failure:
                continue
            match = (same_stage, same_case and str(cell["seed"]) == seed, same_case,
                     trial["request_id"] not in recent_ids, failure.get("kind") == "candidate_contract",
                     abs(delta or 0))
            if best_priority is None or match >= best_priority:
                best_priority = match
                best = {
                    "candidate_id": trial["candidate_id"], "attempt": trial.get("attempt"),
                    "focus": {"case": case, "seed": seed, "stage": target_stage},
                    "cell": {key: cell[key] for key in ("case", "seed", "score", "failure") if key in cell},
                    "cell_score_delta_vs_parent": delta,
                }
    return best


def _pair(key, old, new):
    before = old.get("metrics") or {}
    after = new.get("metrics") or {}
    old_score, new_score = _number(old.get("score")), _number(new.get("score"))
    gain = _delta(new_score, old_score)
    cuts = _metric_delta(after, before, "cut_net_count")
    weighted_cuts = _metric_delta(after, before, "timing_weighted_cut")
    wns = _slack_delta(after, before, "final_global_wns_setup_ns")
    tns = _slack_delta(after, before, "final_global_tns_setup_ns")
    return {
        "case": key[0], "seed": key[1],
        "parent_score": old_score, "candidate_score": new_score, "score_delta": gain,
        "parent_valid": old.get("valid"), "candidate_valid": new.get("valid"),
        "parent_status": old.get("status"), "candidate_status": new.get("status"),
        "parent_reason": _reason(old), "candidate_reason": _reason(new),
        "wns_slack_delta_ns": wns, "tns_slack_delta_ns": tns, "cut_nets_delta": cuts,
        "wns_deficit_improvement_delta": _metric_delta(after, before, "wns_deficit_improvement"),
        "tns_deficit_improvement_delta": _metric_delta(after, before, "tns_deficit_improvement"),
        "timing_weighted_cut_delta": weighted_cuts,
        "wns_improved_tns_regressed": wns is not None and tns is not None and wns > 0 > tns,
        "fewer_cuts_lower_score": cuts is not None and gain is not None and cuts < 0 and gain < 0,
    }


def _unscored(index, expected):
    rows = []
    for case, seed in expected:
        cell = index.get((case, seed))
        if cell is None:
            rows.append({"case": case, "seed": seed, "status": "missing"})
        elif _number(cell.get("score")) is None:
            rows.append({"case": case, "seed": seed, "status": cell.get("status"),
                         "reason": _reason(cell)})
    return rows


def compare_feedback(parent, candidate, schedule):
    """Compare official replies on identical case/seed keys; None means no parent receipt."""
    if parent is None:
        return None

    parent_metrics = parent.get("feedback") or {}
    candidate_metrics = candidate.get("feedback") or {}
    before = _index(parent_metrics.get("cells") or [])
    after = _index(candidate_metrics.get("cells") or [])
    expected = list(_index(schedule))
    pairs = [_pair(key, before[key], after[key]) for key in expected if key in before and key in after]
    scored = [row for row in pairs if row["score_delta"] is not None]
    complete = (bool(expected) and len(scored) == len(expected)
                and before.keys() == after.keys() == set(expected)
                and parent.get("status") in ("succeeded", "published_reference")
                and candidate.get("status") == "succeeded")

    seen_by_case = defaultdict(list)
    for row in scored:
        seen_by_case[row["case"]].append(row["score_delta"])
    expected_by_case = defaultdict(int)
    for case, _seed in expected:
        expected_by_case[case] += 1

    parent_score = _number(parent_metrics.get("score"))
    candidate_score = _number(candidate_metrics.get("score"))
    best = max(scored, key=lambda row: row["score_delta"], default=None)
    worst = min(scored, key=lambda row: row["score_delta"], default=None)
    return {
        "parent_id": parent.get("candidate_id"), "candidate_id": candidate.get("candidate_id"),
        "parent_source": ("published_initial_reference" if parent.get("status") == "published_reference"
                          else "run_evaluation"),
        "complete": complete, "expected_cells": len(expected), "scored_pairs": len(scored),
        "parent_unscored": _unscored(before, expected),
        "candidate_unscored": _unscored(after, expected),
        "official_score_delta": _delta(candidate_score, parent_score) if complete else None,
        "improved_pairs": sum(row["score_delta"] > 0 for row in scored),
        "regressed_pairs": sum(row["score_delta"] < 0 for row in scored),
        "median_paired_score_delta": median(row["score_delta"] for row in scored) if scored else None,
        "largest_gain": {key: best[key] for key in ("case", "seed", "score_delta")} if best and best["score_delta"] > 0 else None,
        "largest_regression": _largest_regression(worst),
        "candidate_invalid_cells": sum(after[key].get("valid") is False for key in expected if key in after),
        "cases": [{"case": case, "scored_pairs": len(seen_by_case[case]),
                   "expected_cells": count,
                   "mean_paired_score_delta": fmean(seen_by_case[case]) if len(seen_by_case[case]) == count else None}
                  for case, count in expected_by_case.items()],
        "cells": pairs,
    }


def load_attempts(log_path):
    """Read lineage from this Agent's own submitted and evaluated events."""
    parents = defaultdict(set)
    details = {}
    try:
        events = log_path.read_bytes().splitlines()
    except FileNotFoundError:
        return {}
    for line in events:
        try:
            event = json.loads(line)
        except ValueError:
            continue  # A stopped process can leave a truncated line.
        if not isinstance(event, dict) or event.get("event") not in ("submitted", "evaluated"):
            continue
        request_id, parent_id = event.get("request_id"), event.get("parent_id")
        if isinstance(request_id, str) and isinstance(parent_id, str):
            parents[request_id].add(parent_id)
            if event["event"] == "submitted":
                details[request_id] = {key: event[key] for key in ("search_intent", "changed_components") if key in event}
    return {request_id: {"parent_id": next(iter(ids)), **details.get(request_id, {})}
            for request_id, ids in parents.items() if len(ids) == 1}


def comparison_for_prompt(result, parent_id, receipts_by_candidate, schedule):
    """Attach a comparison to a receipt using its own parent, even after a restart."""
    if parent_id is None:
        return None
    parent = receipts_by_candidate.get(parent_id)
    if parent is None or parent_id == result["candidate_id"]:
        return {"parent_id": parent_id, "status": "parent_receipt_unavailable"}

    comparison = compare_feedback(parent, result, schedule)
    # Absolute cell values and failure reasons already appear in the receipt summary.
    fields = ("case", "seed", "score_delta", "wns_slack_delta_ns", "tns_slack_delta_ns",
              "wns_deficit_improvement_delta", "tns_deficit_improvement_delta", "cut_nets_delta",
              "timing_weighted_cut_delta", "wns_improved_tns_regressed",
              "fewer_cuts_lower_score")
    comparison["cells"] = [{field: row[field] for field in fields}
                           for row in comparison["cells"] if row["score_delta"] is not None]
    return comparison
