"""Compact evidence from official replies of the current training session."""

from collections import Counter
from math import isfinite


TIMING = ("final_global_wns_setup_ns", "final_global_tns_setup_ns")
TIME_EPS = 1e-6
SCORE_EPS = 1e-8


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _keys(cells):
    keys = [(row["case"], str(row["seed"])) for row in cells
            if isinstance(row, dict) and row.get("case") and row.get("seed") is not None]
    if (len(keys) != 9 or len(cells) != 9 or len(set(keys)) != 9
            or sorted(Counter(case for case, seed in keys).values()) != [3, 3, 3]):
        return None
    return keys


def _signature(summary, schedule):
    if not summary or summary.get("status") != "succeeded":
        return None
    cells = summary.get("cells") or []
    keys = _keys(cells)
    if not keys or (schedule is not None and set(keys) != set(_keys(schedule) or [])):
        return None
    signature = {}
    for key, cell in zip(keys, cells):
        metrics = cell.get("metrics") or {}
        values = (cell.get("score"),) + tuple(metrics.get(name) for name in TIMING)
        if (cell.get("status") != "succeeded" or cell.get("valid") != True
                or cell.get("failure") or not all(_finite(value) for value in values)):
            return None
        signature[key] = values
    return signature


def timing_twins(left, right, schedule=None):
    """Same nine measured scores and raw timing values; implies nothing about assignment."""
    before, after = _signature(left, schedule), _signature(right, schedule)
    return bool(before and after and before.keys() == after.keys() and all(
        all(abs(first - second) <= tolerance for first, second, tolerance
            in zip(before[key], after[key], (SCORE_EPS, TIME_EPS, TIME_EPS))) for key in before))


def _worst_detail(row):
    result = {key: row.get(key) for key in (
        "case", "seed", "score_delta", "parent_score", "candidate_score", "parent_status", "candidate_status")}
    terms = row.get("score_delta_terms")
    if isinstance(terms, dict):
        result["points"] = {key: terms[key] for key in (
            "wns_points", "tns_points", "score_floor_delta_points") if _finite(terms.get(key))}
    if row.get("candidate_valid") and row.get("parent_valid"):
        for key in ("wns_slack_delta_ns", "tns_slack_delta_ns"):
            if _finite(row.get(key)):
                result[key] = row[key]
    return result


def _parent_effect(comparison):
    rows = comparison.get("cells") or []
    if not comparison.get("complete") or not _keys(rows) or not all(_finite(row.get("score_delta")) for row in rows):
        return {"complete": False}
    gains = [row for row in rows if row["score_delta"] > SCORE_EPS]
    losses = [row for row in rows if row["score_delta"] < -SCORE_EPS]
    assignments = [row["assignment"] for row in rows if (row.get("assignment") or {}).get("source") in (
        "published_native_vs_initial", "published_aligned_assignments")]
    identical = sum(record.get("identical") is True for record in assignments)
    flips = sum(record.get("identical") is False and record.get("flip_adjusted_changed_fraction") == 0
                for record in assignments)
    return {"complete": True, "gain_cells": len(gains), "loss_cells": len(losses),
            "tied_score_cells": len(rows) - len(gains) - len(losses),
            "gains_in_matrix_points": sum(row["score_delta"] for row in gains) / len(rows),
            "losses_in_matrix_points": sum(row["score_delta"] for row in losses) / len(rows),
            "worst_loss": _worst_detail(min(losses, key=lambda row: row["score_delta"])) if losses else None,
            "verified_assignment_cells": len(assignments), "identical_assignment_cells": identical,
            "all_labels_flipped_cells": flips,
            "all_nine_assignments_verified_identical": identical == len(rows)}


def _shapes(attempt):
    return {(component, row["shape"]) for component, row in (attempt.get("mechanism_keys") or {}).items()
            if isinstance(row, dict) and row.get("shape")}


def decision_evidence(focus, history, schedule=None):
    """Summarize the latest measured outcome, its parent, and one relevant earlier trial."""
    if not focus:
        return {}
    attempt = focus.get("attempt") or {}
    focus_id = focus.get("request_id")
    parent = attempt.get("parent_id")
    shapes = _shapes(attempt)
    matches = []
    twins = []
    for index, record in enumerate(history):
        if record is focus or (focus_id and record.get("request_id") == focus_id):
            break
        if not record.get("candidate_id") or record["candidate_id"] == focus.get("candidate_id"):
            continue
        old = record.get("attempt") or {}
        if timing_twins(record, focus, schedule):
            priority, reason = 4, "same_nine_measured_timings_and_scores"
            twins.append(record["candidate_id"])
        elif shapes & _shapes(old):
            priority, reason = 3, "same_edited_mechanism_shape"
        elif parent and parent == old.get("parent_id"):
            priority, reason = 2, "same_parent"
        elif set(attempt.get("changed_components") or {}) & set(old.get("changed_components") or {}):
            priority, reason = 1, "same_edited_component"
        else:
            continue
        matches.append((priority, index, reason, record))
    related = None
    if matches:
        priority, index, reason, old = max(matches, key=lambda match: match[:2])
        related = {"reason": reason, "candidate_id": old["candidate_id"], "score": old.get("score"),
                   "status": old.get("status"), "intent": (old.get("attempt") or {}).get("search_intent")}
    return {"candidate_id": focus.get("candidate_id"), "parent_id": parent, "timing_twin_count": len(set(twins)),
            "timing_verified_on_all_nine": _signature(focus, schedule) is not None,
            "parent_effect": _parent_effect(focus.get("parent_comparison") or {}),
            "related_prior_trial": related}
