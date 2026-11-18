"""Summarize paired training feedback without changing official scores."""

from collections import Counter, defaultdict
import csv
import gzip
from hashlib import sha256
import io
import json
from math import isclose, isfinite
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
DECODER_ERRORS = ("decoder must return one binary integer label per group",
                  "decoder produced nonfinite protected centers")
LOSS_ERRORS = ("evolved tp-gnn loss must return a finite scalar autograd tensor",
               "compute_tp_gnn_loss must return one scalar torch.tensor",
               "compute_tp_gnn_loss detached the training objective",
               "compute_tp_gnn_loss returned a non-finite value")
ASSIGNMENT_PATHS = ("retained_artifacts/algorithm/20_partition/partition_assignment.csv",
                    "run/20_partition/partition_assignment.csv")
TIMING_FIELDS = {"wns": "final_global_wns_setup_ns", "tns": "final_global_tns_setup_ns"}


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


def _reason(cell):
    return str(cell.get("reason") or cell.get("error_message") or "")[:240] or None


def _published_cell(result, evaluation_root, key):
    request_id = result.get("request_id")
    case, seed = key
    if (evaluation_root is None or not isinstance(request_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", case) or not seed.isascii() or not seed.isdigit()):
        return None
    root = Path(evaluation_root)
    output = root / request_id
    if result.get("artifacts_dir") != str(output):
        return None
    cell = output / "cells" / case / seed / "result"
    try:
        if cell.resolve() != root.resolve() / cell.relative_to(root):
            return None
    except (OSError, RuntimeError):
        return None
    return cell


def _published_bytes(cell, relative, max_bytes):
    if cell is None:
        return None
    path = cell / relative
    for name, compressed in ((path, False), (path.with_name(path.name + ".gz"), True)):
        try:
            if name.resolve(strict=True) != name.absolute() or not name.is_file():
                continue
            opener = gzip.open if compressed else open
            with opener(name, "rb") as stream:
                data = stream.read(max_bytes + 1)
            if len(data) <= max_bytes:
                return data
        except (OSError, ValueError, EOFError, RuntimeError):
            continue
    return None


def _published_json(cell, relative):
    data = _published_bytes(cell, relative, 2 * 1024 * 1024)
    try:
        return json.loads(data) if data is not None else None
    except (ValueError, RecursionError):
        return None


def published_partition_changes(result, evaluation_root):
    """Read initial-reference metrics only from each public native evaluation result."""
    changes = {}
    for cell in (result.get("feedback") or {}).get("cells", []):
        if cell.get("status") != "succeeded":
            continue
        key = _cell_key(cell)
        native = _published_json(_published_cell(result, evaluation_root, key), "evaluation_result.json")
        source = native.get("metrics") if isinstance(native, dict) else None
        metrics = {**(source if isinstance(source, dict) else {}), **(cell.get("metrics") or {})}
        count = metrics.get("partition_assignment_changed_count")
        fraction = _number(metrics.get("partition_assignment_changed_fraction"))
        identical = metrics.get("partition_assignment_identical")
        if (type(count) is int and count >= 0 and identical == int(count == 0)
                and (fraction is None or (0 <= fraction <= 1 and (fraction == 0) == (count == 0)))):
            changes[key] = {field: metrics[field] for field in PARTITION_FIELDS if _number(metrics.get(field)) is not None}
    return changes


def _assignment_snapshot(result, evaluation_root, key, cache):
    """Fingerprint aligned names and pack measured tier labels without exposing names."""
    cell = _published_cell(result, evaluation_root, key)
    if cell is None:
        return None
    if cell in cache:
        return cache[cell]
    snapshot = None
    for relative in ASSIGNMENT_PATHS:
        data = _published_bytes(cell, relative, 64 * 1024 * 1024)
        if data is None:
            continue
        try:
            rows = csv.reader(io.StringIO(data.decode("utf-8"), newline=""), strict=True)
            if next(rows, None) != ["inst_name", "die"]:
                continue
            assignment = {}
            for name, die in rows:
                if not name or "\x00" in name or name in assignment or die not in {"bottom", "top"}:
                    raise ValueError("Invalid published assignment")
                assignment[name] = die == "top"
            if not assignment:
                continue
            names = sha256()
            labels = bytearray()
            for name in sorted(assignment):
                names.update(name.encode("utf-8") + b"\x00")
                labels.append(assignment[name])
            snapshot = (names.digest(), int.from_bytes(labels, "little"), len(labels))
            break
        except (ValueError, csv.Error):
            continue
    cache[cell] = snapshot
    return snapshot


def _assignment_difference(key, old, new, root, initial_changes, cache):
    if old.get("status") == "published_reference":
        row = initial_changes.get(key)
        if row is None:
            return None
        count = row["partition_assignment_changed_count"]
        fraction = row.get("partition_assignment_changed_fraction")
        return {"source": "published_native_vs_initial", "changed_count": count,
                "changed_fraction": fraction, "identical": count == 0,
                "flip_adjusted_changed_fraction": (min(fraction, 1 - fraction) if fraction is not None else None),
                "repair_moved_area_fraction": row.get("tp_gnn_repair_moved_area_fraction")}
    before = _assignment_snapshot(old, root, key, cache)
    after = _assignment_snapshot(new, root, key, cache)
    if before is None or after is None or before[::2] != after[::2]:
        return None
    changed = bin(before[1] ^ after[1]).count('1')
    total = before[2]
    flip_adjusted = min(changed, total - changed)
    return {"source": "published_aligned_assignments", "changed_count": changed,
            "changed_fraction": changed / total, "identical": changed == 0,
            "flip_adjusted_changed_count": flip_adjusted,
            "flip_adjusted_changed_fraction": flip_adjusted / total}


def failure_evidence(cell):
    status = cell.get("status")
    if status in ("succeeded", "published_reference"):
        return None
    excerpts = [{"source": str(row.get("source", ""))[-120:], "excerpt": str(row.get("excerpt", ""))[-440:]}
                for row in (cell.get("diagnostics") or [])[:2] if isinstance(row, dict)]
    reason = _reason(cell)
    if status in (None, "missing", "not_completed") and not excerpts and not cell.get("error_type"):
        return None
    lowered = " ".join([str(cell.get("reason") or cell.get("error_message") or ""),
                         *(row["source"] + " " + row["excerpt"] for row in excerpts)]).lower()
    stage = PHYSICAL_STAGE.search(lowered)
    kind = "unclassified"
    component = None
    if any(message in lowered for message in DECODER_ERRORS):
        kind, component = "candidate_contract", "decoder"
    elif any(message in lowered for message in LOSS_ERRORS):
        kind, component = "candidate_contract", "loss"
    elif any(message in lowered for message in RESOURCE_ERRORS):
        kind = "runtime_environment"
    elif status == "penalized":
        kind = "physical_penalty"
    elif status == "uncertain" or cell.get("error_type") == "UncertainExecution":
        kind = "execution_uncertain"
    return {"kind": kind, "stage": stage[0] if stage else None, "component": component,
            "error_type": cell.get("error_type"), "reason": reason, "diagnostics": excerpts}


def _largest_regression(row):
    if row is None or row["score_delta"] >= 0:
        return None
    result = {key: row[key] for key in ("case", "seed", "score_delta")}
    terms = row["score_delta_terms"]
    if terms is not None:
        result["raw_score_contribution_deltas"] = {metric: round(terms[metric + "_points"], 4)
                                                  for metric in TIMING_FIELDS}
        result["score_floor_delta_points"] = round(terms["score_floor_delta_points"], 4)
    return result


def _mechanism_match(focus, trial):
    left, right = (row.get("attempt") or {} for row in (focus, trial))
    left_keys, right_keys = (row.get("mechanism_keys") or {} for row in (left, right))
    left_notes, right_notes = (row.get("changed_components") or {} for row in (left, right))
    matched = []
    for component in left_keys.keys() & right_keys.keys():
        a, b = left_keys[component], right_keys[component]
        a_notes, b_notes = left_notes.get(component), right_notes.get(component)
        families = a.get("added_signal_families")
        if a.get("shape") and a["shape"] == b.get("shape"):
            matched.append((3, component + ":same_structure"))
        elif isinstance(a_notes, dict) and isinstance(b_notes, dict) and (
                a_notes.get("mechanism") and a_notes["mechanism"].casefold() == (b_notes.get("mechanism") or "").casefold()):
            matched.append((2, component + ":same_reported_mechanism"))
        elif families and families == b.get("added_signal_families"):
            matched.append((1, component + ":same_added_inputs"))
    return max(matched, default=(0, None))


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
        mechanism_priority, mechanism_match = _mechanism_match(focus, trial)
        pairs = _index((trial.get("parent_comparison") or {}).get("cells", []))
        for cell in trial["cells"]:
            failure = cell.get("failure") or {}
            same_stage = bool(target_stage and failure.get("stage") == target_stage)
            same_case = cell["case"] == case
            if not (same_stage or same_case or mechanism_priority):
                continue
            pair = pairs.get(_cell_key(cell), {})
            delta = pair.get("score_delta")
            if _number(cell.get("score")) is None and not failure:
                continue
            match = (same_stage, mechanism_priority, same_case and str(cell["seed"]) == seed, same_case,
                     trial["request_id"] not in recent_ids, failure.get("kind") == "candidate_contract",
                     abs(delta or 0))
            if best_priority is None or match >= best_priority:
                best_priority = match
                best = {
                    "candidate_id": trial["candidate_id"], "attempt": trial.get("attempt"),
                    "mechanism_match": mechanism_match,
                    "focus": {"case": case, "seed": seed, "stage": target_stage},
                    "cell": {key: cell[key] for key in ("case", "seed", "score", "failure") if key in cell},
                    "cell_score_delta_vs_parent": delta,
                    "paired_evidence": {key: pair[key] for key in ("score_delta_terms", "assignment", "observation")
                                        if pair.get(key) is not None},
                }
    return best


def _score_terms(cell):
    """Return native timing terms only if they reproduce the official clipped score."""
    if cell.get("status") not in {"succeeded", "published_reference"} or cell.get("valid") is False:
        return None
    metrics = cell.get("metrics") or {}
    score = _number(cell.get("score"))
    wns = _number(metrics.get("wns_deficit_improvement"))
    tns = _number(metrics.get("tns_deficit_improvement"))
    if score is None or wns is None or tns is None:
        return None
    raw_score = 100 * (0.4 * wns + 0.6 * tns)
    declared_raw = _number(cell.get("raw_score", metrics.get("raw_score")))
    if (not isfinite(raw_score) or not isclose(score, max(-100.0, raw_score), abs_tol=1e-8)
            or (declared_raw is not None and not isclose(declared_raw, raw_score, abs_tol=1e-8))):
        return None
    return 40 * wns, 60 * tns, score - raw_score


def _cell_failure(result, cell, key, evaluation_root):
    failure = failure_evidence(cell)
    # The successful matrix endpoint includes penalized cells but omits their diagnostics.
    if cell.get("status") == "penalized" and not (failure or {}).get("diagnostics") and evaluation_root is not None:
        directory = _published_cell(result, evaluation_root, key)
        diagnostics = _published_json(directory, "retained_artifacts/diagnostics.json")
        if isinstance(diagnostics, list) and diagnostics:
            return failure_evidence({**cell, "diagnostics": diagnostics})
    return failure


def _pair(key, old, new, assignment, failure):
    before = old.get("metrics") or {}
    after = new.get("metrics") or {}
    old_score, new_score = _number(old.get("score")), _number(new.get("score"))
    gain = _delta(new_score, old_score)
    cuts = _metric_delta(after, before, "cut_net_count")
    weighted_cuts = _metric_delta(after, before, "timing_weighted_cut")
    wns = _metric_delta(after, before, TIMING_FIELDS["wns"])
    tns = _metric_delta(after, before, TIMING_FIELDS["tns"])
    old_terms, new_terms = _score_terms(old), _score_terms(new)
    terms = (dict(zip(("wns_points", "tns_points", "score_floor_delta_points"),
                      (new - old for new, old in zip(new_terms, old_terms))))
             if old_terms is not None and new_terms is not None else None)
    timing_unchanged = (abs(wns) < 1e-6 and abs(tns) < 1e-6) if wns is not None and tns is not None else None
    timing_changed = any(value is not None and abs(value) >= 1e-6 for value in (wns, tns))
    candidate_status, parent_status = new.get("status", "missing"), old.get("status", "missing")
    observation = None
    if assignment is not None:
        if assignment["identical"] and timing_unchanged is True:
            observation = "same_assignments_and_timing"
        elif assignment["identical"] and timing_changed:
            observation = "timing_changed_with_same_assignments"
        elif assignment.get("flip_adjusted_changed_fraction") == 0 and not assignment["identical"]:
            observation = "all_tier_labels_flipped"
    return {
        "case": key[0], "seed": key[1],
        "parent_score": old_score, "candidate_score": new_score, "score_delta": gain,
        "parent_valid": old.get("valid"), "candidate_valid": new.get("valid"),
        "parent_status": parent_status, "candidate_status": candidate_status,
        "parent_reason": _reason(old), "candidate_reason": _reason(new),
        "wns_slack_delta_ns": wns, "tns_slack_delta_ns": tns, "cut_nets_delta": cuts,
        "wns_deficit_improvement_delta": _metric_delta(after, before, "wns_deficit_improvement"),
        "tns_deficit_improvement_delta": _metric_delta(after, before, "tns_deficit_improvement"),
        "score_delta_terms": terms,
        "timing_weighted_cut_delta": weighted_cuts,
        "wns_improved_tns_regressed": wns is not None and tns is not None and wns > 0 > tns,
        "fewer_cuts_lower_score": cuts is not None and gain is not None and cuts < 0 and gain < 0,
        "assignment": assignment, "observation": observation, "failure": failure,
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


def compare_feedback(parent, candidate, schedule, *, evaluation_root=None, partition_changes=None,
                     assignment_cache=None):
    """Compare official replies on identical case/seed keys; None means no parent receipt."""
    if parent is None:
        return None

    parent_metrics = parent.get("feedback") or {}
    candidate_metrics = candidate.get("feedback") or {}
    before = _index(parent_metrics.get("cells") or [])
    after = _index(candidate_metrics.get("cells") or [])
    expected = list(_index(schedule))
    if partition_changes is None:
        partition_changes = (published_partition_changes(candidate, evaluation_root)
                             if parent.get("status") == "published_reference" else {})
    cache = assignment_cache if assignment_cache is not None else {}
    pairs = []
    for key in expected:
        old, new = before.get(key, {}), after.get(key, {})
        assignment = None
        if new.get("status") == "succeeded" and old.get("status") in {"succeeded", "published_reference"}:
            assignment = _assignment_difference(key, parent, candidate, evaluation_root, partition_changes, cache)
        failure = _cell_failure(candidate, new, key, evaluation_root)
        pairs.append(_pair(key, old, new, assignment, failure))
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
    observations = Counter(row["observation"] for row in pairs)
    discrepancies = [row for row in pairs if row["observation"] == "timing_changed_with_same_assignments"]
    discrepancies.sort(key=lambda row: abs(row["score_delta"] or 0), reverse=True)
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
        "failure_counts": dict(Counter(row["failure"]["kind"] for row in pairs if row["failure"] is not None)),
        "assignment_pairs_verified": sum(row["assignment"] is not None for row in pairs),
        "unchanged_assignment_and_timing_pairs": observations["same_assignments_and_timing"],
        "timing_changes_with_identical_assignment": [
            {key: row[key] for key in ("case", "seed", "score_delta", "wns_slack_delta_ns", "tns_slack_delta_ns")}
            for row in discrepancies[:3]],
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
                details[request_id] = {
                    key: event[key] for key in ("search_intent", "changed_components", "mechanism_keys", "model_calls", "policy_action")
                    if key in event
                }
    return {request_id: {"parent_id": next(iter(ids)), **details.get(request_id, {})}
            for request_id, ids in parents.items() if len(ids) == 1}


def comparison_for_prompt(result, parent_id, receipts_by_candidate, schedule, *, evaluation_root=None,
                          partition_changes=None, assignment_cache=None):
    """Attach a comparison to a receipt using its own parent, even after a restart."""
    if parent_id is None:
        return None
    parent = receipts_by_candidate.get(parent_id)
    if parent is None or parent_id == result["candidate_id"]:
        return {"parent_id": parent_id, "status": "parent_receipt_unavailable"}

    comparison = compare_feedback(parent, result, schedule, evaluation_root=evaluation_root,
                                  partition_changes=partition_changes, assignment_cache=assignment_cache)
    # Null means that the official output did not verify this measurement/comparison.
    fields = ("case", "seed", "candidate_status", "parent_status", "candidate_valid", "parent_valid", "candidate_score", "parent_score",
              "score_delta", "wns_slack_delta_ns", "tns_slack_delta_ns", "score_delta_terms",
              "wns_deficit_improvement_delta", "tns_deficit_improvement_delta", "cut_nets_delta",
              "timing_weighted_cut_delta", "wns_improved_tns_regressed", "fewer_cuts_lower_score",
              "assignment", "observation", "failure")
    comparison["cells"] = [{field: round(row[field], 6) if type(row[field]) is float else row[field]
                            for field in fields} for row in comparison["cells"]]
    return comparison
