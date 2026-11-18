"""Select the next experiment solely from this session's ordered training receipts."""

from collections import Counter
from math import isfinite

from decision_evidence import timing_twins


SCOPES = ("loss", "decoder")
EPS = 1e-6


def _key(cell):
    return cell.get("case"), str(cell.get("seed"))


def _number(value):
    return type(value) in (int, float) and isfinite(value)


def _parent(row):
    return (row.get("attempt") or {}).get("parent_id")


def _scope(row):
    changed = (row.get("attempt") or {}).get("changed_components") or {}
    return next(iter(changed)) if len(changed) == 1 and next(iter(changed)) in SCOPES else None


def _complete(row, schedule_keys):
    cells = row.get("cells") or []
    return (row.get("status") == "succeeded" and _number(row.get("score"))
            and len(cells) == len(schedule_keys) and {_key(c) for c in cells} == schedule_keys
            and all(_number(c.get("score")) for c in cells))


def _measured(row):
    return all(c.get("valid") is True and c.get("status") == "succeeded" for c in row["cells"])


def _direction(parent, trials, complete):
    """Try a complementary edit first; avoid repeating the parent's unsuccessful scopes."""
    parent_id = parent["candidate_id"]
    penalties, attempts = Counter(), Counter()
    for child in trials:
        component = _scope(child)
        if _parent(child) != parent_id or not component:
            continue
        if child["candidate_id"] in complete and _measured(child):
            attempts[component] += 1
            if child["score"] <= parent["score"] + EPS:
                penalties[component] += 1
        elif any((c.get("failure") or {}).get("kind") == "candidate_contract" for c in child.get("cells", [])):
            penalties[component] += 1
    return min(SCOPES, key=lambda component: (penalties[component], attempts[component], component == _scope(parent)))


def _repair_evidence(row, original, schedule_keys):
    """Require cross-design gains and a specific, dominant loss on measured valid cells."""
    if not _measured(row) or not _measured(original) or row["score"] >= original["score"] - EPS:
        return None
    if not _scope(row) or (row.get("attempt") or {}).get("policy_action") == "targeted_repair":
        return None
    compared = row.get("parent_comparison") or {}
    if compared.get("parent_id") != original["candidate_id"] or not compared.get("complete"):
        return None
    before = {_key(cell): cell["score"] for cell in original["cells"]}
    after = {_key(cell): cell["score"] for cell in row["cells"]}
    deltas = {key: after[key] - before[key] for key in sorted(schedule_keys)}
    positive = [key for key in deltas if deltas[key] > EPS]
    losses = [key for key in deltas if deltas[key] < -EPS]
    if not losses or len(positive) * 3 <= len(deltas) or len({key[0] for key in positive}) < 2:
        return None
    worst = min(losses, key=deltas.get)
    if -deltas[worst] < 0.65 * sum(-deltas[key] for key in losses):
        return None
    uncertain = {_key(c) for c in compared.get("timing_changes_with_identical_assignment", [])}
    if worst in uncertain or len({key[0] for key in positive if key not in uncertain}) < 2:
        return None
    best = max(positive, key=deltas.get)
    return {"compared_parent_id": original["candidate_id"], "improving_cells": len(positive),
            "regressing_cells": len(losses), "paired_cells": len(deltas),
            "protect_gain": {"case": best[0], "seed": best[1], "score_delta": deltas[best]},
            "repair_regression": {"case": worst[0], "seed": worst[1], "score_delta": deltas[worst]}}


def select_next(trials, baseline_id, baseline_source, schedule, *, opening=("decoder_only", "loss_only"),
                incumbent_id=None):
    """Return (parent id, enforced single-function intent, model-visible decision).

    trials are chronological summarize_for_prompt dicts. baseline_source is accepted for
    caller symmetry; the planner never inspects source or assigns an unmeasured score.
    Only the caller selects the final submission. Persist decision.reason as policy_action
    on each admitted attempt so repair and exploration history survive process restarts.
    """
    schedule_keys = {_key(cell) for cell in schedule}
    if not schedule_keys or len(schedule_keys) != len(schedule):
        raise ValueError("Expected a nonempty training schedule with unique case/seed pairs")

    def choose(parent_id, component, reason, **evidence):
        return parent_id, component + "_only", {"reason": reason, "component": component, **evidence}

    if len(trials) < len(opening):
        return choose(baseline_id, opening[len(trials)].split("_")[0],
                      "opening_control" if not trials else "complementary_opening_with_first_receipt")

    complete = {row["candidate_id"]: row for row in trials if _complete(row, schedule_keys)}
    if not complete:
        return choose(baseline_id, SCOPES[len(trials) % 2], "no_complete_candidate")
    best = max(complete.values(), key=lambda row: row["score"])
    requested = complete.get(incumbent_id)
    if requested is not None and requested["score"] == best["score"]:
        best = requested
    incumbent = best["candidate_id"]
    latest = trials[-1]
    latest_id = latest["candidate_id"]
    selected = complete.get(latest_id)
    origin = complete.get(_parent(latest))
    failures = [c["failure"] for c in latest.get("cells", []) if c.get("failure")]
    if failures and all(f.get("kind") in ("runtime_environment", "execution_uncertain") for f in failures):
        return choose(incumbent, _direction(best, trials[:-1], complete), "execution_evidence_unusable",
                      failed_candidate_id=latest_id)
    broken = {f.get("component") for f in failures if f.get("kind") == "candidate_contract"}
    if len(broken) == 1 and next(iter(broken)) in SCOPES:
        component = next(iter(broken))
        safe_parent = origin if origin is not None and _measured(origin) else best
        if safe_parent["candidate_id"] == latest_id:
            return choose(baseline_id, component, "replace_invalid_component", failed_candidate_id=latest_id)
        return choose(safe_parent["candidate_id"], component, "replace_invalid_component", failed_candidate_id=latest_id)
    repair_trials = sum((row.get("attempt") or {}).get("policy_action") == "targeted_repair" for row in trials)
    if selected is not None and origin is not None and latest_id != origin["candidate_id"] and repair_trials < 2:
        risk = _repair_evidence(selected, origin, schedule_keys)
        visited = any(_parent(other) == latest_id for other in trials)
        if risk and not visited:
            return choose(latest_id, _scope(selected), "targeted_repair", incumbent_id=incumbent, **risk)

    streak = []
    for row in reversed(trials):
        if _parent(row) != incumbent or row["candidate_id"] not in complete or not _measured(row):
            break
        if row["score"] > best["score"] + EPS:
            break
        streak.append(row["candidate_id"])
    twin = (origin is not None and selected is not None and _parent(latest) == incumbent
            and timing_twins(latest, best, schedule))
    if twin or len(streak) >= 2:
        alternatives = [row for cid, row in complete.items()
                        if cid != incumbent and _measured(row) and not timing_twins(row, best, schedule)]
        branch_trials = [row for row in trials if (row.get("attempt") or {}).get("policy_action") == "switch_branch"]
        def priority(row):
            visits = sum(_parent(previous) == row["candidate_id"] or
                         (_parent(previous) in complete and timing_twins(complete[_parent(previous)], row, schedule))
                         for previous in branch_trials)
            return visits, -row["score"]
        result = {"incumbent_id": incumbent, "stalled_children": list(reversed(streak)),
                  "trigger": "same_measured_timing" if twin else "two_unsuccessful_children"}
        if alternatives:
            branch = min(alternatives, key=priority)
            return choose(branch["candidate_id"], _direction(branch, trials, complete), "switch_branch", **result)
        previous_scope = _scope(latest)
        return choose(baseline_id, "decoder" if previous_scope == "loss" else "loss",
                      "reopen_official_start", **result)

    return choose(incumbent, _direction(best, trials, complete), "extend_best", incumbent_id=incumbent)
