"""Mocked-bridge replays of a complete ten-evaluation run.

Run against the revised Agent directory (default) or any other one:

    python3 -m unittest discover -s tests -v
    AGENT_DIR=/path/to/AgenticBench/agent/loss_decoder python3 -m unittest discover -s tests -v

The scripted world encodes what the development experiments measured: adding the
excess-weight timing pull at 1x and at 4x changed the embeddings but left the
official assignment, the crossing nets and the timing identical to the start.
The scripted model is a stubborn proposer whose habit is to retune that
coefficient. Everything else about the world is invented and only has to be
self-consistent: the tests check how the harness spends evaluations, not which
algorithm is physically better.
"""

import json
import os
from pathlib import Path
import re
import shutil
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_support as support

AGENT_DIR = Path(os.environ.get("AGENT_DIR") or Path(__file__).resolve().parents[1]
                 / "AgenticBench" / "agent" / "loss_decoder")
LOSS_EFFECT = {  # name -> (mean points, scatter, fraction of instances reassigned)
    "hard_negative": (-2.0, 9.0, 0.21),
    "all_edges": (-1.5, 6.0, 0.00015),
}
DECODER_EFFECT = {
    "flip": (2.5, 5.0, 1.0), "orient_axis": (7.0, 5.0, 0.5), "orient_count": (-3.0, 6.0, 0.5),
    "gap": (-4.0, 6.0, 0.03), "gap_flip": (-6.0, 6.0, 1.0), "loop": (1.0, 5.0, 0.35), "loop_flip": (0.5, 5.0, 0.65),
}


def identify(source):
    """Name the fixture loss and decoder inside a submitted program."""
    loss_text, decoder_text = source.split(support.DECODER_SIGNATURE)
    loss = next(name for name, body in support.LOSSES.items()
                if body.split("\n")[0] in loss_text)
    coefficient = re.search(r"return sampled \+ ([0-9.e+-]+) \*", loss_text)
    decoder = next(name for name, body in support.DECODERS.items() if body.split("\n")[0] in decoder_text)
    return loss, float(coefficient[1]) if coefficient else 0.0, decoder


def world(source, attempt):
    """What the official flow reports for a fixture program."""
    loss, coefficient, decoder = identify(source)
    if decoder in ("float_labels",):
        return {"kind": "contract"}
    points = {key: 0.0 for key in support.spread("zero", 0.0, 0.0)}
    changed = 0.0
    # A normalized timing pull below 10x never reorders groups at the area-balanced boundary.
    loss_active = loss in LOSS_EFFECT or (loss in ("excess", "slack_axis") and coefficient >= 10)
    for active, name, table in ((loss_active, loss, LOSS_EFFECT), (decoder in DECODER_EFFECT, decoder, DECODER_EFFECT)):
        if active:
            mean_points, scatter, fraction = table.get(name, (-1.0, 6.0, 0.1))
            label = name if table is DECODER_EFFECT else f"{name}:{coefficient:g}"
            for key, value in support.spread(label, mean_points, scatter).items():
                points[key] += value
            changed = max(changed, fraction)
    if loss_active and decoder in DECODER_EFFECT:  # combined changes interfere
        for key, value in support.spread(loss + "+" + decoder, -3.0, 6.0).items():
            points[key] += value
    spec = {"kind": "effective" if changed else "inert", "points": points, "changed_fraction": changed,
            "cut": int(changed * 100)}
    if (loss, decoder, attempt) == ("initial", "orient_axis", 0):  # transient infrastructure failure
        spec["not_completed"] = {("bp-ng45", "1002"): {
            "error_type": "ResourceAdmissionError",
            "reason": "insufficient available memory for candidate admission"}}
    return spec


class StubbornProposer:
    """Scripted model. Its habit is to retune the coefficient of its first loss idea."""

    def __init__(self, losses, decoders):
        self.losses, self.decoders, self.calls = iter(losses), iter(decoders), []

    def __call__(self, system, request):
        if system.startswith("Review these opening proposals"):
            return "PICK A"
        intent = request.get("search_intent")
        component = (request.get("decision") or {}).get("component")
        wants_decoder = (intent == "decoder_only" or (intent == "repair" and component == "decoder")
                         or (request.get("required_component") == "decoder"
                             and request.get("previous_generation_issue")))
        parent_loss, parent_decoder = request["parent_program"].split(support.DECODER_SIGNATURE)
        name = next(self.decoders if wants_decoder else self.losses, None)
        if name is None:
            reply = "I have no further ideas."
        elif wants_decoder:
            reply = parent_loss + support.DECODER_SIGNATURE + support.DECODERS[name].replace("{c}", "0.02")
        else:
            reply = (support.LOSS_SIGNATURE + support.LOSSES[name[0]].replace("{c}", repr(float(name[1])))
                     + "\n\n" + support.DECODER_SIGNATURE + parent_decoder)
        self.calls.append((intent, name))
        return reply


def run(losses, decoders, evaluation_limit=10, model_limit=80, world=world):
    workdir = support.make_workdir()
    try:
        agent, copy = support.load_agent(AGENT_DIR, workdir)
        bridge = support.MockBridge(workdir, world, StubbornProposer(losses, decoders), evaluation_limit, model_limit)
        returned = agent.Agent().run(bridge)
        events = [json.loads(line) for line in (copy / "search.jsonl").read_text(encoding="utf-8").splitlines()]
        return bridge, returned, events
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def timing(result):
    return tuple((cell["case"], cell["seed"], cell["metrics"]["final_global_wns_setup_ns"],
                  cell["metrics"]["final_global_tns_setup_ns"]) for cell in result["feedback"]["cells"]
                 if "metrics" in cell)


def repeats(bridge):
    """Evaluations whose complete matrix reproduces the start or an earlier complete matrix exactly."""
    known = {tuple((cell["case"], cell["seed"], *support.PUBLISHED[cell["case"]]["seeds"][str(cell["seed"])])
                   for cell in support.CELLS)}
    repeated = []
    for result in bridge.evaluations():
        if bridge.complete(result):
            if timing(result) in known:
                repeated.append(result)
            known.add(timing(result))
    return repeated


def trace(bridge):
    lines = []
    repeated = repeats(bridge)
    for index, result in enumerate(bridge.evaluations(), 1):
        loss, coefficient, decoder = identify(result["source"])
        score = result["feedback"]["score"]
        note = "REPEATS A KNOWN RESULT" if result in repeated else "" if bridge.complete(result) else "incomplete"
        lines.append(f"  evaluation {index:2d}: loss={loss + ':' + format(coefficient, 'g'):18s} decoder={decoder:12s} "
                     f"score={'%6.2f' % score if score is not None else '  none'} {note}")
    return "\n".join(lines)


START_SCORE = support.mean(
    support.mean(support.score_metrics(case, {"final_global_wns_setup_ns": wns, "final_global_tns_setup_ns": tns})
                 ["combined_score"] for wns, tns in entry["seeds"].values())
    for case, entry in sorted(support.PUBLISHED.items()))


class ReplayTenEvaluations(unittest.TestCase):
    """One complete formal-sized run against a stubborn proposer."""

    @classmethod
    def setUpClass(cls):
        cls.bridge, cls.returned, cls.events = run(
            losses=[("excess", 1), ("excess", 4), ("excess", 2), ("excess", 8), ("hard_negative", 0.25),
                    ("all_edges", 4), ("excess", 3), ("excess", 6), ("slack_axis", 1), ("excess", 5),
                    ("excess", 7), ("hard_negative", 1.0), ("excess", 9), ("all_edges", 8)],
            decoders=["flip", "orient_axis", "gap", "loop", "orient_count", "loop_flip", "gap_flip"])
        cls.trace = "\n" + trace(cls.bridge)
        if os.environ.get("REPLAY_TRACE"):
            print(f"\n{AGENT_DIR}\n  published start score {START_SCORE:.2f}{cls.trace}\n  returned {cls.returned}, "
                  f"model calls {cls.bridge.model_calls}")

    def test_evaluations_are_not_spent_on_candidates_that_reproduce_the_parent(self):
        self.assertLessEqual(len(repeats(self.bridge)), 2, self.trace)

    def test_a_reviewed_out_or_infrastructure_stopped_mechanism_is_still_measured(self):
        # The proposer wrote the strongest mechanism of this world as its second decoder idea.
        measured = [result for result in self.bridge.evaluations()
                    if identify(result["source"])[2] == "orient_axis" and self.bridge.complete(result)]
        self.assertTrue(measured, self.trace)

    def test_returned_candidate_is_the_best_complete_verified_matrix(self):
        complete = [result for result in self.bridge.evaluations() if self.bridge.complete(result)]
        best = max(complete, key=lambda result: result["feedback"]["score"])
        self.assertEqual(self.returned, best["candidate_id"], self.trace)
        self.assertGreater(best["feedback"]["score"], START_SCORE + 5, self.trace)

    def test_budgets_are_respected(self):
        self.assertEqual(len(self.bridge.evaluations()), 10, self.trace)
        self.assertLess(self.bridge.model_calls, self.bridge.model_limit, self.trace)


class IncompleteMatrix(unittest.TestCase):
    def test_a_candidate_with_an_unfinished_cell_is_never_the_final_answer(self):
        def scripted(source, attempt):
            spec = world(source, attempt)
            if identify(source)[0] == "hard_negative":  # its weakest cell never finishes, for an unknown reason
                spec["points"] = support.spread("lucky", 9.0, 2.0)
                spec["not_completed"] = {("bp-ng45", "1002"): {"reason": "cell did not finish"}}
            return spec

        bridge, returned, _ = run([("hard_negative", 0.25)], ["flip"], evaluation_limit=2, world=scripted)
        by_id = {result["candidate_id"]: result for result in bridge.evaluations()}
        self.assertTrue(bridge.complete(by_id[returned]), "\n" + trace(bridge))


@unittest.skipUnless(support.importlib.util.find_spec("numpy"), "the decoder probe needs NumPy")
class DecoderContract(unittest.TestCase):
    def test_broken_or_equivalent_decoders_never_reach_the_official_evaluator(self):
        bridge, _, events = run([("hard_negative", 0.25)], ["float_labels", "noop", "lopsided", "flip"],
                                evaluation_limit=2)
        submitted = {identify(result["source"])[2] for result in bridge.evaluations()}
        self.assertFalse(submitted & {"float_labels", "noop", "lopsided"}, "\n" + trace(bridge))
        self.assertIn("flip", submitted, "\n" + trace(bridge))


if __name__ == "__main__":
    unittest.main()
