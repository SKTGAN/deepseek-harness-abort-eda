"""Pre-submission contract probe for candidate decoders.

A decoder is deterministic NumPy code, so its contract can be checked before an
official evaluation is spent on it. The probe runs a validated candidate on
small synthetic inputs and reports only two things: a violation of the
published decoder contract, or labels identical to the parent's on every
input. It produces no score, ranks nothing and never replaces the official
evaluation. It is skipped when NumPy is unavailable in the agent container.

The stand-in for the frozen helper mirrors the published excerpt of the
official area-balanced decoder. The organizer's helper stays authoritative.
"""

import builtins
import random
import signal
import threading

try:
    import numpy as np
except Exception:  # The probe is optional; the search runs without it.
    np = None


PROBE_SEEDS = (1001, 1002, 1003, 1004, 1005, 1006)
PROBE_ITERATIONS = 50
PROBE_BALANCE = 0.5
AREA_WINDOW = (0.47, 0.53)
PROBE_SECONDS = 20
SAFE_BUILTINS = {name: getattr(builtins, name)
                 for name in ("abs", "bool", "float", "int", "len", "max", "min", "range", "round", "sum")}


def _area_constrained_labels(preference, weights, target_area):
    order = sorted(range(len(preference)), key=lambda index: (-preference[index], index))
    labels = [0] * len(preference)
    selected_area = 0.0
    for index in order:
        next_area = selected_area + max(float(weights[index]), 0.0)
        if abs(next_area - target_area) <= abs(selected_area - target_area):
            labels[index] = 1
            selected_area = next_area
    return labels


def reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target):
    array = np.asarray(embeddings.tolist(), dtype=np.float64)
    weights = np.maximum(np.asarray(node_areas.tolist(), dtype=np.float64), 1.0)
    target_area = float(weights.sum()) * balance_target
    first, second = random.Random(seed).sample(range(array.shape[0]), 2)
    centers = np.stack([array[first].copy(), array[second].copy()])
    labels = None
    for _ in range(max(1, max_iterations)):
        distance_0 = np.sum((array - centers[0]) ** 2, axis=1)
        distance_1 = np.sum((array - centers[1]) ** 2, axis=1)
        following = np.asarray(_area_constrained_labels((distance_0 - distance_1).tolist(), weights.tolist(),
                                                        target_area), dtype=np.int8)
        if labels is not None and np.array_equal(labels, following):
            break
        labels = following
        for cluster in (0, 1):
            mask = labels == cluster
            if not np.any(mask):
                raise RuntimeError("area-balanced TP-GNN clustering produced an empty cluster")
            centers[cluster] = np.average(array[mask], axis=0, weights=weights[mask])
    return np.asarray(labels.tolist(), dtype=np.int64)


def _probe_inputs(seed):
    """Unit-norm group embeddings in a narrow cone or two unequal lobes, with skewed physical areas."""
    generator = random.Random(seed)
    count, width = 240 + 37 * (seed % 5), 12 + 4 * (seed % 3)
    axis = [generator.gauss(0, 1) for _ in range(width)]
    offset = [generator.gauss(0, 1) for _ in range(width)]
    lobes = seed % 2
    rows = []
    for _ in range(count):
        side = 0.6 if lobes and generator.random() < 0.4 else 0.0
        rows.append([a + side * b + 0.25 * generator.gauss(0, 1) for a, b in zip(axis, offset)])
    embeddings = np.asarray(rows, dtype=np.float64)
    embeddings = (embeddings / np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True))).astype(np.float32)
    areas = np.asarray([20.0 * generator.lognormvariate(0, 0.8) for _ in range(count)], dtype=np.float64)
    embeddings.setflags(write=False)
    areas.setflags(write=False)
    return embeddings, areas


def _decoder(source):
    scope = {"__builtins__": SAFE_BUILTINS, "np": np, "reference_global_two_way": reference_global_two_way}
    exec(compile(source, "<candidate>", "exec"), scope)
    return scope["decode_tp_gnn_embeddings"]


def _contract_issue(labels, areas):
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    if labels.shape != areas.shape:
        return f"it returned shape {labels.shape} for {areas.shape[0]} groups"
    if not np.issubdtype(labels.dtype, np.integer):
        return f"it returned dtype {labels.dtype}; labels must be integers 0 or 1"
    if not np.all((labels == 0) | (labels == 1)):
        return "it returned values other than 0 and 1"
    share = float(areas[labels == 1].sum() / areas.sum())
    if not AREA_WINDOW[0] <= share <= AREA_WINDOW[1]:
        return (f"it put {share:.3f} of the area on the top die; outside {AREA_WINDOW[0]}-{AREA_WINDOW[1]} "
                "the protected repair rewrites the assignment, so the edit would not be tested")
    return None


class _Timeout(Exception):
    pass


def _expired(signum, frame):
    raise _Timeout()


def probe_decoder(source, parent_source):
    """Return None when the decoder may be submitted, otherwise one sentence for the model."""
    if np is None:
        return None
    timed = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGALRM, _expired) if timed else None
    try:
        if timed:
            signal.setitimer(signal.ITIMER_REAL, PROBE_SECONDS)
        candidate, parent = _decoder(source), _decoder(parent_source)
        identical = True
        for seed in PROBE_SEEDS:
            embeddings, areas = _probe_inputs(seed)
            arguments = (embeddings, areas, seed, PROBE_ITERATIONS, PROBE_BALANCE)
            try:
                labels = candidate(*arguments)
            except (NameError, _Timeout):
                return None  # The probe's sandbox is narrower than the official one: inconclusive.
            except Exception as error:
                return (f"The decoder raised {type(error).__name__} on a synthetic probe with read-only "
                        f"inputs: {str(error)[:160]}")
            issue = _contract_issue(labels, areas)
            if issue:
                return "The decoder breaks the published contract on a synthetic probe: " + issue
            try:
                identical = identical and np.array_equal(np.asarray(labels), np.asarray(parent(*arguments)))
            except Exception:
                identical = False
        if identical:
            return ("The decoder returns exactly the parent's labels on every synthetic probe, so it is "
                    "behaviorally equivalent to the parent. Change a decision that alters the labels.")
        return None
    except (_Timeout, SyntaxError, KeyError):
        return None
    finally:
        if timed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
