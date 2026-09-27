"""
metrics.py
----------
Evaluation metrics for the shared harness, computed on candidate-id orderings
so every method is scored the same way (SPEC section 10).

`spearman` here reproduces `evaluator._line_spearman` exactly for well-formed
rankings — Pearson correlation on rank positions, rounded to 2 decimal places.
`tests/test_harness_identity.py` asserts that equivalence on the real instance
set rather than assuming it.

These metrics measure agreement with the oracle, not absolute correctness. The
oracle encodes designer-chosen weights; see oracle.py.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean(values: Iterable[float]) -> Optional[float]:
    """Mean, or None for an empty sequence. Rounded to 4 dp."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _positions(order: Sequence[int]) -> Dict[int, int]:
    """candidate id -> 1-based position."""
    return {cid: i for i, cid in enumerate(order, start=1)}


def _aligned(
    order: Sequence[int], reference: Sequence[int]
) -> Tuple[List[int], List[int]]:
    """
    Rank pairs for the ids present in both orderings, in `order`'s sequence.
    Matches the alignment `evaluator._line_spearman` performs.
    """
    ref = _positions(reference)
    a: List[int] = []
    b: List[int] = []
    for pos, cid in enumerate(order, start=1):
        if cid in ref:
            a.append(pos)
            b.append(ref[cid])
    return a, b


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------

def spearman(order: Sequence[int], reference: Sequence[int]) -> float:
    """
    Spearman rho between two orderings, in [-1, 1].

    Deliberately identical in formula and rounding to
    `evaluator._line_spearman` so harness numbers are comparable with anything
    the pre-existing code recorded.
    """
    a, b = _aligned(order, reference)
    n = len(a)
    if n < 2:
        return 0.0

    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0 or db == 0:
        return 0.0
    return round(num / (da * db), 2)


def kendall_tau_b(order: Sequence[int], reference: Sequence[int]) -> float:
    """
    Kendall's tau-b, in [-1, 1], tie-corrected.

    tau_b = (C - D) / sqrt((C + D + Ta) * (C + D + Tb))

    where C and D are concordant and discordant pairs, Ta is pairs tied in the
    first ordering only and Tb pairs tied in the second only. Both inputs are
    strict orderings here, so Ta = Tb = 0 and this reduces to tau-a; the
    correction is kept so the metric stays valid if a method ever emits ties.
    """
    a, b = _aligned(order, reference)
    n = len(a)
    if n < 2:
        return 0.0

    concordant = discordant = tied_a = tied_b = 0
    for i, j in itertools.combinations(range(n), 2):
        da = a[i] - a[j]
        db = b[i] - b[j]
        if da == 0 and db == 0:
            continue
        if da == 0:
            tied_a += 1
        elif db == 0:
            tied_b += 1
        elif da * db > 0:
            concordant += 1
        else:
            discordant += 1

    denom = math.sqrt(
        (concordant + discordant + tied_a) * (concordant + discordant + tied_b)
    )
    if denom == 0:
        return 0.0
    return round((concordant - discordant) / denom, 4)


# ---------------------------------------------------------------------------
# Top-3
# ---------------------------------------------------------------------------

def top3_accuracy(order: Sequence[int], reference: Sequence[int]) -> float:
    """
    1.0 when the oracle's best candidate appears in the method's top 3.

    This is the direct generalisation of the pre-existing `top1_match`
    boolean, which is how the headline "top-3 accuracy" figure is computed.
    `top3_set_overlap` is reported alongside it because the phrase is
    ambiguous and the two answer different questions.
    """
    if not order or not reference:
        return 0.0
    return 1.0 if reference[0] in order[:3] else 0.0


def top3_set_overlap(order: Sequence[int], reference: Sequence[int]) -> float:
    """Fraction of the oracle's top 3 that appears in the method's top 3."""
    if not order or not reference:
        return 0.0
    k = min(3, len(reference))
    if k == 0:
        return 0.0
    return round(len(set(order[:3]) & set(reference[:k])) / k, 4)


# ---------------------------------------------------------------------------
# Run-to-run consistency
# ---------------------------------------------------------------------------

def run_consistency(orders: Sequence[Sequence[int]]) -> Optional[float]:
    """
    Mean Spearman between every pair of repetitions.

    None for fewer than two repetitions — a single run has no consistency to
    measure, and reporting 1.0 there would be misleading.
    """
    if len(orders) < 2:
        return None
    vals = [spearman(a, b) for a, b in itertools.combinations(orders, 2)]
    return mean(vals)


# ---------------------------------------------------------------------------
# Tie rate
# ---------------------------------------------------------------------------

def tie_rate(scores: Optional[Dict[int, float]]) -> Optional[float]:
    """
    Fraction of candidate pairs receiving an identical score.

    Degeneracy is a known weakness of boolean scoring, so this is a
    first-class metric (SPEC section 7). None for a method that produces no
    score at all — Rank-All emits a rank directly, so it has no tie rate and
    reporting 0.0 would flatter it.
    """
    if not scores or len(scores) < 2:
        return None
    vals = list(scores.values())
    n = len(vals)
    tied = sum(
        1 for x, y in itertools.combinations(vals, 2) if x == y
    )
    return round(tied / (n * (n - 1) / 2), 4)


# ---------------------------------------------------------------------------
# PBS bid metrics (research note, section 5)
# ---------------------------------------------------------------------------

def top_k_overlap(order: Sequence[int], reference: Sequence[int], k: int) -> float:
    """Fraction of the reference's top k that appears in the method's top k."""
    k = min(k, len(reference))
    if k == 0 or not order:
        return 0.0
    return round(len(set(order[:k]) & set(reference[:k])) / k, 4)


def selection_prf(
    selected: Iterable[str], truth: Iterable[str]
) -> Dict[str, float]:
    """
    Precision, recall and F1 of an activated-column set z_p against z*_p.

    Both empty counts as perfect; one empty and the other not as 0.
    """
    s, t = set(selected), set(truth)
    if not s and not t:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(s & t)
    p = tp / len(s) if s else 0.0
    r = tp / len(t) if t else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def weight_error(
    weights: Dict[str, float], truth: Dict[str, float], budget: float
) -> float:
    """
    ||w - w*||_1 / (2B), in [0, 1] when both sum to B.

    0 is the same allocation; 1 means no budget on any shared column. Missing
    columns count as weight 0, so selection errors show up here too.
    """
    keys = set(weights) | set(truth)
    l1 = sum(abs(weights.get(k, 0.0) - truth.get(k, 0.0)) for k in keys)
    return round(l1 / (2 * budget), 4)


def direction_accuracy(
    directions: Dict[str, int], truth: Dict[str, int]
) -> Optional[float]:
    """Share of columns weighted in both bids whose direction sigma agrees."""
    shared = set(directions) & set(truth)
    if not shared:
        return None
    return round(sum(directions[k] == truth[k] for k in shared) / len(shared), 4)


def rate(flags: Iterable[Optional[bool]]) -> Optional[float]:
    """Share of True among the non-None flags; None if there are none."""
    vals = [f for f in flags if f is not None]
    if not vals:
        return None
    return round(sum(1 for f in vals if f) / len(vals), 4)


def oracle_satisfaction(
    schedule_ids: Iterable[int], oracle_scores: Dict[int, float]
) -> float:
    """
    S*_p(x_p) for the pairing part: the sum of the oracle's pairing scores
    s*_p(j) over the pairings in the schedule. Schedules differ in length, so
    compare it alongside the schedule-level instruction compliance.
    """
    return round(sum(oracle_scores[j] for j in schedule_ids), 4)
