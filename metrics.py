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
