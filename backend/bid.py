"""
bid.py
------
A pilot's bid in column form, the pairing score it induces, and the ranking of
pairings that is the LLM stage's output (research note, section 2).

A bid activates a set of columns (z_p) and spreads a budget B over the scored
ones (w_p, eq. 1). It also carries a direction per scored column. The note
writes sigma_k per feature, but its own example — "long trips" -> TAFB with
sigma = +1, while TAFB's default is -1 — needs the direction to be chosen per
pilot, so the bid stores sigma_pk and the library's sigma_k is only the default.

    s_p(j) = sum_k  w_pk * sigma_pk * phi~_jk                          (eq. 2)

Besides the scored columns a bid can hold
  hard_exclusions  pairing-indicator columns from firm instructions: pairings
                   with phi = 1 on any of them go to the bottom, flagged avoid;
  schedule_prefs   schedule-indicator columns (psi), kept for the PBS stage and
                   not part of the pairing score.
All three count as activated for the selection metrics.

The bid-form oracle (Step 3) is `oracle_bid`: the base oracle's four weights
rescaled to the budget, extended with the columns the monthly instructions
imply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Sequence, Tuple

from features import (
    PAIRING_INDICATOR,
    SCHEDULE_INDICATOR,
    FeatureLibrary,
    FeatureMatrix,
)
from models import Pairing, Pilot

DEFAULT_BUDGET = 10.0
BUDGET_TOLERANCE = 0.01     # |sum w - B| allowed, in budget units
# Cap on the budget share monthly instructions may take in the oracle bid, so
# the pilot's stable profile always keeps a say.
MAX_INSTRUCTION_SHARE = 0.6


@dataclass(frozen=True)
class Bid:
    pilot_id: int
    budget: float
    weights: Mapping[str, float]            # scored columns -> w >= 0
    directions: Mapping[str, int]           # scored columns -> sigma in {-1, +1}
    hard_exclusions: Tuple[str, ...] = ()
    schedule_prefs: Tuple[str, ...] = ()
    rationales: Mapping[str, str] = field(default_factory=dict)

    @property
    def scored(self) -> FrozenSet[str]:
        return frozenset(self.weights)

    @property
    def selected(self) -> FrozenSet[str]:
        """z_p: every activated column."""
        return frozenset(self.weights) | frozenset(self.hard_exclusions) | frozenset(self.schedule_prefs)

    def signed_weights(self) -> Dict[str, float]:
        return {k: w * self.directions[k] for k, w in self.weights.items()}

    def validate(self, library: FeatureLibrary) -> List[str]:
        """Every way this bid breaks the rules; empty if it is valid."""
        errs: List[str] = []
        for k in self.selected:
            if k not in library:
                errs.append(f"unknown column '{k}'")
        for k, w in self.weights.items():
            if k in library and not library[k].is_pairing_level:
                errs.append(f"'{k}' is schedule-level and cannot be weighted")
            if w < 0:
                errs.append(f"weight of '{k}' is negative ({w})")
            if self.directions.get(k) not in (-1, 1):
                errs.append(f"direction of '{k}' must be +1 or -1")
        for k in self.hard_exclusions:
            if k in library and library[k].type != PAIRING_INDICATOR:
                errs.append(f"hard exclusion '{k}' must be a pairing indicator")
        for k in self.schedule_prefs:
            if k in library and library[k].type != SCHEDULE_INDICATOR:
                errs.append(f"schedule preference '{k}' must be a schedule indicator")
        total = sum(self.weights.values())
        if self.weights and abs(total - self.budget) > BUDGET_TOLERANCE:
            errs.append(f"weights sum to {total:g}, not the budget {self.budget:g}")
        return errs

    def as_dict(self) -> Dict:
        return {
            "budget": self.budget,
            "weights": dict(self.weights),
            "directions": dict(self.directions),
            "hard_exclusions": list(self.hard_exclusions),
            "schedule_prefs": list(self.schedule_prefs),
            "rationales": dict(self.rationales),
        }


# ---------------------------------------------------------------------------
# Score and ranking
# ---------------------------------------------------------------------------

def score_pairings(bid: Bid, phi: FeatureMatrix) -> Dict[int, float]:
    """s_p(j) for every pairing in phi (eq. 2), rounded to kill float noise in ties."""
    signed = bid.signed_weights()
    return {
        pid: round(sum(w * phi.norm[pid][k] for k, w in signed.items()), 9)
        for pid in phi.pairing_ids
    }


@dataclass(frozen=True)
class PairingRanking:
    """r_p: pairing ids best -> worst, with the tiers that shaped it."""
    ordered_ids: Tuple[int, ...]
    scores: Dict[int, float]
    avoid: FrozenSet[int]          # hit a hard exclusion
    ineligible: FrozenSet[int]     # pilot not qualified on the aircraft


def rank_pairings(
    bid: Bid,
    phi: FeatureMatrix,
    pairings: Sequence[Pairing],
    pilot: Pilot,
) -> PairingRanking:
    """
    Sort by decreasing score in three tiers: eligible, then eligible but under
    a hard exclusion (avoid), then pairings the pilot is not qualified for.
    Ties are broken by pairing id so the ranking is deterministic.
    """
    scores = score_pairings(bid, phi)
    ids = [p.id for p in pairings]
    ineligible = frozenset(p.id for p in pairings if not p.is_qualified(pilot))
    avoid = frozenset(
        pid for pid in ids
        if any(phi.raw[pid][k] >= 0.5 for k in bid.hard_exclusions)
    ) - ineligible

    def key(pid: int):
        tier = 2 if pid in ineligible else (1 if pid in avoid else 0)
        return (tier, -scores[pid], pid)

    return PairingRanking(tuple(sorted(ids, key=key)), scores, avoid, ineligible)


# ---------------------------------------------------------------------------
# Ordinal bids
# ---------------------------------------------------------------------------

def roc_weights(ordered_keys: Sequence[str], budget: float) -> Dict[str, float]:
    """
    Rank-order centroid weights, scaled to the budget:
    w_i = B / n * sum_{r=i..n} 1/r for the column ranked i (1 = most important).
    """
    n = len(ordered_keys)
    return {
        k: budget / n * sum(1.0 / r for r in range(i, n + 1))
        for i, k in enumerate(ordered_keys, start=1)
    }


def normalise_to_budget(weights: Mapping[str, float], budget: float) -> Dict[str, float]:
    """Clip negatives and rescale to sum to the budget; uniform if all are zero."""
    clipped = {k: max(0.0, float(w)) for k, w in weights.items()}
    total = sum(clipped.values())
    if not clipped:
        return {}
    if total == 0:
        return {k: budget / len(clipped) for k in clipped}
    return {k: w * budget / total for k, w in clipped.items()}


# ---------------------------------------------------------------------------
# Bid-form oracle (Step 3)
# ---------------------------------------------------------------------------

def base_oracle_columns(pilot: Pilot) -> Dict[str, Tuple[float, int]]:
    """
    The base oracle (oracle.py) as columns: key -> (share of budget, sigma).

    Weights are the pilot's OracleWeights / 100. Directions follow the base
    oracle: shorter TAFB, a less early report and more pay are better for
    everyone; hotel nights are bad for family pilots and good for the others.

    The base oracle's report sub-score is flat after 07:00 and linear before
    it, which is exactly the report_earliness column. Its other sub-scores use
    fixed clipped bounds, replaced here by min-max normalisation over J
    (eq. 2), so the rankings agree closely but not exactly.
    """
    W = pilot.weights
    return {
        "tafb": (W.tafb / 100, -1),
        "hotel_nights": (W.hotel_nights / 100, -1 if pilot.has_kids else +1),
        "report_earliness": (W.report_time / 100, -1),
        "credit_pay": (W.credit_pay / 100, +1),
    }


def oracle_bid(
    pilot: Pilot,
    instructions: Sequence = (),
    budget: float = DEFAULT_BUDGET,
) -> Bid:
    """
    Ground-truth bid (z*, w*, sigma*) for a pilot and their monthly instructions.

    Each soft pairing-level instruction takes `share * B` of the budget, split
    evenly over its effect columns (a weekend request puts half on each day).
    The base columns fill the rest in their base proportions. When an
    instruction's column is already in the base with the opposite direction
    ("long trips" vs. the base's short-TAFB preference), the instruction wins:
    the column takes the instruction's direction and the base weight is added
    to it, so the pilot's stated wish for this month overrides their archetype.

    Firm instructions become hard exclusions and take no budget. Schedule-level
    instructions become schedule preferences.
    """
    soft = [i for i in instructions if i.level == "pairing" and not i.firm]
    instr_share = sum(i.share for i in soft)
    if instr_share > MAX_INSTRUCTION_SHARE:
        scale = MAX_INSTRUCTION_SHARE / instr_share
    else:
        scale = 1.0

    base_budget = budget * (1 - instr_share * scale)
    weights: Dict[str, float] = {}
    directions: Dict[str, int] = {}
    rationales: Dict[str, str] = {}
    for k, (share, sigma) in base_oracle_columns(pilot).items():
        weights[k] = share * base_budget
        directions[k] = sigma
        rationales[k] = "profile"

    for ins in soft:
        per_col = ins.share * scale * budget / len(ins.effects)
        for eff in ins.effects:
            if eff.key in weights:
                weights[eff.key] += per_col
            else:
                weights[eff.key] = per_col
            directions[eff.key] = eff.sigma
            rationales[eff.key] = ins.text

    exclusions: List[str] = []
    prefs: List[str] = []
    for ins in instructions:
        if ins.level == "pairing" and ins.firm:
            for eff in ins.effects:
                if eff.key not in exclusions:
                    exclusions.append(eff.key)
                    rationales[eff.key] = ins.text
        elif ins.level == "schedule":
            for eff in ins.effects:
                if eff.key not in prefs:
                    prefs.append(eff.key)
                    rationales[eff.key] = ins.text

    return Bid(
        pilot_id=pilot.id,
        budget=budget,
        weights=weights,
        directions=directions,
        hard_exclusions=tuple(exclusions),
        schedule_prefs=tuple(prefs),
        rationales=rationales,
    )
