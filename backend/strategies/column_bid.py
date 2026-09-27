"""
Column-selection and column-weighting bid (research note, sections 2-3).

The LLM does not rank pairings. It produces a bid (z_p, w_p, sigma_p) and the
ranking r_p follows from eq. (2) — so the bid explains the ranking.

  mode="separated"  z_p = LLM_sel(theta, iota); w_p = LLM_wgt(theta, iota, z_p, I_p)
  mode="joint"      (z_p, w_p) in one call

  regime  information at the weighting stage
    R1    blind: profile, instructions and column definitions only
    R2    full: the trips' values on the selected columns (joint: all columns)
    R3    partial: (1 - alpha) B allocated blind, then alpha B with the trips
          in view (optionally only a random sample of sample_size trips)
    R4    interactive: blind weights, then up to max_rounds rounds of feedback
          on the top shortlist_size trips, with weights revised each round

  weight_format="budget"  the LLM spreads B points
  weight_format="rank"    the LLM ranks the columns; rank-order centroid
                          weights scaled to B (robustness comparison)

Selection never sees the trips, in either mode, except joint under R2.

A reply that breaks the schema is sent back with the violation stated, up to
max_retries times. If it is still invalid, the last reply is repaired
(unknown columns dropped, weights clipped and rescaled) and the stage is
listed in artifacts["repaired"], so repaired bids can be filtered out.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from bid import BUDGET_TOLERANCE, Bid, normalise_to_budget, rank_pairings, roc_weights
from bid_prompts import (
    adjust_prompt,
    compact_trip_table,
    increment_prompt,
    joint_prompt,
    selection_prompt,
    trip_table,
    weighting_prompt,
)
from features import PAIRING_INDICATOR, SCHEDULE_INDICATOR, FeatureLibrary
from harness import Ranking
from models import Pairing, Pilot

MODES = ("separated", "joint")
REGIMES = ("R1", "R2", "R3", "R4")
WEIGHT_FORMATS = ("budget", "rank")


class BidParseError(ValueError):
    """A reply that does not follow the requested schema."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _json_object(raw: str) -> dict:
    text = raw.replace("```json", "").replace("```", "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise BidParseError("no JSON object found in the reply")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise BidParseError(f"invalid JSON: {exc}") from None
    if not isinstance(obj, dict):
        raise BidParseError("the reply must be a JSON object")
    return obj


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise BidParseError(f"{what} must be a number")
    try:
        return float(v)
    except ValueError:
        raise BidParseError(f"{what} must be a number") from None


@dataclass
class Selection:
    directions: Dict[str, int] = field(default_factory=dict)     # scored columns, in order
    exclusions: List[str] = field(default_factory=list)
    schedule_prefs: List[str] = field(default_factory=list)
    rationales: Dict[str, str] = field(default_factory=dict)
    amounts: Dict[str, float] = field(default_factory=dict)      # joint only: weight or rank


def parse_selection(
    raw: str, library: FeatureLibrary, strict: bool, amount_key: Optional[str] = None,
) -> Selection:
    """
    Parse {"selected": [...]}. With amount_key ("weight" or "rank") each scored
    item must also carry that field (joint bids). Non-strict drops bad items
    and falls back to the library's default direction.
    """
    obj = _json_object(raw)
    items = obj.get("selected")
    if not isinstance(items, list):
        raise BidParseError('the reply must have a "selected" list')
    sel = Selection()
    errs: List[str] = []
    seen = set()
    for n, it in enumerate(items, 1):
        if not isinstance(it, dict) or not isinstance(it.get("key"), str):
            errs.append(f"item {n} has no string \"key\"")
            continue
        key = it["key"].strip()
        if key in seen:
            errs.append(f"'{key}' is listed twice")
            continue
        if key not in library:
            errs.append(f"'{key}' is not a column in the library")
            continue
        seen.add(key)
        f = library[key]
        rationale = str(it.get("rationale", ""))
        if f.type == SCHEDULE_INDICATOR:
            sel.schedule_prefs.append(key)
            sel.rationales[key] = rationale
            continue
        if it.get("hard_exclusion") is True:
            if f.type != PAIRING_INDICATOR:
                errs.append(f"'{key}' cannot be a hard exclusion: only indicator columns can")
                if strict:
                    continue
            else:
                sel.exclusions.append(key)
                sel.rationales[key] = rationale
                continue
        d = it.get("direction")
        try:
            d_int = int(d) if not isinstance(d, bool) else None
        except (TypeError, ValueError):
            d_int = None
        if d_int not in (-1, 1):
            errs.append(f"direction of '{key}' must be +1 or -1")
            d_int = f.sigma
        sel.directions[key] = d_int
        sel.rationales[key] = rationale
        if amount_key is not None:
            if amount_key not in it:
                errs.append(f"'{key}' has no \"{amount_key}\"")
            else:
                try:
                    sel.amounts[key] = _num(it[amount_key], f"{amount_key} of '{key}'")
                except BidParseError as exc:
                    errs.append(str(exc))
    if strict and errs:
        raise BidParseError("; ".join(errs))
    return sel


def parse_weights(
    raw: str, keys: Sequence[str], budget: float, strict: bool, field_name: str = "weights",
    require_all: bool = True,
) -> Dict[str, float]:
    """Parse {field_name: {key: w}} summing to budget over exactly `keys`."""
    obj = _json_object(raw)
    w = obj.get(field_name)
    if not isinstance(w, dict):
        raise BidParseError(f'the reply must have a "{field_name}" object')
    errs: List[str] = []
    out: Dict[str, float] = {}
    for k, v in w.items():
        if k not in keys:
            errs.append(f"'{k}' is not one of the columns to weight")
            continue
        try:
            val = _num(v, f"weight of '{k}'")
        except BidParseError as exc:
            errs.append(str(exc))
            continue
        if val < 0:
            errs.append(f"weight of '{k}' is negative")
        out[k] = val
    missing = [k for k in keys if k not in out]
    if require_all and missing:
        errs.append("missing weights for " + ", ".join(missing))
    total = sum(out.values())
    if abs(total - budget) > BUDGET_TOLERANCE:
        errs.append(f"weights sum to {total:g}, not {budget:g}")
    if strict and errs:
        raise BidParseError("; ".join(errs))
    full = {k: out.get(k, 0.0) for k in keys}
    return full if not errs else normalise_to_budget(full, budget)


def parse_ranking(raw: str, keys: Sequence[str], strict: bool) -> List[str]:
    """Parse {"ranking": [...]}, a permutation of `keys`."""
    obj = _json_object(raw)
    r = obj.get("ranking")
    if not isinstance(r, list):
        raise BidParseError('the reply must have a "ranking" list')
    order: List[str] = []
    errs: List[str] = []
    for k in r:
        if k not in keys:
            errs.append(f"'{k}' is not one of the columns to rank")
        elif k in order:
            errs.append(f"'{k}' appears twice")
        else:
            order.append(k)
    missing = [k for k in keys if k not in order]
    if missing:
        errs.append("missing " + ", ".join(missing))
    if strict and errs:
        raise BidParseError("; ".join(errs))
    return order + missing


def ranks_to_order(amounts: Dict[str, float], keys: Sequence[str], strict: bool) -> List[str]:
    """Joint rank format: the 'rank' fields must form 1..n."""
    ranks = [amounts.get(k) for k in keys]
    if strict and sorted(r for r in ranks if r is not None) != list(range(1, len(keys) + 1)):
        raise BidParseError(f"ranks must be 1..{len(keys)} with no repeats")
    return sorted(keys, key=lambda k: (amounts.get(k, float("inf")), list(keys).index(k)))


# ---------------------------------------------------------------------------
# Session: LLM calls with retries
# ---------------------------------------------------------------------------

class _Session:
    def __init__(self, client, max_retries: int, keep_prompts: bool):
        self.client = client
        self.max_retries = max_retries
        self.keep_prompts = keep_prompts
        self.calls = 0
        self.repaired: List[str] = []
        self.transcript: List[Dict[str, Any]] = []

    def ask(self, stage: str, prompt: str, parse: Callable[[str, bool], Any]) -> Any:
        attempt_prompt = prompt
        raw = ""
        last_err = ""
        for attempt in range(self.max_retries + 1):
            raw = self.client.complete(attempt_prompt)
            self.calls += 1
            entry = {"stage": stage, "attempt": attempt, "response": raw}
            if self.keep_prompts:
                entry["prompt"] = attempt_prompt
            try:
                value = parse(raw, True)
                self.transcript.append(entry)
                return value
            except BidParseError as exc:
                last_err = str(exc)
                entry["error"] = last_err
                self.transcript.append(entry)
                attempt_prompt = (
                    prompt + "\n\nYour previous reply was not valid: " + last_err
                    + "\nYour previous reply was:\n" + raw
                    + "\nReturn the corrected JSON only."
                )
        try:
            value = parse(raw, False)
        except BidParseError as exc:
            raise ValueError(
                f"{stage}: no usable reply after {self.max_retries + 1} attempts ({exc})"
            ) from None
        self.repaired.append(stage)
        return value


# ---------------------------------------------------------------------------
# Method
# ---------------------------------------------------------------------------

class ColumnBid:
    """Implements harness.RankingMethod over the pairings of a PBSInstance."""

    def __init__(
        self,
        client,
        instance,
        mode: str = "separated",
        regime: str = "R1",
        weight_format: str = "budget",
        alpha: float = 0.2,
        sample_size: Optional[int] = None,
        shortlist_size: int = 5,
        max_rounds: int = 2,
        include_priorities: bool = True,
        max_retries: int = 2,
        feedback: Optional[Callable[[Pilot, Sequence[int]], List[Tuple[int, str, str]]]] = None,
        keep_prompts: bool = False,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}")
        if weight_format not in WEIGHT_FORMATS:
            raise ValueError(f"weight_format must be one of {WEIGHT_FORMATS}")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.client = client
        self.instance = instance
        self.mode = mode
        self.regime = regime
        self.weight_format = weight_format
        self.alpha = alpha
        self.sample_size = sample_size
        self.shortlist_size = shortlist_size
        self.max_rounds = max_rounds
        self.include_priorities = include_priorities
        self.max_retries = max_retries
        self.feedback = feedback or instance.oracle_feedback
        self.keep_prompts = keep_prompts

        parts = [mode, regime, weight_format]
        if regime == "R3":
            parts.append(f"a{alpha:g}" + (f"-s{sample_size}" if sample_size else ""))
        if regime == "R4":
            parts.append(f"L{shortlist_size}-t{max_rounds}")
        if not include_priorities:
            parts.append("noprio")
        self.variant = "/".join(parts)
        self.name = "column_bid/" + self.variant

    # ------------------------------------------------------------------
    def rank(self, pilot: Pilot, candidates: Sequence[Pairing], *, seed: int) -> Ranking:
        inst = self.instance
        lib, phi, B = inst.library, inst.phi, inst.budget
        instrs = inst.instructions[pilot.id]
        pairings = list(candidates)
        rng = random.Random(seed)
        s = _Session(self.client, self.max_retries, self.keep_prompts)
        prio = self.include_priorities

        def table_for(sel: Selection, sample: bool = False) -> str:
            shown = pairings
            if sample and self.sample_size and self.sample_size < len(pairings):
                shown = sorted(rng.sample(pairings, self.sample_size), key=lambda p: p.id)
            return trip_table(phi, shown, list(sel.directions) + sel.exclusions, lib, pilot)

        def weigh(sel: Selection, budget: float, table: Optional[str], stage: str) -> Dict[str, float]:
            keys = list(sel.directions)
            if not keys:
                return {}
            prompt = weighting_prompt(pilot, instrs, lib, sel.directions, sel.exclusions,
                                      sel.schedule_prefs, budget, self.weight_format, prio, table)
            return self._ask_weights(s, stage, prompt, keys, budget)

        # -- selection (+ weights) ------------------------------------------------
        first_budget = B * (1 - self.alpha) if self.regime == "R3" else B
        if self.mode == "separated":
            sel = s.ask("selection", selection_prompt(pilot, instrs, lib, prio),
                        lambda raw, strict: parse_selection(raw, lib, strict))
            if self.regime == "R2":
                weights = weigh(sel, B, table_for(sel), "weighting")
            else:
                weights = weigh(sel, first_budget, None, "weighting")
        else:
            table = compact_trip_table(phi, pairings, lib, pilot) if self.regime == "R2" else None
            sel, weights = self._ask_joint(s, pilot, instrs, first_budget, table)

        # -- R3: informed increment -------------------------------------------------
        if self.regime == "R3" and sel.directions:
            inc_budget = B * self.alpha
            prompt = increment_prompt(pilot, instrs, lib, sel.directions, sel.exclusions,
                                      sel.schedule_prefs, weights, inc_budget, prio,
                                      table_for(sel, sample=True))
            keys = list(sel.directions)
            inc = s.ask("increment", prompt, lambda raw, strict: parse_weights(
                raw, keys, inc_budget, strict, field_name="increment", require_all=False))
            weights = {k: weights.get(k, 0.0) + inc.get(k, 0.0) for k in keys}

        # -- R4: feedback rounds ----------------------------------------------------
        rounds = 0
        history: List[Dict[str, Any]] = []
        if self.regime == "R4" and sel.directions:
            for t in range(1, self.max_rounds + 1):
                ranked = rank_pairings(self._bid(pilot, sel, weights), phi, pairings, pilot)
                ok = [pid for pid in ranked.ordered_ids
                      if pid not in ranked.ineligible and pid not in ranked.avoid]
                shortlist = ok[:self.shortlist_size]
                fb = self.feedback(pilot, shortlist)
                history.append({"round": t, "weights": dict(weights), "shortlist": shortlist,
                                "feedback": [list(x) for x in fb]})
                if all(v == "great" for _, v, _ in fb):
                    break
                by_id = {p.id: p for p in pairings}
                table = trip_table(phi, [by_id[i] for i in shortlist],
                                   list(sel.directions) + sel.exclusions, lib, pilot)
                prompt = adjust_prompt(pilot, instrs, lib, sel.directions, sel.exclusions,
                                       sel.schedule_prefs, weights, table, fb, B,
                                       self.weight_format, prio, t)
                weights = self._ask_weights(s, f"adjust_{t}", prompt, list(sel.directions), B)
                rounds += 1

        # -- ranking ---------------------------------------------------------------
        bid = self._bid(pilot, sel, weights)
        problems = bid.validate(lib)
        if problems:
            raise ValueError(f"internal: invalid bid for pilot {pilot.id}: {problems}")
        ranked = rank_pairings(bid, phi, pairings, pilot)
        return Ranking(
            method=self.name,
            pilot_id=pilot.id,
            ordered_ids=ranked.ordered_ids,
            scores=dict(ranked.scores),
            artifacts={
                "bid": bid.as_dict(),
                "n_selected": len(bid.selected),
                "avoid": sorted(ranked.avoid),
                "n_llm_calls": s.calls,
                "feedback_rounds": rounds,
                "feedback_history": history,
                "repaired": list(s.repaired),
                "transcript": s.transcript,
            },
            variant=self.variant,
            selection_context="trips" if (self.mode == "joint" and self.regime == "R2") else "blind",
            weight_source=self.regime,
        )

    # ------------------------------------------------------------------
    def _bid(self, pilot: Pilot, sel: Selection, weights: Dict[str, float]) -> Bid:
        return Bid(
            pilot_id=pilot.id,
            budget=self.instance.budget,
            weights={k: weights.get(k, 0.0) for k in sel.directions},
            directions=dict(sel.directions),
            hard_exclusions=tuple(sel.exclusions),
            schedule_prefs=tuple(sel.schedule_prefs),
            rationales=dict(sel.rationales),
        )

    def _ask_weights(self, s: _Session, stage: str, prompt: str,
                     keys: List[str], budget: float) -> Dict[str, float]:
        if self.weight_format == "rank":
            order = s.ask(stage, prompt, lambda raw, strict: parse_ranking(raw, keys, strict))
            return roc_weights(order, budget)
        return s.ask(stage, prompt, lambda raw, strict: parse_weights(raw, keys, budget, strict))

    def _ask_joint(self, s: _Session, pilot: Pilot, instrs, budget: float,
                   table: Optional[str]) -> Tuple[Selection, Dict[str, float]]:
        lib = self.instance.library
        amount_key = "rank" if self.weight_format == "rank" else "weight"
        prompt = joint_prompt(pilot, instrs, lib, budget, self.weight_format,
                              self.include_priorities, table)

        def parse(raw: str, strict: bool):
            sel = parse_selection(raw, lib, strict, amount_key=amount_key)
            keys = list(sel.directions)
            if not keys:
                return sel, {}
            if amount_key == "rank":
                return sel, roc_weights(ranks_to_order(sel.amounts, keys, strict), budget)
            w = {k: sel.amounts.get(k, 0.0) for k in keys}
            total = sum(w.values())
            if strict and (any(v < 0 for v in w.values()) or abs(total - budget) > BUDGET_TOLERANCE):
                raise BidParseError(
                    f"weights must be non-negative and sum to exactly {budget:g} (they sum to {total:g})")
            return sel, (w if strict else normalise_to_budget(w, budget))

        return s.ask("joint", prompt, parse)
