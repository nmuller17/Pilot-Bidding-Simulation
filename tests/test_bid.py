"""
Feature library, Phi, bids and the bid-form oracle (research note Steps 2-3).

Run directly:  python tests/test_bid.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

import metrics as M
from bid import Bid, oracle_bid, rank_pairings, roc_weights, score_pairings
from features import (
    PAIRING_INDICATOR,
    SCHEDULE_INDICATOR,
    FeatureLibrary,
    FeatureMatrix,
)
from generator import ScenarioGenerator
from oracle import oracle_rank_pilot

GEN = ScenarioGenerator()
PILOTS = GEN.build_pilots(8)
PAIRINGS = GEN.build_pairings(40, PILOTS, max_b767=13)
LIB = FeatureLibrary.for_pairings(PAIRINGS, GEN.month_start)
PHI = FeatureMatrix.build(LIB, PAIRINGS)
FAMILY = next(p for p in PILOTS if p.has_kids)
NON_FAMILY = next(p for p in PILOTS if not p.has_kids)
B737_ONLY = next((p for p in PILOTS if p.qualified_types == ["B737"]), None)


def _effect(key, sigma):
    return SimpleNamespace(key=key, sigma=sigma)


def _instr(text, effects, share=0.2, firm=False, level="pairing"):
    return SimpleNamespace(text=text, effects=tuple(effects), share=share, firm=firm, level=level)


# ---------------------------------------------------------------------------
# Library and Phi
# ---------------------------------------------------------------------------

def test_library_has_one_calendar_column_per_day():
    for d in range(1, LIB.days_in_month + 1):
        assert f"touches_day_{d}" in LIB and f"away_evening_{d}" in LIB
    assert "touches_day_0" not in LIB and f"touches_day_{LIB.days_in_month + 1}" not in LIB


def test_library_types():
    assert LIB["tafb"].type == "continuous"
    assert LIB["touches_day_3"].type == PAIRING_INDICATOR
    assert LIB["sched_free_weekends_2"].type == SCHEDULE_INDICATOR
    assert all(f.extract is not None for f in LIB.pairing_level())
    assert all(f.evaluate is not None for f in LIB.schedule_level())


def test_layover_columns_match_the_pairings():
    codes = {c for p in PAIRINGS for c in p.layover_airports}
    assert set(LIB.layover_airports) == codes
    for p in PAIRINGS:
        for c in codes:
            assert PHI.raw[p.id][f"layover_{c}"] == float(c in p.layover_airports)


def test_phi_is_normalised_over_j():
    for k in PHI.keys:
        col = PHI.column(k, normalised=True).values()
        assert all(0.0 <= v <= 1.0 for v in col), k
        if PHI.is_constant(k):
            assert all(v == 0.0 for v in col), f"constant column {k} must normalise to 0"
        else:
            assert min(col) == 0.0 and max(col) == 1.0, k


def test_touches_day_matches_the_trip_dates():
    for p in PAIRINGS:
        for d in range(1, LIB.days_in_month + 1):
            day = LIB.day(d)
            expected = float(p.start_date <= day <= p.end_date)
            assert PHI.raw[p.id][f"touches_day_{d}"] == expected


def test_report_earliness_is_flat_after_seven():
    for p in PAIRINGS:
        mins = p.report_dt.hour * 60 + p.report_dt.minute
        assert PHI.raw[p.id]["report_earliness"] == max(0, 420 - mins)


def test_free_weekends_schedule_indicator():
    # Oct 2026: Saturdays 3, 10, 17, 24 have their Sunday in the month; the 31st does not.
    assert LIB["sched_free_weekends_4"].evaluate([]) is True
    on_weekend = next(p for p in PAIRINGS
                      if PHI.raw[p.id]["touches_day_10"] or PHI.raw[p.id]["touches_day_11"])
    # A trip of at most 4 days that touches the 10th/11th cannot reach the other weekends.
    assert LIB["sched_free_weekends_4"].evaluate([on_weekend]) is False
    assert LIB["sched_free_weekends_3"].evaluate([on_weekend]) is True


# ---------------------------------------------------------------------------
# Bids
# ---------------------------------------------------------------------------

def test_score_is_eq_2():
    bid = Bid(0, 10.0, {"tafb": 7.0, "credit_pay": 3.0}, {"tafb": -1, "credit_pay": 1})
    s = score_pairings(bid, PHI)
    for pid in PHI.pairing_ids:
        exp = -7.0 * PHI.norm[pid]["tafb"] + 3.0 * PHI.norm[pid]["credit_pay"]
        assert abs(s[pid] - exp) < 1e-9


def test_validate_catches_budget_unknown_and_type_errors():
    bad = Bid(0, 10.0, {"tafb": 4.0, "nope": 1.0, "sched_free_weekends_1": 1.0},
              {"tafb": -1, "nope": 1, "sched_free_weekends_1": 1},
              hard_exclusions=("tafb",))
    errs = " | ".join(bad.validate(LIB))
    assert "unknown column 'nope'" in errs
    assert "schedule-level" in errs
    assert "must be a pairing indicator" in errs
    assert "not the budget" in errs
    ok = Bid(0, 10.0, {"tafb": 6.0, "credit_pay": 4.0}, {"tafb": -1, "credit_pay": 1})
    assert ok.validate(LIB) == []


def test_ranking_tiers_exclusions_and_eligibility():
    bid = Bid(0, 10.0, {"credit_pay": 10.0}, {"credit_pay": 1},
              hard_exclusions=("touches_day_10",))
    pilot = B737_ONLY or FAMILY
    r = rank_pairings(bid, PHI, PAIRINGS, pilot)
    pos = {pid: i for i, pid in enumerate(r.ordered_ids)}
    ok = [p.id for p in PAIRINGS if p.id not in r.avoid and p.id not in r.ineligible]
    for a in ok:
        for b in r.avoid:
            assert pos[a] < pos[b]
        for c in r.ineligible:
            assert pos[a] < pos[c]
    for pid in r.avoid:
        assert PHI.raw[pid]["touches_day_10"] == 1.0
    assert sorted(r.ordered_ids) == sorted(p.id for p in PAIRINGS)
    # Within the eligible tier scores are non-increasing.
    s = [r.scores[pid] for pid in r.ordered_ids if pid in ok]
    assert s == sorted(s, reverse=True)


def test_roc_weights():
    w = roc_weights(["a", "b", "c"], 10.0)
    assert abs(sum(w.values()) - 10.0) < 1e-9
    assert abs(w["a"] - 10 * (1 + 1 / 2 + 1 / 3) / 3) < 1e-9
    assert w["a"] > w["b"] > w["c"]


# ---------------------------------------------------------------------------
# Oracle in bid form
# ---------------------------------------------------------------------------

def test_oracle_bid_without_instructions_is_the_base_oracle():
    for p in PILOTS:
        b = oracle_bid(p)
        assert b.validate(LIB) == []
        assert set(b.weights) == {"tafb", "hotel_nights", "report_earliness", "credit_pay"}
        assert abs(b.weights["tafb"] - p.weights.tafb / 10) < 1e-9
        assert b.directions["hotel_nights"] == (-1 if p.has_kids else +1)


def test_oracle_bid_ranking_tracks_the_base_oracle():
    """Not identical (normalisation over J vs fixed bounds) but close."""
    rhos = []
    for p in PILOTS:
        new = rank_pairings(oracle_bid(p), PHI, PAIRINGS, p).ordered_ids
        old = tuple(r.pairing.id for r in oracle_rank_pilot(p, PAIRINGS))
        rhos.append(M.spearman(new, old))
    assert min(rhos) >= 0.7, rhos
    assert sum(rhos) / len(rhos) >= 0.85, rhos


def test_oracle_bid_with_instructions():
    weekend = _instr("weekend of the 17th", [_effect("touches_day_17", -1), _effect("touches_day_18", -1)], 0.2)
    long_trips = _instr("long trips", [_effect("tafb", +1)], 0.2)
    firm = _instr("must be off the 6th", [_effect("touches_day_6", -1)], 0.15, firm=True)
    sched = _instr("two free weekends", [_effect("sched_free_weekends_2", +1)], 0.0, level="schedule")
    b = oracle_bid(FAMILY, [weekend, long_trips, firm, sched])
    assert b.validate(LIB) == []
    assert abs(b.weights["touches_day_17"] - 1.0) < 1e-9     # 0.2 * 10 / 2 days
    assert b.directions["tafb"] == +1                         # instruction overrides archetype
    base_tafb = FAMILY.weights.tafb / 100 * 10 * (1 - 0.4)
    assert abs(b.weights["tafb"] - (base_tafb + 2.0)) < 1e-9
    assert b.hard_exclusions == ("touches_day_6",)
    assert b.schedule_prefs == ("sched_free_weekends_2",)
    assert "touches_day_6" not in b.weights
    assert b.selected >= {"tafb", "touches_day_17", "touches_day_6", "sched_free_weekends_2"}


def test_oracle_bid_caps_the_instruction_share():
    many = [_instr(f"i{k}", [_effect(f"touches_day_{k}", -1)], 0.3) for k in range(1, 5)]
    b = oracle_bid(NON_FAMILY, many)
    assert b.validate(LIB) == []
    base = sum(b.weights[k] for k in ("tafb", "hotel_nights", "report_earliness", "credit_pay"))
    assert abs(base - 4.0) < 1e-9     # instructions capped at 60% of B = 10


# ---------------------------------------------------------------------------

def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
