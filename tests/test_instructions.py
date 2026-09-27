"""
Monthly instructions: generation, oracle mapping and compliance checks.

Run directly:  python tests/test_instructions.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from bid import oracle_bid, rank_pairings
from features import FeatureLibrary, FeatureMatrix
from generator import ScenarioGenerator
from instructions import Effect, InstructionGenerator, MonthlyInstruction, compliance

GEN = ScenarioGenerator()
PILOTS = GEN.build_pilots(30)
PAIRINGS = GEN.build_pairings(40, PILOTS, max_b767=13)
LIB = FeatureLibrary.for_pairings(PAIRINGS, GEN.month_start)
PHI = FeatureMatrix.build(LIB, PAIRINGS)
INSTR = InstructionGenerator(LIB, PHI, seed=7).for_pilots(PILOTS)
ALL_IDS = [p.id for p in PAIRINGS]


def _ranking_ids(pilot, instrs):
    return rank_pairings(oracle_bid(pilot, instrs), PHI, PAIRINGS, pilot).ordered_ids


def test_more_than_ten_pilots_and_first_ten_unchanged():
    assert len(PILOTS) == 30 and len({p.name for p in PILOTS}) == 30
    ten = ScenarioGenerator().build_pilots(10)
    assert [(p.name, p.age, p.family_status) for p in ten] == \
           [(p.name, p.age, p.family_status) for p in PILOTS[:10]]


def test_generation_is_deterministic():
    again = InstructionGenerator(LIB, PHI, seed=7).for_pilots(PILOTS)
    assert again == INSTR
    other = InstructionGenerator(LIB, PHI, seed=8).for_pilots(PILOTS)
    assert other != INSTR


def test_every_effect_is_a_library_column_and_counts_are_bounded():
    kinds_seen = set()
    for pid, instrs in INSTR.items():
        assert len(instrs) <= 3
        assert len({i.kind for i in instrs}) == len(instrs), "one instruction per kind"
        assert not {"long_trips", "short_trips"} <= {i.kind for i in instrs}
        for i in instrs:
            kinds_seen.add(i.kind)
            assert i.text
            for e in i.effects:
                assert e.key in LIB, e.key
                assert e.sigma in (-1, 1)
            if i.level == "schedule":
                assert all(LIB[e.key].type == "schedule_indicator" for e in i.effects)
            if i.firm:
                assert all(LIB[e.key].type == "pairing_indicator" for e in i.effects)
    assert len(kinds_seen) >= 6, kinds_seen


def test_no_pilot_repeats_a_day():
    for instrs in INSTR.values():
        days = [e.key.rsplit("_", 1)[1] for i in instrs for e in i.effects
                if e.key.startswith(("touches_day_", "away_evening_"))]
        assert len(days) == len(set(days))


def test_oracle_bids_are_valid_for_every_pilot():
    for p in PILOTS:
        b = oracle_bid(p, INSTR[p.id])
        assert b.validate(LIB) == [], (p.id, b.validate(LIB))


def test_oracle_ranking_meets_every_firm_instruction():
    for p in PILOTS:
        order = _ranking_ids(p, INSTR[p.id])
        elig = [q.id for q in PAIRINGS if q.is_qualified(p)]
        for ins, ok in zip(INSTR[p.id], compliance(INSTR[p.id], order, LIB, PHI, elig)):
            if ins.firm:
                assert ok is True, (p.id, ins.text)
            if ins.level == "schedule":
                assert ok is None


def test_firm_check_fails_when_a_violator_is_on_top():
    day = next(d for d in range(1, LIB.days_in_month + 1)
               if 0 < sum(PHI.raw[j][f"touches_day_{d}"] for j in ALL_IDS) < len(ALL_IDS))
    ins = MonthlyInstruction("day_off", "x", (Effect(f"touches_day_{day}", -1),), 0.15, firm=True)
    hit = [j for j in ALL_IDS if PHI.raw[j][f"touches_day_{day}"]]
    clean = [j for j in ALL_IDS if j not in hit]
    assert ins.check_ranking(clean + hit, LIB, PHI, ALL_IDS) is True
    assert ins.check_ranking([hit[0]] + clean + hit[1:], LIB, PHI, ALL_IDS) is False


def test_soft_avoid_allows_violators_only_when_unavoidable():
    day = next(d for d in range(1, LIB.days_in_month + 1)
               if sum(PHI.raw[j][f"touches_day_{d}"] for j in ALL_IDS) >= 2)
    ins = MonthlyInstruction("day_off", "x", (Effect(f"touches_day_{day}", -1),), 0.15)
    hit = [j for j in ALL_IDS if PHI.raw[j][f"touches_day_{day}"]]
    clean = [j for j in ALL_IDS if j not in hit]
    assert ins.check_ranking(clean + hit, LIB, PHI, ALL_IDS, k=5) is True
    assert ins.check_ranking(hit[:1] + clean + hit[1:], LIB, PHI, ALL_IDS, k=5) is False
    # With only 2 clean pairings among 4 eligible, 2 violators in the top 4 is unavoidable.
    elig = clean[:2] + hit[:2]
    assert ins.check_ranking(clean[:2] + hit[:2], LIB, PHI, elig, k=4) is True


def test_want_and_continuous_checks():
    code = next(c for c in LIB.layover_airports
                if sum(PHI.raw[j][f"layover_{c}"] for j in ALL_IDS) >= 1)
    want = MonthlyInstruction("layover_want", "x", (Effect(f"layover_{code}", +1),), 0.15)
    hit = [j for j in ALL_IDS if PHI.raw[j][f"layover_{code}"]]
    rest = [j for j in ALL_IDS if j not in hit]
    assert want.check_ranking(hit + rest, LIB, PHI, ALL_IDS, k=3) is True
    assert want.check_ranking(rest + hit, LIB, PHI, ALL_IDS, k=3) is False

    long_ = MonthlyInstruction("long_trips", "x", (Effect("tafb", +1),), 0.2)
    by_tafb = sorted(ALL_IDS, key=lambda j: -PHI.raw[j]["tafb"])
    assert long_.check_ranking(by_tafb, LIB, PHI, ALL_IDS) is True
    assert long_.check_ranking(by_tafb[::-1], LIB, PHI, ALL_IDS) is False


def test_schedule_checks():
    by_id = {p.id: p for p in PAIRINGS}
    code = LIB.layover_airports[0]
    sched = MonthlyInstruction("sched_layover", "x", (Effect(f"sched_layover_{code}", +1),),
                               level="schedule")
    there = [p for p in PAIRINGS if code in p.layover_airports]
    elsewhere = [p for p in PAIRINGS if code not in p.layover_airports][:3]
    assert sched.check_schedule(there[:1] + elsewhere, LIB, PHI, ALL_IDS) is True
    assert sched.check_schedule(elsewhere, LIB, PHI, ALL_IDS) is False

    day_ins = MonthlyInstruction("day_off", "x", (Effect("touches_day_10", -1),), 0.15)
    on_10 = [by_id[j] for j in ALL_IDS if PHI.raw[j]["touches_day_10"]]
    off_10 = [by_id[j] for j in ALL_IDS if not PHI.raw[j]["touches_day_10"]]
    assert day_ins.check_schedule(off_10[:3], LIB, PHI, ALL_IDS) is True
    assert day_ins.check_schedule(off_10[:2] + on_10[:1], LIB, PHI, ALL_IDS) is False


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
