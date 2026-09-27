"""
Calendar dates, schedule conflicts and rest checks.

Run directly (no pytest needed):
    python tests/test_schedule.py
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from generator import ScenarioGenerator
from models import Leg, Line, Pairing, is_legal_schedule, schedule_violations


def _leg(dep_day: int, dep_time: str, arr_day: int, arr_time: str) -> Leg:
    return Leg(dep="BOS", arr="ORD", dep_city="Boston", arr_city="Chicago",
               distance_mi=854, block_mins=120, dep_day=dep_day,
               dep_time=dep_time, arr_day=arr_day, arr_time=arr_time)


def _trip(pid: int, start: date, legs) -> Pairing:
    return Pairing(id=pid, aircraft="B737", aircraft_label="Boeing 737",
                   base="BOS", legs=legs, nights_away=legs[-1].dep_day - 1,
                   start_date=start)


# Oct 5: report 7:00, release Oct 6 18:30.
TWO_DAY = _trip(1, date(2026, 10, 5), [_leg(1, "8:00 AM", 1, "10:00 AM"),
                                       _leg(2, "4:00 PM", 2, "6:00 PM")])


def test_absolute_times():
    assert TWO_DAY.start_dow == "Mon"
    assert TWO_DAY.start_label == "Mon 2026-10-05"
    assert TWO_DAY.end_date == date(2026, 10, 6)
    assert TWO_DAY.report_dt == datetime(2026, 10, 5, 7, 0)
    assert TWO_DAY.release_dt == datetime(2026, 10, 6, 18, 30)
    # Hotel rest: released Oct 5 10:30, reports Oct 6 15:00.
    assert TWO_DAY.layover_rests == [timedelta(hours=28, minutes=30)]


def test_overlap_is_detected():
    later = _trip(2, date(2026, 10, 6), [_leg(1, "5:00 PM", 1, "7:00 PM")])
    v = schedule_violations([later, TWO_DAY])
    assert [x.kind for x in v] == ["overlap"]
    assert (v[0].earlier.id, v[0].later.id) == (1, 2)
    assert not is_legal_schedule([TWO_DAY, later], 0)


def test_overlap_inside_a_long_trip_is_detected():
    # A short trip nested inside a long one, followed by a legal third trip.
    long_trip = _trip(1, date(2026, 10, 5), [_leg(1, "8:00 AM", 1, "10:00 AM"),
                                             _leg(4, "8:00 AM", 4, "10:00 AM")])
    inner = _trip(2, date(2026, 10, 6), [_leg(1, "8:00 AM", 1, "10:00 AM")])
    after = _trip(3, date(2026, 10, 20), [_leg(1, "8:00 AM", 1, "10:00 AM")])
    kinds = [x.kind for x in schedule_violations([long_trip, inner, after])]
    assert kinds == ["overlap"]


def test_rest_shortfall_depends_on_the_pilot_minimum():
    # Released Oct 6 18:30; next report Oct 7 06:00 → 11h30 rest.
    nxt = _trip(2, date(2026, 10, 7), [_leg(1, "7:00 AM", 1, "9:00 AM")])
    assert is_legal_schedule([TWO_DAY, nxt], 11)
    v = schedule_violations([TWO_DAY, nxt], 12)
    assert [x.kind for x in v] == ["rest"]
    assert v[0].rest == timedelta(hours=11, minutes=30)
    assert Line(id=1, pairings=[TWO_DAY, nxt]).has_conflicts() is False


def test_generated_lines_are_legal_and_inside_the_month():
    gen = ScenarioGenerator(1234567, 42)
    pilots = gen.build_pilots(5)
    pairings = gen.build_pairings(100, pilots, "BOS", max_b767=32)
    lines = gen.build_lines(pairings, 20, 5)
    for line in lines:
        assert not line.has_conflicts(), line.id
        for pilot in pilots:
            assert line.is_legal_for(pilot), (line.id, pilot.name)
        starts = [p.start_date for p in line.pairings]
        assert starts == sorted(starts), line.id
    for p in pairings:
        assert (p.start_date.year, p.start_date.month) == (2026, 10)
        assert (p.end_date.year, p.end_date.month) == (2026, 10)


def test_bid_month_is_configurable():
    gen = ScenarioGenerator(1, 2, bid_month=date(2027, 2, 14))
    pairings = gen.build_pairings(20)
    for p in pairings:
        assert date(2027, 2, 1) <= p.start_date and p.end_date <= date(2027, 2, 28)


def test_dates_do_not_change_routes_or_times():
    # Dates use their own PRNG, so the rest of the pairing is seed-stable
    # whatever the bid month.
    a = ScenarioGenerator(7, 9).build_pairings(30)
    b = ScenarioGenerator(7, 9, bid_month=date(2027, 2, 1)).build_pairings(30)
    assert [p.legs for p in a] == [p.legs for p in b]
    assert [p.aircraft for p in a] == [p.aircraft for p in b]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS ", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")
