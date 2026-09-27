"""
features.py
-----------
The feature library K (bid columns) and the pairing-feature matrix Phi of the
PBS research note (section 2, Step 2).

Every column has a key, a name, a natural-language description, a default
preferred direction sigma (+1: higher is better, -1: lower is better) and a
type:

  continuous          a number per pairing (TAFB, report time, ...)
  pairing_indicator   0/1 per pairing ("touches day 17", "lays over in DEN")
  schedule_indicator  a yes/no property of a whole month's schedule ("at least
                      one layover in SEA"). These cannot be scored pairing by
                      pairing; they are collected in the bid and left for the
                      PBS stage.

Phi holds only the pairing-level columns. Each column is min-max normalised
over the pairing set J, so phi~_jk is in [0, 1]. A column that is constant
over J normalises to 0 everywhere: it cannot separate pairings, so it must not
move any score.

Calendar columns are one per day of the bid month, and airport columns one per
layover airport that occurs in J, so the library depends on the instance and
is built from it (FeatureLibrary.for_pairings).
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from models import Pairing

CONTINUOUS = "continuous"
PAIRING_INDICATOR = "pairing_indicator"
SCHEDULE_INDICATOR = "schedule_indicator"

# Thresholds behind the fixed indicator columns.
EARLY_REPORT_BEFORE = time(6, 0)
COMFORTABLE_REPORT = time(7, 0)            # the base oracle's "no burden" report time
LATE_RELEASE_AFTER = time(20, 0)
EVENING_STARTS = time(18, 0)
REDEYE_WINDOW = (time(0, 0), time(5, 0))   # any time airborne in this window
B767_PAY_PREMIUM = 1.04                    # as in oracle.normalise_pay_scores
MAX_FREE_WEEKENDS = 4

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True)
class Feature:
    key: str
    name: str
    description: str
    sigma: int
    type: str
    unit: str = ""
    family: Optional[str] = None     # parameterised families share a family name
    param: object = None             # the day number or airport code of a family member
    extract: Optional[Callable[[Pairing], float]] = field(
        default=None, compare=False, repr=False
    )
    evaluate: Optional[Callable[[Sequence[Pairing]], bool]] = field(
        default=None, compare=False, repr=False
    )

    @property
    def is_pairing_level(self) -> bool:
        return self.type in (CONTINUOUS, PAIRING_INDICATOR)


# ---------------------------------------------------------------------------
# Raw pairing measurements
# ---------------------------------------------------------------------------

def _dates(p: Pairing) -> List[date]:
    return [p.start_date + timedelta(days=i) for i in range(p.calendar_days)]


def _mins_of_day(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def _airborne(p: Pairing) -> List[Tuple[datetime, datetime]]:
    out = []
    for leg in p.legs:
        dep = p._abs_time(leg.dep_day, leg.dep_time)
        out.append((dep, dep + timedelta(minutes=leg.block_mins)))
    return out


def _is_redeye(p: Pairing) -> bool:
    lo, hi = REDEYE_WINDOW
    for dep, arr in _airborne(p):
        day = dep.date()
        while day <= arr.date():
            w_lo = datetime.combine(day, lo)
            w_hi = datetime.combine(day, hi)
            if dep < w_hi and arr > w_lo:
                return True
            day += timedelta(days=1)
    return False


def _away_evening(p: Pairing, d: date) -> bool:
    """On duty or away from base at some point between 18:00 and midnight of d."""
    ev_lo = datetime.combine(d, EVENING_STARTS)
    ev_hi = datetime.combine(d + timedelta(days=1), time(0, 0))
    return p.report_dt < ev_hi and p.release_dt > ev_lo


def _free_weekends(pairings: Sequence[Pairing], month_start: date, n_days: int) -> int:
    busy = {d for p in pairings for d in _dates(p)}
    count = 0
    for day in range(1, n_days):
        sat = month_start.replace(day=day)
        if sat.weekday() == 5:
            if sat not in busy and sat + timedelta(days=1) not in busy:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

def _fixed_columns() -> List[Feature]:
    """Columns whose definition does not depend on the bid month or airports."""
    F = Feature
    return [
        F("tafb", "Time away from base",
          "Hours from report to release of the trip.", -1, CONTINUOUS, "h",
          extract=lambda p: p.tafb),
        F("hotel_nights", "Hotel nights",
          "Number of nights spent in a hotel away from base.", -1, CONTINUOUS, "nights",
          extract=lambda p: p.nights_away),
        F("calendar_days", "Calendar days covered",
          "Number of calendar days the trip touches, start to end inclusive.", -1,
          CONTINUOUS, "days", extract=lambda p: p.calendar_days),
        F("credit_pay", "Credit pay",
          "Paid credit hours, with the 4% B767 rate premium applied "
          "(proportional to the pilot's pay for the trip).", +1, CONTINUOUS,
          "pay-h", extract=lambda p: p.credit_hours * (B767_PAY_PREMIUM if p.aircraft == "B767" else 1.0)),
        F("per_diem", "Per diem",
          "Per-diem allowance in dollars (proportional to TAFB).", +1, CONTINUOUS, "$",
          extract=lambda p: p.per_diem),
        F("block_hours", "Block hours",
          "Hours actually flown.", +1, CONTINUOUS, "h",
          extract=lambda p: p.block_hours),
        F("num_legs", "Number of legs",
          "Number of flights in the trip.", -1, CONTINUOUS, "legs",
          extract=lambda p: p.num_legs),
        F("report_time", "Report time",
          "Clock time of report on the first day; higher = later report.", +1,
          CONTINUOUS, "min after midnight", extract=lambda p: _mins_of_day(p.report_dt)),
        F("report_earliness", "Report earliness",
          f"Minutes the first-day report is earlier than {COMFORTABLE_REPORT:%H:%M} "
          "(0 for a report at or after it); higher = earlier wake-up.", -1, CONTINUOUS, "min",
          extract=lambda p: max(0, COMFORTABLE_REPORT.hour * 60 + COMFORTABLE_REPORT.minute
                                - _mins_of_day(p.report_dt))),
        F("release_time", "Release time",
          "Clock time of release on the last day; higher = later home.", -1,
          CONTINUOUS, "min after midnight", extract=lambda p: _mins_of_day(p.release_dt)),
        F("min_layover_rest", "Shortest layover rest",
          "Shortest hotel rest of the trip, in hours.", +1, CONTINUOUS, "h",
          extract=lambda p: min((r.total_seconds() / 3600 for r in p.layover_rests), default=0.0)),
        F("weekend_days", "Weekend days touched",
          "Number of Saturdays and Sundays the trip touches.", -1, CONTINUOUS, "days",
          extract=lambda p: sum(1 for d in _dates(p) if d.weekday() >= 5)),
        F("early_report", "Early report",
          f"1 if the trip reports before {EARLY_REPORT_BEFORE:%H:%M}.", -1, PAIRING_INDICATOR,
          extract=lambda p: float(p.report_dt.time() < EARLY_REPORT_BEFORE)),
        F("late_release", "Late release",
          f"1 if the trip releases after {LATE_RELEASE_AFTER:%H:%M}.", -1, PAIRING_INDICATOR,
          extract=lambda p: float(p.release_dt.time() > LATE_RELEASE_AFTER)),
        F("redeye", "Red-eye",
          "1 if any leg is airborne between 00:00 and 05:00.", -1, PAIRING_INDICATOR,
          extract=lambda p: float(_is_redeye(p))),
        F("aircraft_b767", "Flies the B767",
          "1 if the trip is on the B767 (wide-body), 0 for the B737.", +1, PAIRING_INDICATOR,
          extract=lambda p: float(p.aircraft == "B767")),
    ]


class FeatureLibrary:
    """The column library K for one bid month and one set of layover airports."""

    def __init__(
        self,
        month_start: date,
        layover_airports: Iterable[str],
        airport_names: Optional[Dict[str, str]] = None,
    ):
        self.month_start = month_start.replace(day=1)
        self.days_in_month = calendar.monthrange(month_start.year, month_start.month)[1]
        self.layover_airports = tuple(sorted(set(layover_airports)))
        self.airport_names = dict(airport_names or {})
        self.columns: Dict[str, Feature] = {}
        for f in self._build():
            self.columns[f.key] = f

    @classmethod
    def for_pairings(cls, pairings: Sequence[Pairing], month_start: date) -> "FeatureLibrary":
        from generator import AIRPORTS

        names = {a["code"]: a["city"] for a in AIRPORTS}
        codes = {c for p in pairings for c in p.layover_airports}
        return cls(month_start, codes, names)

    # ------------------------------------------------------------------
    def day(self, d: int) -> date:
        return self.month_start.replace(day=d)

    def day_label(self, d: int) -> str:
        return f"{_WEEKDAYS[self.day(d).weekday()]} {d}"

    def _build(self) -> List[Feature]:
        cols = _fixed_columns()
        for d in range(1, self.days_in_month + 1):
            dd = self.day(d)
            cols.append(Feature(
                f"touches_day_{d}", f"Touches {self.day_label(d)}",
                f"1 if the trip is on duty or away from base at any time on {self.day_label(d)}.",
                -1, PAIRING_INDICATOR, family="touches_day", param=d,
                extract=lambda p, dd=dd: float(p.start_date <= dd <= p.end_date),
            ))
        for d in range(1, self.days_in_month + 1):
            dd = self.day(d)
            cols.append(Feature(
                f"away_evening_{d}", f"Away on the evening of {self.day_label(d)}",
                f"1 if the trip is on duty or away from base between "
                f"{EVENING_STARTS:%H:%M} and midnight on {self.day_label(d)}.",
                -1, PAIRING_INDICATOR, family="away_evening", param=d,
                extract=lambda p, dd=dd: float(_away_evening(p, dd)),
            ))
        for code in self.layover_airports:
            city = self.airport_names.get(code, code)
            cols.append(Feature(
                f"layover_{code}", f"Layover in {city}",
                f"1 if the trip spends a night in {city} ({code}).",
                +1, PAIRING_INDICATOR, family="layover", param=code,
                extract=lambda p, code=code: float(code in p.layover_airports),
            ))
        for code in self.layover_airports:
            city = self.airport_names.get(code, code)
            cols.append(Feature(
                f"sched_layover_{code}", f"At least one layover in {city} this month",
                f"Schedule-level: satisfied if at least one trip of the month overnights in {city} ({code}). "
                "Extra trips there add nothing.",
                +1, SCHEDULE_INDICATOR, family="sched_layover", param=code,
                evaluate=lambda ps, code=code: any(code in p.layover_airports for p in ps),
            ))
        ms, nd = self.month_start, self.days_in_month
        for n in range(1, MAX_FREE_WEEKENDS + 1):
            cols.append(Feature(
                f"sched_free_weekends_{n}", f"At least {n} free weekend(s)",
                f"Schedule-level: satisfied if at least {n} Saturday+Sunday pair(s) of the month "
                "are completely free of trips, whichever they are.",
                +1, SCHEDULE_INDICATOR, family="sched_free_weekends", param=n,
                evaluate=lambda ps, n=n: _free_weekends(ps, ms, nd) >= n,
            ))
        return cols

    # ------------------------------------------------------------------
    def __contains__(self, key: str) -> bool:
        return key in self.columns

    def __getitem__(self, key: str) -> Feature:
        return self.columns[key]

    def pairing_level(self) -> List[Feature]:
        return [f for f in self.columns.values() if f.is_pairing_level]

    def schedule_level(self) -> List[Feature]:
        return [f for f in self.columns.values() if f.type == SCHEDULE_INDICATOR]

    def month_calendar(self) -> str:
        """One line per week, e.g. 'Thu 1  Fri 2  Sat 3  Sun 4'."""
        rows, cur = [], []
        for d in range(1, self.days_in_month + 1):
            cur.append(self.day_label(d))
            if self.day(d).weekday() == 6:
                rows.append("  ".join(cur))
                cur = []
        if cur:
            rows.append("  ".join(cur))
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Phi
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureMatrix:
    """
    Phi over a pairing set: raw values, bounds over J, and normalised values.

    raw[j][k] is the measurement in the column's unit; norm[j][k] is
    (raw - min_J) / (max_J - min_J), or 0 for a column constant over J.
    """
    pairing_ids: Tuple[int, ...]
    keys: Tuple[str, ...]
    raw: Dict[int, Dict[str, float]]
    norm: Dict[int, Dict[str, float]]
    bounds: Dict[str, Tuple[float, float]]

    @classmethod
    def build(cls, library: FeatureLibrary, pairings: Sequence[Pairing]) -> "FeatureMatrix":
        cols = library.pairing_level()
        keys = tuple(f.key for f in cols)
        raw = {p.id: {f.key: float(f.extract(p)) for f in cols} for p in pairings}
        bounds: Dict[str, Tuple[float, float]] = {}
        for k in keys:
            vals = [raw[p.id][k] for p in pairings]
            bounds[k] = (min(vals), max(vals))
        norm: Dict[int, Dict[str, float]] = {}
        for p in pairings:
            row = {}
            for k in keys:
                lo, hi = bounds[k]
                row[k] = 0.0 if hi == lo else (raw[p.id][k] - lo) / (hi - lo)
            norm[p.id] = row
        return cls(tuple(p.id for p in pairings), keys, raw, norm, bounds)

    def is_constant(self, key: str) -> bool:
        lo, hi = self.bounds[key]
        return lo == hi

    def column(self, key: str, normalised: bool = False) -> Dict[int, float]:
        src = self.norm if normalised else self.raw
        return {pid: src[pid][key] for pid in self.pairing_ids}
