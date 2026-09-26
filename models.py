"""
models.py
---------
Dataclasses for the pilot bidding simulation.
Mirrors the data structures in the HTML POC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from evaluator import BradleyTerryModel


# ---------------------------------------------------------------------------
# Duty-time conventions
# ---------------------------------------------------------------------------

# All leg times are on one clock (the home base's local time); the airport
# time zones in generator.AIRPORTS are not applied.
REPORT_BEFORE_MINS = 60   # report 60 min before first departure
RELEASE_AFTER_MINS = 30   # released 30 min after last arrival (debrief)

_WEEKDAYS = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')


# ---------------------------------------------------------------------------
# Oracle weights
# ---------------------------------------------------------------------------

@dataclass
class OracleWeights:
    """
    Preference weights used by the oracle scorer.
    Four factors — must sum to 100.

    Derivation from pilot profile (as in the POC):
      - tafb, hotel_nights: derived from family status and age
      - report_time:        fixed at 12 for all pilots
      - credit_pay:         remainder after tafb + hotel_nights + report_time
                            (absorbs the former aircraft weight)
    """
    tafb:         int = 20
    hotel_nights: int = 18
    report_time:  int = 12
    credit_pay:   int = 50   # fills to 100

    def total(self) -> int:
        return self.tafb + self.hotel_nights + self.report_time + self.credit_pay

    def validate(self) -> bool:
        return self.total() == 100


# ---------------------------------------------------------------------------
# Leg (single flight within a pairing)
# ---------------------------------------------------------------------------

@dataclass
class Leg:
    dep:        str   # IATA code e.g. 'BOS'
    arr:        str
    dep_city:   str
    arr_city:   str
    distance_mi: int
    block_mins: int   # block time in minutes
    dep_day:    int   # day within the pairing (1-indexed)
    dep_time:   str   # human-readable e.g. '9:00 AM'
    arr_day:    int
    arr_time:   str

    @property
    def block_hours(self) -> float:
        return round(self.block_mins / 60, 1)


# ---------------------------------------------------------------------------
# Pairing (multi-leg trip, circular — starts and ends at base)
# ---------------------------------------------------------------------------

@dataclass
class Pairing:
    id:             int
    aircraft:       str    # 'B737' or 'B767'
    aircraft_label: str    # 'Boeing 737' or 'Boeing 767'
    base:           str    # home base airport e.g. 'BOS'
    legs:           List[Leg]
    nights_away:    int
    start_date:     date   # calendar date of Day1 (the first departure)
    hotel_quality:  str = 'Standard'  # fixed per POC decision
    min_seniority:  int = 999         # unused — all open to all pilots

    # ------------------------------------------------------------------
    # Calendar placement
    # ------------------------------------------------------------------

    @property
    def start_dow(self) -> str:
        """Day of week of Day1, e.g. 'Tue'."""
        return _WEEKDAYS[self.start_date.weekday()]

    @property
    def start_label(self) -> str:
        """Weekday and ISO date, e.g. 'Tue 2026-10-06'."""
        return f"{self.start_dow} {self.start_date.isoformat()}"

    @property
    def end_date(self) -> date:
        """Calendar date of the last arrival."""
        return self.start_date + timedelta(days=self.legs[-1].arr_day - 1)

    @property
    def calendar_days(self) -> int:
        """Number of calendar days the trip touches, start to end inclusive."""
        return self.legs[-1].arr_day

    def _abs_time(self, day: int, time_str: str) -> datetime:
        base = datetime.combine(self.start_date, datetime.min.time())
        return base + timedelta(days=day - 1, minutes=self._time_to_mins(time_str))

    @property
    def report_dt(self) -> datetime:
        """Start of duty: REPORT_BEFORE_MINS before the first departure."""
        first = self.legs[0]
        return self._abs_time(first.dep_day, first.dep_time) - timedelta(minutes=REPORT_BEFORE_MINS)

    @property
    def release_dt(self) -> datetime:
        """End of duty: RELEASE_AFTER_MINS after the last arrival."""
        last = self.legs[-1]
        return self._abs_time(last.arr_day, last.arr_time) + timedelta(minutes=RELEASE_AFTER_MINS)

    @property
    def layover_rests(self) -> List[timedelta]:
        """Hotel rest at each overnight: release after one day's last leg to report for the next."""
        rests = []
        for a, b in zip(self.legs, self.legs[1:]):
            if b.dep_day > a.arr_day:
                off = self._abs_time(a.arr_day, a.arr_time) + timedelta(minutes=RELEASE_AFTER_MINS)
                on  = self._abs_time(b.dep_day, b.dep_time) - timedelta(minutes=REPORT_BEFORE_MINS)
                rests.append(on - off)
        return rests

    @property
    def num_legs(self) -> int:
        return len(self.legs)

    @property
    def total_block_mins(self) -> int:
        return sum(l.block_mins for l in self.legs)

    @property
    def block_hours(self) -> float:
        return round(self.total_block_mins / 60, 1)

    @property
    def credit_hours(self) -> float:
        """Credit hours = max(block, 1-for-3.5 rotation rig) per Section 12 K."""
        rotation_credit = self.tafb / 3.5
        return round(max(self.block_hours, rotation_credit), 1)

    @property
    def tafb(self) -> float:
        """
        Time Away From Base in hours.
        Computed from first departure to last arrival across all days.
        """
        first_dep = self.legs[0].dep_day * 1440 + self._time_to_mins(self.legs[0].dep_time)
        last_arr  = self.legs[-1].arr_day * 1440 + self._time_to_mins(self.legs[-1].arr_time)
        return round((last_arr - first_dep) / 60, 1)

    @property
    def report_time_mins(self) -> int:
        """Report time = 60 minutes before first departure."""
        return self._time_to_mins(self.legs[0].dep_time) - 60

    @property
    def per_diem(self) -> int:
        """Per diem = TAFB × $2.85, per Delta 2023 Section 5."""
        return round(self.tafb * 2.85)

    @property
    def cities(self) -> List[str]:
        return list(dict.fromkeys([l.dep_city for l in self.legs] + [self.legs[-1].arr_city]))

    def pay_for_pilot(self, base_pay: float) -> int:
        """Total block pay for a specific pilot's hourly rate."""
        return round(self.credit_hours * base_pay)

    def total_trip_value(self, base_pay: float) -> int:
        """Block pay + per diem."""
        return self.pay_for_pilot(base_pay) + self.per_diem

    def is_qualified(self, pilot: 'Pilot') -> bool:
        return self.aircraft in pilot.qualified_types

    @staticmethod
    def _time_to_mins(time_str: str) -> int:
        """Convert '9:00 AM' → minutes from midnight."""
        parts = time_str.split()
        h, m  = map(int, parts[0].split(':'))
        if parts[1] == 'PM' and h != 12:
            h += 12
        if parts[1] == 'AM' and h == 12:
            h = 0
        return h * 60 + m

    def report_time_str(self) -> str:
        mins = ((self.report_time_mins % 1440) + 1440) % 1440
        h, m = divmod(mins, 60)
        ampm = 'AM' if h < 12 else 'PM'
        h12  = h if 1 <= h <= 12 else (12 if h == 0 else h - 12)
        return f"{h12}:{m:02d} {ampm}"


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------

@dataclass
class Pilot:
    id:              int
    name:            str
    age:             int
    family_status:   str   # e.g. 'Single', 'Married, 2+ kids'
    seniority:       int   # 1 = most senior
    home_base:       str   # e.g. 'BOS'
    qualified_types: List[str]  # e.g. ['B737', 'B767']
    min_rest:        int   # minimum rest hours required
    base_pay:        float # hourly rate in USD
    weights:         OracleWeights = field(default_factory=OracleWeights)

    @property
    def has_kids(self) -> bool:
        status = self.family_status.lower()
        if 'no kids' in status or 'no child' in status:
            return False
        return any(k in status for k in ('child', 'kids', 'parent'))

    @property
    def is_mid_career(self) -> bool:
        return self.age >= 40

    def auto_weights(self) -> OracleWeights:
        """
        Derive oracle weights from pilot profile.
        Mirrors the JavaScript buildPilots() logic in the POC.
        """
        hotel_w  = 26 if self.has_kids else (16 if self.is_mid_career else 9)
        tafb_w   = 22 if self.has_kids else (15 if self.is_mid_career else 9)
        report_w = 12  # fixed for all pilots
        credit_w = 100 - hotel_w - tafb_w - report_w
        return OracleWeights(
            tafb=tafb_w,
            hotel_nights=hotel_w,
            report_time=report_w,
            credit_pay=credit_w
        )

    def can_fly(self, pairing: Pairing) -> bool:
        return pairing.aircraft in self.qualified_types


# ---------------------------------------------------------------------------
# Ranked pairing (output of oracle scoring)
# ---------------------------------------------------------------------------

@dataclass
class RankedPairing:
    pairing:   Pairing
    pilot:     Pilot
    qualified: bool
    oracle_score: int
    oracle_rank:  int = 0   # set after sorting

    @property
    def eligible(self) -> bool:
        return self.qualified


# ---------------------------------------------------------------------------
# LLM response (parsed from JSON)
# ---------------------------------------------------------------------------

@dataclass
class LLMPairingRank:
    pairing_id:   int
    rank:         int
    eligible:     bool
    short_reason: str
    pros:         List[str]
    cons:         List[str]


@dataclass
class ScoredPairing:
    """
    Result of an independent scoring call for one pilot+pairing.
    One LLM API call produces one ScoredPairing.
    Scores are sorted externally to produce a ranking.
    """
    pairing:      Pairing
    score:        int         # 0–100
    eligible:     bool
    short_reason: str
    pros:         List[str]
    cons:         List[str]
    call_idx:     int         # index of the API call (for debugging)


@dataclass
class LineScoringResult:
    """
    Aggregated result of scoring a single pilot×line combination over n_runs
    independent LLM calls.

    Stability thresholds (applied to score std across runs):
      std < 5   → "stable"   — mean score used directly
      5–10      → "marginal" — mean score used with a warning
      > 10      → "unstable" — pairwise tiebreak used if adjacent scores overlap

    When ranking_method is "pairwise_tiebreak", a head-to-head comparison was
    run against an adjacent line and the result overrides the score ordering.
    """
    line:           Line
    score_runs:     List[int]       # raw score from each run
    score_mean:     float           # mean across runs
    score_std:      float           # std across runs
    stability:      str             # "stable", "marginal", or "unstable"
    eligible:       bool
    short_reason:   str
    pros:           List[str]
    cons:           List[str]
    ranking_method: str = "score"   # "score" or "pairwise_tiebreak"


@dataclass
class EvalMetrics:
    spearman:     float
    top1_match:   bool
    elig_accuracy: float  # 0.0–1.0
    overall:      int     # composite for logging


@dataclass
class AllocationResult:
    pilot:        Pilot
    pairing:      Optional[Pairing]
    rank_awarded: int    # which rank choice the pilot got (1=first choice)
    bumped_by:    List[str] = field(default_factory=list)  # names of pilots who took higher choices
    line:         Optional["Line"] = None  # set when operating in line-bidding mode


# ---------------------------------------------------------------------------
# Line (monthly schedule — group of pairings)
# ---------------------------------------------------------------------------

@dataclass
class Line:
    """
    A pre-assembled monthly schedule consisting of pairings.
    In real airline bidding the line — not the individual pairing — is the
    atomic unit that pilots bid on.
    """
    id:       int
    pairings: List[Pairing]  # ordered list; all pairings in this line

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    @property
    def total_credit_hours(self) -> float:
        """Sum of credit hours across all pairings."""
        return round(sum(p.credit_hours for p in self.pairings), 1)

    @property
    def total_nights_away(self) -> int:
        """Sum of hotel nights across all pairings."""
        return sum(p.nights_away for p in self.pairings)

    @property
    def total_tafb(self) -> float:
        """Sum of TAFB hours across all pairings."""
        return round(sum(p.tafb for p in self.pairings), 1)

    @property
    def total_block_hours(self) -> float:
        """Sum of block hours across all pairings."""
        return round(sum(p.block_hours for p in self.pairings), 1)

    @property
    def total_per_diem(self) -> int:
        """Sum of per-diem dollars across all pairings."""
        return sum(p.per_diem for p in self.pairings)

    @property
    def aircraft_types(self) -> List[str]:
        """Unique aircraft types used, in order of first appearance."""
        seen: List[str] = []
        for p in self.pairings:
            if p.aircraft not in seen:
                seen.append(p.aircraft)
        return seen

    @property
    def start_days(self) -> List[str]:
        """Start days of week for each pairing."""
        return [p.start_dow for p in self.pairings]

    @property
    def cities(self) -> List[str]:
        """All unique cities visited across all pairings."""
        seen: List[str] = []
        for p in self.pairings:
            for c in p.cities:
                if c not in seen:
                    seen.append(c)
        return seen

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def has_conflicts(self) -> bool:
        """True if the duty periods of any two pairings in this line overlap."""
        return any(v.kind == 'overlap' for v in schedule_violations(self.pairings))

    def schedule_violations(self, min_rest_hours: float = 0) -> List["ScheduleViolation"]:
        """Overlaps and rest shortfalls between this line's pairings."""
        return schedule_violations(self.pairings, min_rest_hours)

    def is_legal_for(self, pilot: "Pilot") -> bool:
        """No overlaps and at least pilot.min_rest hours at base between trips."""
        return not self.schedule_violations(pilot.min_rest)

    # ------------------------------------------------------------------
    # Pilot-specific helpers
    # ------------------------------------------------------------------

    def total_pay_for_pilot(self, pilot: "Pilot") -> int:
        """Sum of block pay (credit hours × base rate) across all pairings."""
        return sum(p.pay_for_pilot(pilot.base_pay) for p in self.pairings)

    def total_trip_value_for_pilot(self, pilot: "Pilot") -> int:
        """Block pay + per diem across all pairings."""
        return self.total_pay_for_pilot(pilot) + self.total_per_diem

    def is_qualified(self, pilot: "Pilot") -> bool:
        """True if the pilot is qualified for every aircraft type in this line."""
        return all(p.is_qualified(pilot) for p in self.pairings)


# ---------------------------------------------------------------------------
# Schedule legality (any set of pairings flown by one pilot)
# ---------------------------------------------------------------------------

@dataclass
class ScheduleViolation:
    """
    A problem between two pairings flown by the same pilot, in time order.

    kind 'overlap': `later` reports before `earlier` is released.
    kind 'rest':    the rest at base between them is under the required minimum.
    """
    kind:    str
    earlier: Pairing
    later:   Pairing
    rest:    timedelta   # later.report_dt - earlier.release_dt (negative for overlaps)


def rest_between(earlier: Pairing, later: Pairing) -> timedelta:
    """Rest at base from `earlier`'s release to `later`'s report."""
    return later.report_dt - earlier.release_dt


def schedule_violations(
    pairings: Sequence[Pairing], min_rest_hours: float = 0,
) -> List[ScheduleViolation]:
    """
    Every overlap and rest shortfall in a schedule, in time order.

    Overlaps are checked between all pairs, so a long trip that swallows a
    later short one is caught. Rest is checked between consecutive
    non-overlapping trips. min_rest_hours=0 checks overlaps only.
    """
    ordered = sorted(pairings, key=lambda p: p.report_dt)
    min_rest = timedelta(hours=min_rest_hours)
    out: List[ScheduleViolation] = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b.report_dt >= a.release_dt:
                break  # sorted by report time: nothing later overlaps a
            out.append(ScheduleViolation('overlap', a, b, rest_between(a, b)))
    for a, b in zip(ordered, ordered[1:]):
        gap = rest_between(a, b)
        if timedelta(0) <= gap < min_rest:
            out.append(ScheduleViolation('rest', a, b, gap))
    return out


def is_legal_schedule(pairings: Sequence[Pairing], min_rest_hours: float) -> bool:
    """True if the pairings can all be flown by one pilot with this minimum rest."""
    return not schedule_violations(pairings, min_rest_hours)


# ---------------------------------------------------------------------------
# Ranked line (output of oracle line scoring)
# ---------------------------------------------------------------------------

@dataclass
class RankedLine:
    """Oracle ranking result for a single pilot–line combination."""
    line:         Line
    pilot:        Pilot
    qualified:    bool
    oracle_score: int
    oracle_rank:  int = 0   # assigned after sorting


# ---------------------------------------------------------------------------
# LLM line response (parsed from JSON)
# ---------------------------------------------------------------------------

@dataclass
class LLMLineRank:
    """Parsed LLM response for a single line in oracle line-ranking mode."""
    line_id:      int
    rank:         int
    eligible:     bool
    short_reason: str
    pros:         List[str]
    cons:         List[str]


# ---------------------------------------------------------------------------
# Adaptive pairwise comparison models
# ---------------------------------------------------------------------------

@dataclass
class PairwiseComparison:
    item_a_id:  int     # line or pairing id
    item_b_id:  int
    winner_id:  int     # must be item_a_id or item_b_id
    confidence: str     # high / medium / low
    reason:     str
    round_num:  int     # 1 = initial, 2 = adaptive follow-up


@dataclass
class AdaptivePairwiseResult:
    pilot:            Pilot
    comparisons:      List[PairwiseComparison]
    bt_model:         BradleyTerryModel
    final_ranking:    List[Tuple[int, float]]   # (item_id, strength)
    rank_positions:   Dict[int, int]            # item_id → rank
    confidence_intervals: Dict[int, Tuple[float, float]]
    fit_quality:      float
    n_comparisons:    int
    n_rounds:         int
    consistency_note: str  # e.g. "fit quality 0.82, comparable to human baseline"
