"""
bid_prompts.py
--------------
Prompts for the column-selection and column-weighting bid (research note,
section 3 and Steps 4-5).

Stages:
  selection   z_p = LLM_sel(theta_p, iota_p)            — never sees the trips
  weighting   w_p = LLM_wgt(theta_p, iota_p, z_p, I_p)  — I_p set by the regime
  joint       (z_p, w_p) in one call
  increment   R3: spend alpha*B more points with the trips in view
  adjust      R4: revise weights after feedback on the shortlist

Every prompt states its stage on a line "Stage: <name>" and, where it asks for
weights, "Budget: <B>" and "Columns to weight: <keys>". The model sees these as
plain context; the scripted oracle responder used in tests and dry runs reads
them.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from features import CONTINUOUS, PAIRING_INDICATOR, SCHEDULE_INDICATOR, FeatureLibrary, FeatureMatrix
from models import Pairing, Pilot

_CLOCK_COLUMNS = {"report_time", "release_time"}


# ---------------------------------------------------------------------------
# Shared blocks
# ---------------------------------------------------------------------------

def _header(library: FeatureLibrary) -> str:
    month = library.month_start.strftime("%B %Y")
    return (
        f"You are preparing a Preferential Bidding System (PBS) bid for an airline pilot for {month}.\n"
        "A PBS bid does not rank trips by hand. It activates the bid columns (trip properties) that "
        "matter to this pilot and gives them weights; every trip is then scored as\n"
        "  score(trip) = sum over columns of  weight x direction x (column value rescaled to 0-1 over all trips)\n"
        "and the trips are ranked by score. Trips under a hard exclusion go to the bottom."
    )


def profile_block(pilot: Pilot, include_priorities: bool) -> str:
    from prompt_builder import _pilot_profile, _priority_list

    out = "PILOT PROFILE (stable from month to month)\n" + _pilot_profile(pilot)
    if include_priorities:
        prio = _priority_list(pilot).replace(
            "the pay figures shown for each line", "the credit_pay column"
        )
        out += "\n\nUSUAL PRIORITIES FOR THIS PROFILE\n" + prio
    return out


def instructions_block(instructions: Sequence) -> str:
    head = ("MONTHLY INSTRUCTIONS (this month only; where they conflict with the usual "
            "priorities, the instructions win)")
    if not instructions:
        return head + "\n  (none this month)"
    return head + "\n" + "\n".join(f"  {n}. {i.text}" for n, i in enumerate(instructions, 1))


def library_block(library: FeatureLibrary) -> str:
    def dir_label(s: int) -> str:
        return "+1" if s > 0 else "-1"

    fixed = [f for f in library.columns.values() if f.family is None]
    cont = [f for f in fixed if f.type == CONTINUOUS]
    ind = [f for f in fixed if f.type == PAIRING_INDICATOR]
    lines = [
        "FEATURE LIBRARY (bid columns). Use the keys exactly as written. "
        "Direction +1 = higher values are better, -1 = lower values are better; "
        "the default shown is the usual case, you choose per pilot.",
        "",
        "Continuous columns:",
    ]
    for f in cont:
        unit = f" [{f.unit}]" if f.unit else ""
        lines.append(f"  {f.key}{unit} — {f.description} Default {dir_label(f.sigma)}.")
    lines += ["", "Indicator columns (1 or 0 per trip):"]
    for f in ind:
        lines.append(f"  {f.key} — {f.description} Default {dir_label(f.sigma)}.")
    n = library.days_in_month
    codes = ", ".join(f"{c} ({library.airport_names.get(c, c)})" for c in library.layover_airports)
    lines += [
        "",
        f"Calendar indicator columns (one per day d = 1..{n} of the month):",
        "  touches_day_<d> — 1 if the trip is on duty or away from base at any time on day d. Default -1.",
        "  away_evening_<d> — 1 if the trip is on duty or away from base between 18:00 and midnight "
        "on day d. Default -1.",
        "",
        "Layover indicator columns (one per airport):",
        f"  layover_<CODE> — 1 if the trip spends a night in CODE. Codes: {codes}.",
        "",
        "Schedule-level columns (about the whole month's schedule, not one trip; they get no weight "
        "and no direction and are passed to the solver as yes/no goals):",
        "  sched_layover_<CODE> — at least one layover in CODE during the month (one is enough).",
        "  sched_free_weekends_<n> — at least n Saturday+Sunday pairs completely free, whichever "
        "they are (n = 1..4).",
        "",
        "Calendar of the bid month:",
        library.month_calendar(),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trip tables (information shown in R2 / R3 / R4)
# ---------------------------------------------------------------------------

def _fmt(key: str, value: float) -> str:
    if key in _CLOCK_COLUMNS:
        m = int(round(value))
        return f"{m // 60:02d}:{m % 60:02d}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _dates(p: Pairing) -> str:
    a, b = p.start_date, p.end_date
    return f"{a:%a} {a.day}-{b:%a} {b.day}"


def trip_table(
    phi: FeatureMatrix,
    pairings: Sequence[Pairing],
    keys: Sequence[str],
    library: FeatureLibrary,
    pilot: Pilot,
) -> str:
    """One row per trip with the raw values of the given columns."""
    cols = list(keys)
    head = "trip | dates | aircraft | " + " | ".join(cols)
    rows = [head]
    for p in pairings:
        qual = "" if p.is_qualified(pilot) else " (NOT QUALIFIED)"
        vals = " | ".join(_fmt(k, phi.raw[p.id][k]) for k in cols)
        rows.append(f"P{p.id} | {_dates(p)} | {p.aircraft}{qual} | {vals}")
    return "\n".join(rows)


def compact_trip_table(
    phi: FeatureMatrix,
    pairings: Sequence[Pairing],
    library: FeatureLibrary,
    pilot: Pilot,
) -> str:
    """
    All trips before any column is selected (joint bid under R2): every
    continuous column and fixed indicator, with the calendar and layover
    families summarised as lists.
    """
    fixed = [f.key for f in library.columns.values()
             if f.family is None and f.is_pairing_level]
    head = ("trip | dates | aircraft | " + " | ".join(fixed)
            + " | evenings away (days) | layovers")
    rows = [head]
    n = library.days_in_month
    for p in pairings:
        qual = "" if p.is_qualified(pilot) else " (NOT QUALIFIED)"
        vals = " | ".join(_fmt(k, phi.raw[p.id][k]) for k in fixed)
        evenings = ",".join(str(d) for d in range(1, n + 1) if phi.raw[p.id][f"away_evening_{d}"])
        lays = ",".join(p.layover_airports)
        rows.append(f"P{p.id} | {_dates(p)} | {p.aircraft}{qual} | {vals} | {evenings} | {lays}")
    note = ("touches_day_<d> is 1 for every day in the trip's dates, away_evening_<d> for the "
            "listed evenings, layover_<CODE> for the listed layovers.")
    return "\n".join(rows) + "\n" + note


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

_SELECTION_RULES = """\
- Include the columns that reflect the pilot's stable priorities, and every column needed to
  express each monthly instruction. A date request maps to calendar columns (e.g. "the weekend
  of Saturday the 17th" -> touches_day_17 and touches_day_18). A qualitative request activates or
  re-directs an existing column (e.g. "long trips" -> tafb with direction +1).
- Give each column a direction: +1 if higher values are better for this pilot, -1 if lower
  values are better.
- Set "hard_exclusion": true only for a firm requirement ("I cannot", "I must be off"); trips with
  that property will be placed at the bottom of the ranking. Hard exclusions must be indicator
  columns (1/0), and they get no weight.
- A wish about the month as a whole (e.g. "at least one layover in X", "two free weekends")
  goes in as a schedule-level column.
- Leave out columns that do not matter to this pilot; a few well-chosen columns beat many."""


def selection_prompt(
    pilot: Pilot, instructions: Sequence, library: FeatureLibrary, include_priorities: bool,
) -> str:
    return "\n\n".join([
        _header(library),
        "Stage: selection",
        profile_block(pilot, include_priorities),
        instructions_block(instructions),
        library_block(library),
        "TASK\nSelect the columns that should drive this pilot's bid this month.\n" + _SELECTION_RULES,
        'Return JSON only:\n{"selected": [{"key": "tafb", "direction": -1, "hard_exclusion": false, '
        '"rationale": "short reason"}, ...]}',
    ])


# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------

def _selected_block(
    library: FeatureLibrary,
    directions: Mapping[str, int],
    exclusions: Sequence[str],
    schedule_prefs: Sequence[str],
) -> str:
    lines = ["SELECTED COLUMNS"]
    for k, s in directions.items():
        word = "higher is better" if s > 0 else "lower is better"
        lines.append(f"  - {k} (direction {'+1' if s > 0 else '-1'}: {word}) — {library[k].description}")
    if exclusions:
        lines.append("Hard exclusions (already handled, no weight): " + ", ".join(exclusions))
    if schedule_prefs:
        lines.append("Schedule-level goals (passed to the solver, no weight): " + ", ".join(schedule_prefs))
    return "\n".join(lines)


def _weight_task(keys: Sequence[str], budget: float, weight_format: str, what: str) -> str:
    cols = ", ".join(keys)
    if weight_format == "rank":
        return (
            f"TASK\nRank the columns to weight from most to least important for this pilot "
            f"{what}. Every column exactly once.\n"
            f"Columns to weight: {cols}\n\n"
            'Return JSON only:\n{"ranking": ["most_important_key", "...", "least_important_key"]}'
        )
    return (
        f"TASK\nDistribute a budget of {budget:g} points over the columns to weight according to "
        f"how much each matters to this pilot {what}. Weights are non-negative numbers and must "
        f"sum to exactly {budget:g}. A column's weight scales its effect on every trip's score.\n"
        f"Budget: {budget:g}\nColumns to weight: {cols}\n\n"
        'Return JSON only:\n{"weights": {"key": number, ...}}'
    )


def weighting_prompt(
    pilot: Pilot,
    instructions: Sequence,
    library: FeatureLibrary,
    directions: Mapping[str, int],
    exclusions: Sequence[str],
    schedule_prefs: Sequence[str],
    budget: float,
    weight_format: str,
    include_priorities: bool,
    table: Optional[str] = None,
) -> str:
    info = (
        "TRIPS THIS MONTH (values of the selected columns; use them to see where the trips "
        "actually differ)\n" + table
        if table is not None else
        "You do not see this month's trips. Weight from the profile and the instructions alone."
    )
    return "\n\n".join([
        _header(library),
        "Stage: weighting",
        profile_block(pilot, include_priorities),
        instructions_block(instructions),
        _selected_block(library, directions, exclusions, schedule_prefs),
        info,
        _weight_task(list(directions), budget, weight_format, "this month"),
    ])


# ---------------------------------------------------------------------------
# Joint
# ---------------------------------------------------------------------------

def joint_prompt(
    pilot: Pilot,
    instructions: Sequence,
    library: FeatureLibrary,
    budget: float,
    weight_format: str,
    include_priorities: bool,
    table: Optional[str] = None,
) -> str:
    info = (
        "TRIPS THIS MONTH\n" + table if table is not None else
        "You do not see this month's trips. Decide from the profile and the instructions alone."
    )
    if weight_format == "rank":
        amount = ('"rank": integer importance rank (1 = most important) on every column that is '
                  "neither a hard exclusion nor schedule-level; the ranks form 1..n with no repeats")
        example = '"rank": 1'
    else:
        amount = (f'"weight": a non-negative number on every column that is neither a hard exclusion '
                  f"nor schedule-level; these weights must sum to exactly {budget:g}")
        example = '"weight": 4'
    return "\n\n".join([
        _header(library),
        "Stage: joint",
        profile_block(pilot, include_priorities),
        instructions_block(instructions),
        library_block(library),
        info,
        "TASK\nSelect the columns that should drive this pilot's bid this month and, in the same "
        "answer, say how much each matters.\n" + _SELECTION_RULES + f"\n- Give {amount}.\n"
        f"Budget: {budget:g}",
        'Return JSON only:\n{"selected": [{"key": "tafb", "direction": -1, "hard_exclusion": false, '
        f'{example}, "rationale": "short reason"}}, ...]}}',
    ])


# ---------------------------------------------------------------------------
# R3 increment and R4 adjustment
# ---------------------------------------------------------------------------

def _weights_line(weights: Mapping[str, float]) -> str:
    return ", ".join(f"{k}: {w:.2f}" for k, w in weights.items())


def increment_prompt(
    pilot: Pilot,
    instructions: Sequence,
    library: FeatureLibrary,
    directions: Mapping[str, int],
    exclusions: Sequence[str],
    schedule_prefs: Sequence[str],
    blind_weights: Mapping[str, float],
    increment_budget: float,
    include_priorities: bool,
    table: str,
) -> str:
    total = sum(blind_weights.values())
    return "\n\n".join([
        _header(library),
        "Stage: increment",
        profile_block(pilot, include_priorities),
        instructions_block(instructions),
        _selected_block(library, directions, exclusions, schedule_prefs),
        f"WEIGHTS ALREADY FIXED without seeing the trips (sum {total:g}):\n  " + _weights_line(blind_weights),
        "TRIPS THIS MONTH (values of the selected columns)\n" + table,
        f"TASK\nYou now see the trips. Allocate {increment_budget:g} additional points over the "
        "selected columns, where seeing the trips changes what matters (e.g. a column on which the "
        "trips barely differ needs little extra weight). The fixed weights stay; your points are "
        f"added to them. Non-negative numbers summing to exactly {increment_budget:g}.\n"
        f"Budget: {increment_budget:g}\nColumns to weight: {', '.join(directions)}",
        'Return JSON only:\n{"increment": {"key": number, ...}}',
    ])


def adjust_prompt(
    pilot: Pilot,
    instructions: Sequence,
    library: FeatureLibrary,
    directions: Mapping[str, int],
    exclusions: Sequence[str],
    schedule_prefs: Sequence[str],
    weights: Mapping[str, float],
    shortlist_table: str,
    feedback: Sequence[Tuple[int, str, str]],
    budget: float,
    weight_format: str,
    include_priorities: bool,
    round_num: int,
) -> str:
    fb = "\n".join(f"  P{pid}: {verdict} — {text}" for pid, verdict, text in feedback)
    return "\n\n".join([
        _header(library),
        f"Stage: adjust (feedback round {round_num})",
        profile_block(pilot, include_priorities),
        instructions_block(instructions),
        _selected_block(library, directions, exclusions, schedule_prefs),
        "CURRENT WEIGHTS\n  " + _weights_line(weights),
        "The current weights put these trips at the top of the ranking (the shortlist):\n" + shortlist_table,
        "PILOT FEEDBACK ON THE SHORTLIST\n" + fb,
        _weight_task(list(directions), budget, weight_format,
                     "so that trips like the ones the pilot liked rise and the others fall"),
    ])
