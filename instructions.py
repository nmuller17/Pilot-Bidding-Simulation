"""
instructions.py
---------------
Monthly instructions iota_p^t (research note, section 2 and Step 1).

An instruction is free text for the LLM plus, for evaluation only, the columns
it maps to:

  date requests        -> calendar columns   ("weekend of Sat 17" -> touches_day_17,
                                               touches_day_18, sigma = -1)
  qualitative requests -> existing columns   ("long trips" -> tafb, sigma = +1)
  schedule-level wishes -> schedule columns  ("two free weekends" -> sched_free_weekends_2)

`firm` marks a requirement ("I cannot work ..."): the oracle turns it into a
hard exclusion. `share` is the fraction of the budget a soft instruction takes
in the oracle bid (bid.oracle_bid). The shares are designer choices, like the
base oracle's weights.

Compliance (section 5, "share of iota met in r_p and x_p") is decided per
instruction by `check_ranking` on a pairing ranking and `check_schedule` on a
month's schedule; see their docstrings for the exact rules.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

from features import CONTINUOUS, FeatureLibrary, FeatureMatrix
from models import Pairing, Pilot

DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class Effect:
    key: str
    sigma: int


@dataclass(frozen=True)
class MonthlyInstruction:
    kind: str
    text: str
    effects: Tuple[Effect, ...]
    share: float = 0.0
    firm: bool = False
    level: str = "pairing"         # "pairing" or "schedule"

    # ------------------------------------------------------------------
    def _is_continuous(self, library: FeatureLibrary) -> bool:
        return library[self.effects[0].key].type == CONTINUOUS

    def _hits(self, phi: FeatureMatrix, pid: int) -> bool:
        """For indicator instructions: pairing has phi = 1 on any effect column."""
        return any(phi.raw[pid][e.key] >= 0.5 for e in self.effects)

    def check_ranking(
        self,
        order: Sequence[int],
        library: FeatureLibrary,
        phi: FeatureMatrix,
        eligible: Sequence[int],
        k: int = DEFAULT_TOP_K,
        other_firm_hits: frozenset = frozenset(),
    ) -> Optional[bool]:
        """
        Is this instruction met by the ranking? Only the pilot's eligible
        pairings count, in ranked order; the top k of them is the shortlist.

          avoid, firm   every pairing hitting the columns is ranked below every
                        eligible pairing that breaks no firm instruction
                        (other_firm_hits: pairings breaking the pilot's other
                        firm instructions, which share the bottom tier)
          avoid, soft   the top k holds no pairing hitting the columns, unless
                        there are fewer than k that don't
          want          at least one of the top k hits the column
          continuous    the top k's mean is on the requested side of the mean
                        over all eligible pairings

        None when it cannot be judged on a ranking: schedule-level
        instructions, or a "want" that no eligible pairing can satisfy.
        """
        if self.level == "schedule":
            return None
        elig = set(eligible)
        ranked = [pid for pid in order if pid in elig]
        top = ranked[:k]

        if self._is_continuous(library):
            e = self.effects[0]
            all_mean = mean(phi.raw[pid][e.key] for pid in ranked)
            top_mean = mean(phi.raw[pid][e.key] for pid in top)
            return top_mean > all_mean if e.sigma > 0 else top_mean < all_mean

        hits = [pid for pid in ranked if self._hits(phi, pid)]
        if self.effects[0].sigma > 0:
            if not hits:
                return None
            return any(self._hits(phi, pid) for pid in top)

        if self.firm:
            pos = {pid: i for i, pid in enumerate(ranked)}
            clean = [pid for pid in ranked
                     if not self._hits(phi, pid) and pid not in other_firm_hits]
            return all(pos[h] > pos[c] for h in hits for c in clean)
        n_clean = len(ranked) - len(hits)
        allowed = max(0, len(top) - n_clean)
        return sum(1 for pid in top if self._hits(phi, pid)) <= allowed

    def check_schedule(
        self,
        schedule: Sequence[Pairing],
        library: FeatureLibrary,
        phi: FeatureMatrix,
        eligible: Sequence[int],
    ) -> bool:
        """
        Is this instruction met by a month's schedule x_p (for the PBS stage)?

          schedule-level  the column's own yes/no evaluation
          avoid           no trip in the schedule hits the columns
          want            at least one trip does
          continuous      the schedule's mean is on the requested side of the
                          mean over the pilot's eligible pairings
        """
        if self.level == "schedule":
            return bool(library[self.effects[0].key].evaluate(schedule))
        ids = [p.id for p in schedule]
        if self._is_continuous(library):
            e = self.effects[0]
            if not ids:
                return False
            all_mean = mean(phi.raw[pid][e.key] for pid in eligible)
            s_mean = mean(phi.raw[pid][e.key] for pid in ids)
            return s_mean > all_mean if e.sigma > 0 else s_mean < all_mean
        if self.effects[0].sigma > 0:
            return any(self._hits(phi, pid) for pid in ids)
        return not any(self._hits(phi, pid) for pid in ids)

    def as_dict(self) -> Dict:
        return {
            "kind": self.kind,
            "text": self.text,
            "firm": self.firm,
            "level": self.level,
            "share": self.share,
            "effects": [{"key": e.key, "sigma": e.sigma} for e in self.effects],
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# kind -> (relative frequency, budget share). Shares are designer choices.
KINDS: Dict[str, Tuple[float, float]] = {
    "weekend_off":         (3, 0.20),
    "day_off":             (3, 0.15),
    "evening_home":        (2, 0.15),
    "long_trips":          (1, 0.20),
    "short_trips":         (1, 0.20),
    "layover_want":        (2, 0.15),
    "layover_avoid":       (1, 0.10),
    "no_early_reports":    (2, 0.15),
    "sched_layover":       (1, 0.0),
    "sched_free_weekends": (1, 0.0),
}
_CONFLICTS = [{"long_trips", "short_trips"}, {"layover_want", "sched_layover"}]
FIRM_PROBABILITY = {"weekend_off": 0.3, "day_off": 0.4}


class InstructionGenerator:
    """
    Draws 0-3 instructions per pilot from the templates below, seeded.

    Dates come from the bid month; airports only from layovers that occur in
    at least two pairings, so a "want" is always satisfiable.
    """

    def __init__(self, library: FeatureLibrary, phi: FeatureMatrix, seed: int = 7):
        self.lib = library
        self.phi = phi
        self.rng = random.Random(f"instructions-{seed}")
        self._layover_codes = [
            c for c in library.layover_airports
            if sum(phi.raw[pid][f"layover_{c}"] for pid in phi.pairing_ids) >= 2
        ]
        early = [phi.raw[pid]["early_report"] for pid in phi.pairing_ids]
        self._early_varies = 0 < sum(early) < len(early)

    # ------------------------------------------------------------------
    def for_pilots(self, pilots: Sequence[Pilot]) -> Dict[int, Tuple[MonthlyInstruction, ...]]:
        return {p.id: self.for_pilot() for p in pilots}

    def for_pilot(self) -> Tuple[MonthlyInstruction, ...]:
        rng = self.rng
        n = rng.choice([0, 1, 1, 2, 2, 3])
        kinds = list(KINDS)
        freqs = [KINDS[k][0] for k in kinds]
        chosen: List[MonthlyInstruction] = []
        used_kinds: set = set()
        used_days: set = set()
        used_codes: set = set()
        attempts = 0
        while len(chosen) < n and attempts < 50:
            attempts += 1
            kind = rng.choices(kinds, weights=freqs)[0]
            if kind in used_kinds or any(kind in g and used_kinds & (g - {kind}) for g in _CONFLICTS):
                continue
            ins = self._make(kind, used_days, used_codes)
            if ins is None:
                continue
            used_kinds.add(kind)
            chosen.append(ins)
        return tuple(chosen)

    # ------------------------------------------------------------------
    def _saturdays(self) -> List[int]:
        return [d for d in range(1, self.lib.days_in_month)
                if self.lib.day(d).weekday() == 5]

    def _make(self, kind: str, used_days: set, used_codes: set) -> Optional[MonthlyInstruction]:
        rng, lib = self.rng, self.lib
        share = KINDS[kind][1]
        firm = rng.random() < FIRM_PROBABILITY.get(kind, 0.0)

        if kind == "weekend_off":
            sats = [d for d in self._saturdays() if not {d, d + 1} & used_days]
            if not sats:
                return None
            d = rng.choice(sats)
            used_days.update({d, d + 1})
            if firm:
                text = rng.choice([
                    f"I cannot work the weekend of Saturday the {_ordinal(d)} — family commitment.",
                    f"I must be off Saturday the {_ordinal(d)} and Sunday the {_ordinal(d + 1)}.",
                ])
            else:
                text = rng.choice([
                    f"I don't want to work on the weekend of the {_ordinal(d)}.",
                    f"If possible, keep the weekend of Saturday the {_ordinal(d)} free.",
                ])
            effects = (Effect(f"touches_day_{d}", -1), Effect(f"touches_day_{d + 1}", -1))
            return MonthlyInstruction(kind, text, effects, share, firm)

        if kind == "day_off":
            days = [d for d in range(1, lib.days_in_month + 1)
                    if lib.day(d).weekday() < 5 and d not in used_days]
            d = rng.choice(days)
            used_days.add(d)
            wd = _WEEKDAY_NAMES[lib.day(d).weekday()]
            if firm:
                text = rng.choice([
                    f"I have a medical appointment on {wd} the {_ordinal(d)} — I cannot fly that day.",
                    f"I must be off on {wd} the {_ordinal(d)}.",
                ])
            else:
                text = rng.choice([
                    f"I'd like {wd} the {_ordinal(d)} off if possible.",
                    f"Please try to keep {wd} the {_ordinal(d)} free.",
                ])
            return MonthlyInstruction(kind, text, (Effect(f"touches_day_{d}", -1),), share, firm)

        if kind == "evening_home":
            days = [d for d in range(1, lib.days_in_month + 1) if d not in used_days]
            d = rng.choice(days)
            used_days.add(d)
            wd = _WEEKDAY_NAMES[lib.day(d).weekday()]
            text = rng.choice([
                f"I prefer to be home for the evening of the {_ordinal(d)}.",
                f"I have a dinner I can't miss on the evening of {wd} the {_ordinal(d)}; I'd like to be home.",
            ])
            return MonthlyInstruction(kind, text, (Effect(f"away_evening_{d}", -1),), share)

        if kind == "long_trips":
            text = rng.choice([
                "This month I would like long trips.",
                "I'd rather do fewer, longer trips this month.",
            ])
            return MonthlyInstruction(kind, text, (Effect("tafb", +1),), share)

        if kind == "short_trips":
            text = rng.choice([
                "This month I'd prefer short trips.",
                "Keep my trips as short as possible this month.",
            ])
            return MonthlyInstruction(kind, text, (Effect("tafb", -1),), share)

        if kind in ("layover_want", "layover_avoid", "sched_layover"):
            codes = [c for c in self._layover_codes if c not in used_codes]
            if not codes:
                return None
            code = rng.choice(codes)
            used_codes.add(code)
            city = lib.airport_names.get(code, code)
            if kind == "layover_want":
                text = rng.choice([
                    f"I'd love a layover in {city}.",
                    f"Trips that overnight in {city} would be great.",
                ])
                return MonthlyInstruction(kind, text, (Effect(f"layover_{code}", +1),), share)
            if kind == "layover_avoid":
                text = rng.choice([
                    f"Please avoid overnights in {city}.",
                    f"I'd rather not lay over in {city} this month.",
                ])
                return MonthlyInstruction(kind, text, (Effect(f"layover_{code}", -1),), share)
            text = rng.choice([
                f"I'd like at least one layover in {city} this month — one is enough.",
                f"I want to see a friend in {city}, so one overnight there this month would do.",
            ])
            return MonthlyInstruction(kind, text, (Effect(f"sched_layover_{code}", +1),),
                                      0.0, level="schedule")

        if kind == "no_early_reports":
            if not self._early_varies:
                return None
            text = rng.choice([
                "No early report times this month, please.",
                "I'm struggling with early wake-ups; avoid very early reports.",
            ])
            return MonthlyInstruction(kind, text, (Effect("early_report", -1),), share)

        if kind == "sched_free_weekends":
            text = rng.choice([
                "I want two free weekends this month, whichever they are.",
                "Give me at least two full weekends off this month.",
            ])
            return MonthlyInstruction(kind, text, (Effect("sched_free_weekends_2", +1),),
                                      0.0, level="schedule")

        raise ValueError(f"unknown instruction kind {kind!r}")


def compliance(
    instructions: Sequence[MonthlyInstruction],
    order: Sequence[int],
    library: FeatureLibrary,
    phi: FeatureMatrix,
    eligible: Sequence[int],
    k: int = DEFAULT_TOP_K,
) -> List[Optional[bool]]:
    """check_ranking for each instruction, in order."""
    firm_hits = {
        id(i): frozenset(pid for pid in phi.pairing_ids if i._hits(phi, pid))
        for i in instructions if i.firm and i.level == "pairing"
    }
    out = []
    for i in instructions:
        others = frozenset().union(*(h for key, h in firm_hits.items() if key != id(i)))
        out.append(i.check_ranking(order, library, phi, eligible, k, others))
    return out
