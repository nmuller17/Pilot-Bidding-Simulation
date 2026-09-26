"""
pbs_instance.py
---------------
The instance for the PBS pipeline of the research note: pilots, dated pairings
(the candidates), the feature library K, Phi, monthly instructions, and the
bid-form oracle that every LLM bid is evaluated against.

It plugs into harness.ExperimentRunner like harness.Instance does, and adds
the bid-level metrics of section 5 through `extra_metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import metrics as M
from bid import DEFAULT_BUDGET, Bid, PairingRanking, oracle_bid, rank_pairings
from features import FeatureLibrary, FeatureMatrix
from instructions import DEFAULT_TOP_K, InstructionGenerator, MonthlyInstruction, compliance
from models import Pairing, Pilot

# R4 feedback thresholds on the oracle's rank among the pilot's eligible trips.
GREAT_TOP_SHARE = 0.2
FINE_TOP_SHARE = 0.5


@dataclass
class PBSInstance:
    pilots: Tuple[Pilot, ...]
    pairings: Tuple[Pairing, ...]
    library: FeatureLibrary
    phi: FeatureMatrix
    instructions: Dict[int, Tuple[MonthlyInstruction, ...]]
    budget: float
    pilot_seed: int
    pairing_seed: int
    instruction_seed: int
    base: str
    top_k: int = DEFAULT_TOP_K
    _oracle: Dict[int, Tuple[Bid, PairingRanking]] = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls,
        n_pilots: int = 5,
        n_pairings: int = 40,
        bid_month: date = date(2026, 10, 1),
        base: str = "BOS",
        pilot_seed: int = 1234567,
        pairing_seed: int = 42,
        instruction_seed: int = 7,
        budget: float = DEFAULT_BUDGET,
        b767_share: float = 0.32,
        top_k: int = DEFAULT_TOP_K,
    ) -> "PBSInstance":
        """
        Pilots and pairings from the same generator and seeds as the bid-line
        experiments; the pairings keep the random start dates build_pairings
        gives them (no line layout).
        """
        from generator import ScenarioGenerator

        gen = ScenarioGenerator(pilot_seed, pairing_seed, bid_month)
        pilots = gen.build_pilots(n_pilots, base)
        pairings = gen.build_pairings(
            n_pairings, pilots, base, max_b767=round(n_pairings * b767_share)
        )
        library = FeatureLibrary.for_pairings(pairings, gen.month_start)
        phi = FeatureMatrix.build(library, pairings)
        instructions = InstructionGenerator(library, phi, instruction_seed).for_pilots(pilots)
        return cls(
            pilots=tuple(pilots),
            pairings=tuple(pairings),
            library=library,
            phi=phi,
            instructions=instructions,
            budget=budget,
            pilot_seed=pilot_seed,
            pairing_seed=pairing_seed,
            instruction_seed=instruction_seed,
            base=base,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Candidates and oracle
    # ------------------------------------------------------------------

    def candidates(self) -> List[Pairing]:
        return list(self.pairings)

    def pilot_by_id(self, pilot_id: int) -> Pilot:
        return next(p for p in self.pilots if p.id == pilot_id)

    def eligible_ids(self, pilot: Pilot) -> List[int]:
        return [p.id for p in self.pairings if p.is_qualified(pilot)]

    def _oracle_for(self, pilot: Pilot) -> Tuple[Bid, PairingRanking]:
        if pilot.id not in self._oracle:
            b = oracle_bid(pilot, self.instructions[pilot.id], self.budget)
            self._oracle[pilot.id] = (b, rank_pairings(b, self.phi, self.pairings, pilot))
        return self._oracle[pilot.id]

    def oracle_bid(self, pilot: Pilot) -> Bid:
        return self._oracle_for(pilot)[0]

    def oracle_ranking(self, pilot: Pilot) -> PairingRanking:
        return self._oracle_for(pilot)[1]

    def oracle_rankings(self) -> Dict[int, Tuple[int, ...]]:
        return {p.id: self.oracle_ranking(p).ordered_ids for p in self.pilots}

    def oracle_scores(self) -> Dict[int, Dict[int, float]]:
        return {p.id: dict(self.oracle_ranking(p).scores) for p in self.pilots}

    # ------------------------------------------------------------------
    # R4 feedback
    # ------------------------------------------------------------------

    def oracle_feedback(self, pilot: Pilot, shortlist: Sequence[int]) -> List[Tuple[int, str, str]]:
        """
        The oracle playing the pilot in R4: (pairing id, verdict, comment) for
        each shortlisted trip. The verdict comes from the oracle's rank among
        the pilot's eligible trips; the comment names any monthly instruction
        the trip breaks. Only verdicts and instruction texts are revealed, never
        the oracle's weights.
        """
        r = self.oracle_ranking(pilot)
        elig = [pid for pid in r.ordered_ids if pid not in r.ineligible]
        pos = {pid: i for i, pid in enumerate(elig)}
        n = len(elig)
        out = []
        for pid in shortlist:
            share = pos[pid] / n if pid in pos else 1.0
            if share < GREAT_TOP_SHARE:
                verdict, text = "great", "one of my favourite trips this month"
            elif share < FINE_TOP_SHARE:
                verdict, text = "fine", "acceptable, but I have better options"
            else:
                verdict, text = "poor", "I would not bid this trip this high"
            broken = [
                i.text for i in self.instructions[pilot.id]
                if i.level == "pairing" and i.effects[0].sigma < 0
                and self.library[i.effects[0].key].type != "continuous"
                and i._hits(self.phi, pid)
            ]
            if broken:
                verdict = "poor"
                text += "; it conflicts with my request: " + " / ".join(f'"{b}"' for b in broken)
            out.append((pid, verdict, text))
        return out

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def extra_metrics(self, ranking) -> Dict[str, Any]:
        """
        Section 5 metrics beyond the rank correlations the runner already
        computes: top-k overlap, instruction compliance of r_p, and, when the
        method exposes its bid, selection P/R/F1, weight error and direction
        accuracy against the oracle bid.
        """
        pilot = self.pilot_by_id(ranking.pilot_id)
        oracle_order = self.oracle_ranking(pilot).ordered_ids
        order = ranking.ordered_ids
        instrs = self.instructions[pilot.id]
        flags = compliance(instrs, order, self.library, self.phi,
                           self.eligible_ids(pilot), self.top_k)
        out: Dict[str, Any] = {
            "top5_overlap": M.top_k_overlap(order, oracle_order, 5),
            "top10_overlap": M.top_k_overlap(order, oracle_order, 10),
            "compliance_rate": M.rate(flags),
            "compliance_firm": M.rate(f for f, i in zip(flags, instrs) if i.firm),
            "compliance_soft": M.rate(f for f, i in zip(flags, instrs) if not i.firm),
        }
        bid = ranking.artifacts.get("bid")
        if bid is not None:
            truth = self.oracle_bid(pilot)
            selected = (set(bid["weights"]) | set(bid["hard_exclusions"])
                        | set(bid["schedule_prefs"]))
            prf = M.selection_prf(selected, truth.selected)
            out.update({
                "selection_precision": prf["precision"],
                "selection_recall": prf["recall"],
                "selection_f1": prf["f1"],
                "weight_error": M.weight_error(bid["weights"], dict(truth.weights), self.budget),
                "direction_accuracy": M.direction_accuracy(bid["directions"], dict(truth.directions)),
                "exclusion_f1": M.selection_prf(bid["hard_exclusions"], truth.hard_exclusions)["f1"],
            })
        return out

    def schedule_outcome(self, pilot: Pilot, schedule: Sequence[Pairing]) -> Dict[str, Any]:
        """
        Outcome metrics for a PBS schedule x_p (for when the solver exists):
        oracle satisfaction S*_p(x_p) and the share of instructions it meets.
        """
        scores = self.oracle_ranking(pilot).scores
        elig = self.eligible_ids(pilot)
        flags = [i.check_schedule(schedule, self.library, self.phi, elig)
                 for i in self.instructions[pilot.id]]
        return {
            "oracle_satisfaction": M.oracle_satisfaction([p.id for p in schedule], scores),
            "n_trips": len(schedule),
            "schedule_compliance": M.rate(flags),
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": "pbs",
            "n_pilots": len(self.pilots),
            "n_pairings": len(self.pairings),
            "bid_month": self.library.month_start.isoformat(),
            "base": self.base,
            "budget": self.budget,
            "top_k": self.top_k,
            "n_columns": len(self.library.columns),
            "n_pairing_columns": len(self.phi.keys),
            "constant_columns": [k for k in self.phi.keys if self.phi.is_constant(k)],
            "pilot_seed": self.pilot_seed,
            "pairing_seed": self.pairing_seed,
            "instruction_seed": self.instruction_seed,
        }

    def pilot_metadata(self, pilot: Pilot) -> Dict[str, Any]:
        return {
            "monthly_instructions": [i.as_dict() for i in self.instructions[pilot.id]],
            "oracle_bid": self.oracle_bid(pilot).as_dict(),
        }
