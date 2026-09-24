"""
harness.py
----------
One interface and one evaluation harness for every preference-elicitation
method in this repo.

All four methods (Rank-All, Independent Scoring, Pairwise/Bradley-Terry, and
the indicator-based Method D) implement `RankingMethod` and are run by
`ExperimentRunner` over the same instance set, so the comparison between them
is apples-to-apples.

The three pre-existing methods are wrapped, not rewritten: the adapters in
`strategies/` call the same prompt builders, parsers and fitting code they
always did.  `tests/test_harness_identity.py` pins that down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from models import Line, Pilot

# Bumped whenever a prompt template changes, so cached LLM responses are never
# silently reused across a definition change (see llm_cache.PROMPT_VERSION use).
HARNESS_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Method output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ranking:
    """
    One method's ranking of one candidate set for one pilot.

    ordered_ids is the canonical output: candidate ids best -> worst, always a
    permutation of the candidate ids handed to `rank()`.

    scores carries a per-candidate numeric score when the method produces one
    (Independent Scoring's mean score, Bradley-Terry strength, Method D's
    G_iS).  Rank-All produces no score, so it is None there and the tie-rate
    metric is reported as n/a rather than as zero.

    artifacts is the method-specific payload: for Method D the selected
    indicators, signed weights, rationales and the per-candidate score
    breakdown; for Pairwise the comparisons and fit quality; and so on.
    """
    method: str
    pilot_id: int
    ordered_ids: Tuple[int, ...]
    scores: Optional[Dict[int, float]] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # Method-axis labels. Present on every Ranking from the start so the
    # results schema carries them even for methods that do not vary them
    # (see SPEC sections 4, 8 and 9).
    variant: Optional[str] = None
    selection_context: Optional[str] = None
    weight_source: Optional[str] = None

    def rank_of(self) -> Dict[int, int]:
        """candidate id -> rank position, 1 = best."""
        return {cid: i for i, cid in enumerate(self.ordered_ids, start=1)}


class RankingMethod(Protocol):
    """Common protocol for every method. Implementations live in strategies/."""

    name: str

    def rank(
        self,
        pilot: Pilot,
        candidates: Sequence[Line],
        *,
        seed: int,
    ) -> Ranking:
        """Rank `candidates` for `pilot`. Must not mutate either argument."""
        ...


# ---------------------------------------------------------------------------
# Instance set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instance:
    """The pilots and lines every method is evaluated on."""
    pilots: Tuple[Pilot, ...]
    lines: Tuple[Line, ...]
    pilot_seed: int
    pairing_seed: int
    base: str
    pairings_per_line: int

    @classmethod
    def build(
        cls,
        n_pilots: int = 5,
        n_lines: int = 20,
        pairings_per_line: int = 5,
        base: str = "BOS",
        pilot_seed: int = 1234567,
        pairing_seed: int = 42,
    ) -> "Instance":
        """
        Rebuild the instance set the existing experiments use.

        Mirrors main.main()'s line-mode construction exactly, including the
        ~32% B767 cap, so the lines are identical to those the three existing
        methods were run on.
        """
        from generator import ScenarioGenerator

        gen = ScenarioGenerator(pilot_seed, pairing_seed)
        pilots = gen.build_pilots(n_pilots, base)
        n_pair = n_lines * pairings_per_line
        max_b767 = round(n_pair * 0.32)
        pairings = gen.build_pairings(n_pair, pilots, base, max_b767=max_b767)
        lines = gen.build_lines(pairings, n_lines, pairings_per_line)
        return cls(
            pilots=tuple(pilots),
            lines=tuple(lines),
            pilot_seed=pilot_seed,
            pairing_seed=pairing_seed,
            base=base,
            pairings_per_line=pairings_per_line,
        )

    def oracle_rankings(self) -> Dict[int, Tuple[int, ...]]:
        """pilot.id -> oracle line ids best -> worst. The oracle is unmodified."""
        from oracle import oracle_rank_lines

        return {
            p.id: tuple(r.line.id for r in oracle_rank_lines(p, list(self.lines)))
            for p in self.pilots
        }

    def oracle_scores(self) -> Dict[int, Dict[int, float]]:
        """pilot.id -> {line id: oracle score}. Used for the oracle's own tie rate."""
        from oracle import oracle_rank_lines

        return {
            p.id: {r.line.id: float(r.oracle_score)
                   for r in oracle_rank_lines(p, list(self.lines))}
            for p in self.pilots
        }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

@dataclass
class LLMSettings:
    """
    Shared LLM configuration. Identical for every method — fairness of the
    comparison depends on it (SPEC section 9).
    """
    provider: str = "anthropic"
    model: str = "claude-sonnet-5"
    temperature: float = 1.0
    max_tokens: int = 4096

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class ExperimentRunner:
    """
    Runs a list of methods over one instance set for a number of repetitions
    and writes every result in one consistent format.
    """

    def __init__(
        self,
        instance: Instance,
        llm: Optional[LLMSettings] = None,
        base_seed: int = 20260921,
    ):
        self.instance = instance
        self.llm = llm or LLMSettings()
        self.base_seed = base_seed

    # ------------------------------------------------------------------
    def run(
        self,
        methods: Sequence[RankingMethod],
        n_reps: int = 5,
        on_error: str = "raise",
    ) -> Dict[str, Any]:
        """
        Rank every candidate set with every method, n_reps times per pilot.

        The seed handed to each rank() call is derived from base_seed, the
        pilot id and the repetition index, so a rerun reproduces the same
        seeds and any method with a stochastic component is reproducible.
        """
        import metrics as M

        oracle = self.instance.oracle_rankings()
        rankings: Dict[Tuple[str, int], List[Ranking]] = {}
        rows: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []

        for method in methods:
            for pilot in self.instance.pilots:
                for rep in range(n_reps):
                    seed = self._seed_for(pilot.id, rep)
                    try:
                        r = method.rank(
                            pilot, list(self.instance.lines), seed=seed
                        )
                    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                        if on_error == "raise":
                            raise
                        failures.append({
                            "method": method.name,
                            "pilot_id": pilot.id,
                            "rep": rep,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        continue

                    rankings.setdefault((method.name, pilot.id), []).append(r)
                    rows.append(self._row(r, rep, seed, oracle[pilot.id]))

        return {
            "metadata": self._metadata(n_reps, methods),
            "runs": rows,
            "per_pilot": self._per_pilot(rankings, oracle),
            "aggregate": self._aggregate(rankings, oracle),
            "failures": failures,
        }

    # ------------------------------------------------------------------
    def _seed_for(self, pilot_id: int, rep: int) -> int:
        return self.base_seed + pilot_id * 1000 + rep

    def _row(
        self,
        r: Ranking,
        rep: int,
        seed: int,
        oracle_order: Tuple[int, ...],
    ) -> Dict[str, Any]:
        import metrics as M

        return {
            "method": r.method,
            "variant": r.variant,
            "selection_context": r.selection_context,
            "weight_source": r.weight_source,
            "pilot_id": r.pilot_id,
            "rep": rep,
            "seed": seed,
            "ordered_ids": list(r.ordered_ids),
            "scores": ({str(k): v for k, v in r.scores.items()}
                       if r.scores is not None else None),
            "metrics": {
                "spearman": M.spearman(r.ordered_ids, oracle_order),
                "kendall_tau_b": M.kendall_tau_b(r.ordered_ids, oracle_order),
                "top3_accuracy": M.top3_accuracy(r.ordered_ids, oracle_order),
                "top3_set_overlap": M.top3_set_overlap(r.ordered_ids, oracle_order),
                "tie_rate": M.tie_rate(r.scores),
            },
            "artifacts": r.artifacts,
        }

    def _per_pilot(
        self,
        rankings: Dict[Tuple[str, int], List[Ranking]],
        oracle: Dict[int, Tuple[int, ...]],
    ) -> List[Dict[str, Any]]:
        import metrics as M

        out: List[Dict[str, Any]] = []
        for (method_name, pilot_id), reps in sorted(rankings.items()):
            orders = [r.ordered_ids for r in reps]
            ora = oracle[pilot_id]
            ties = [M.tie_rate(r.scores) for r in reps]
            ties = [t for t in ties if t is not None]
            out.append({
                "method": method_name,
                "variant": reps[0].variant,
                "pilot_id": pilot_id,
                "n_reps": len(reps),
                "spearman": M.mean([M.spearman(o, ora) for o in orders]),
                "kendall_tau_b": M.mean([M.kendall_tau_b(o, ora) for o in orders]),
                "consistency": M.run_consistency(orders),
                "top3_accuracy": M.mean([M.top3_accuracy(o, ora) for o in orders]),
                "top3_set_overlap": M.mean(
                    [M.top3_set_overlap(o, ora) for o in orders]
                ),
                "tie_rate": M.mean(ties) if ties else None,
                "n_selected": M.mean([
                    r.artifacts["n_selected"] for r in reps
                    if "n_selected" in r.artifacts
                ]) or None,
            })
        return out

    def _aggregate(
        self,
        rankings: Dict[Tuple[str, int], List[Ranking]],
        oracle: Dict[int, Tuple[int, ...]],
    ) -> List[Dict[str, Any]]:
        import metrics as M

        by_method: Dict[str, List[Dict[str, Any]]] = {}
        for row in self._per_pilot(rankings, oracle):
            by_method.setdefault(row["method"], []).append(row)

        out: List[Dict[str, Any]] = []
        for method_name, per_pilot in sorted(by_method.items()):
            def col(key: str) -> Optional[float]:
                vals = [r[key] for r in per_pilot if r[key] is not None]
                return M.mean(vals) if vals else None

            out.append({
                "method": method_name,
                "variant": per_pilot[0]["variant"],
                "n_pilots": len(per_pilot),
                "spearman": col("spearman"),
                "kendall_tau_b": col("kendall_tau_b"),
                "consistency": col("consistency"),
                "top3_accuracy": col("top3_accuracy"),
                "top3_set_overlap": col("top3_set_overlap"),
                "tie_rate": col("tie_rate"),
            })
        return out

    def _metadata(
        self, n_reps: int, methods: Sequence[RankingMethod]
    ) -> Dict[str, Any]:
        import metrics as M

        oracle_scores = self.instance.oracle_scores()
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "harness_version": HARNESS_VERSION,
            "n_reps": n_reps,
            "methods": [m.name for m in methods],
            "llm": self.llm.as_dict(),
            "base_seed": self.base_seed,
            "instance": {
                "n_pilots": len(self.instance.pilots),
                "n_lines": len(self.instance.lines),
                "pairings_per_line": self.instance.pairings_per_line,
                "base": self.instance.base,
                "pilot_seed": self.instance.pilot_seed,
                "pairing_seed": self.instance.pairing_seed,
            },
            "pilots": [
                {
                    "id": p.id,
                    "name": p.name,
                    "age": p.age,
                    "family_status": p.family_status,
                    "seniority": p.seniority,
                    "qualified_types": list(p.qualified_types),
                    "base_pay": p.base_pay,
                    "monthly_instructions": getattr(p, "monthly_instructions", ""),
                    "oracle_weights": {
                        "tafb": p.weights.tafb,
                        "hotel_nights": p.weights.hotel_nights,
                        "report_time": p.weights.report_time,
                        "credit_pay": p.weights.credit_pay,
                    },
                }
                for p in self.instance.pilots
            ],
            "oracle_tie_rate": {
                str(pid): M.tie_rate(scores)
                for pid, scores in oracle_scores.items()
            },
        }

    # ------------------------------------------------------------------
    @staticmethod
    def write(results: Dict[str, Any], path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
