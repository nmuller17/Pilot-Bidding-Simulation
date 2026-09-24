"""
Independent Scoring: one LLM call per line, scored 0-100, sorted by score.

Adapter only. The aggregation — repeated runs, stability tiering, and the
pairwise tiebreak on overlapping unstable adjacent pairs — is
`main.run_scored_lines`, called here with its two LLM-calling steps pointed at
the shared client instead of at inline provider clients.
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Optional, Sequence

from harness import Ranking
from models import Line, Pilot

# The CLI's default for --consistency-runs in line mode.
DEFAULT_N_RUNS = 2


class IndependentScoring:
    """Method C — independent 0-100 scores."""

    name = "independent_scoring"

    def __init__(self, client, n_runs: int = DEFAULT_N_RUNS, batch_size: int = 20):
        self.client = client
        self.n_runs = n_runs
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    def rank(
        self, pilot: Pilot, candidates: Sequence[Line], *, seed: int
    ) -> Ranking:
        from main import parse_line_score_response, run_scored_lines

        lines = list(candidates)
        client = self.client

        async def score_fn(pilot_, line, call_idx, provider, model, sem):
            """Stands in for main._score_one_line, same signature and parsing."""
            from prompt_builder import (
                generate_line_anchor,
                independent_scoring_line_prompt,
            )

            prompt = independent_scoring_line_prompt(
                pilot_, line, anchor=generate_line_anchor(pilot_)
            )
            async with sem:
                try:
                    raw = await client.acomplete(prompt)
                    return parse_line_score_response(raw, call_idx)
                except Exception as exc:  # noqa: BLE001 - mirrors main's handling
                    print(
                        f"  ⚠ API error (call {call_idx}, L{line.id} "
                        f"for {pilot_.name}): {exc}"
                    )
                    return None

        async def pairwise_fn(pilot_, line_a, line_b, provider, model, sem):
            """Stands in for main._pairwise_one_line, same signature and parsing."""
            from prompt_builder import pairwise_line_prompt

            prompt = pairwise_line_prompt(pilot_, line_a, line_b)
            async with sem:
                try:
                    raw = await client.acomplete(prompt)
                    clean = (
                        raw.strip()
                        .replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )
                    winner = int(json.loads(clean)["winner"])
                    return line_a.id if winner == 1 else line_b.id
                except Exception as exc:  # noqa: BLE001 - mirrors main's handling
                    print(
                        f"  ⚠ Pairwise error (L{line_a.id} vs L{line_b.id} "
                        f"for {pilot_.name}): {exc}"
                    )
                    return None

        by_pilot = asyncio.run(
            run_scored_lines(
                [pilot],
                lines,
                provider="",      # unused: score_fn/pairwise_fn hold the client
                model="",
                n_runs=self.n_runs,
                batch_size=self.batch_size,
                score_fn=score_fn,
                pairwise_fn=pairwise_fn,
            )
        )

        results = by_pilot.get(pilot.id, [])
        return ranking_from_scored_lines(
            self.name, pilot, results, [ln.id for ln in lines]
        )


def ranking_from_scored_lines(
    method: str,
    pilot: Pilot,
    results,
    candidate_ids: Sequence[int],
) -> Ranking:
    """
    Build a Ranking from main.run_scored_lines' already-ordered output.

    run_scored_lines returns the list in final ranked order — ineligible last,
    then by mean score descending with the line id as tiebreak, then with any
    pairwise tiebreak swaps applied — so the order is taken as given rather
    than re-sorted here. Lines that produced no usable response are dropped by
    run_scored_lines and are appended in id order to keep the output a
    permutation of the candidate set.
    """
    ordered = [r.line.id for r in results]
    ordered += [cid for cid in candidate_ids if cid not in set(ordered)]
    scores = {r.line.id: float(r.score_mean) for r in results}

    return Ranking(
        method=method,
        pilot_id=pilot.id,
        ordered_ids=tuple(ordered),
        scores=scores or None,
        artifacts={
            "n_scored": len(results),
            "stability": {r.line.id: r.stability for r in results},
            "score_runs": {r.line.id: list(r.score_runs) for r in results},
            "score_std": {r.line.id: r.score_std for r in results},
            "ranking_method": {r.line.id: r.ranking_method for r in results},
            "reasons": {r.line.id: r.short_reason for r in results},
        },
    )
