"""
Pairwise / Bradley-Terry: ~N adaptive comparisons fitted into a global ranking.

This is the same two-round algorithm as `main.run_adaptive_pairwise_mode`,
built from the same primitives (`evaluator.design_adaptive_comparisons`,
`BradleyTerryModel`, `evaluator.run_adaptive_pairwise`) and reproduced step for
step so it can run non-interactively for one pilot without the CLI's printing,
stdin reading and oracle lookup.

`tests/test_harness_identity.py` drives this and
`main.run_adaptive_pairwise_mode` with the same stubbed comparison answers and
the same seeded RNG, and asserts the two produce an identical ordering.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

from harness import Ranking
from models import Line, PairwiseComparison, Pilot


class PairwiseBT:
    """Method B — adaptive pairwise with a Bradley-Terry fit."""

    name = "pairwise_bt"

    def __init__(self, client):
        self.client = client

    # ------------------------------------------------------------------
    def rank(
        self, pilot: Pilot, candidates: Sequence[Line], *, seed: int
    ) -> Ranking:
        from evaluator import (
            BradleyTerryModel,
            design_adaptive_comparisons,
            run_adaptive_pairwise,
        )
        from prompt_builder import adaptive_pairwise_prompt

        items = list(candidates)
        item_ids = [it.id for it in items]
        item_map = {it.id: it for it in items}
        rng = random.Random(seed)
        comparisons: List[PairwiseComparison] = []

        def ask(item_a, item_b, round_num: int, context: str = "") -> Optional[int]:
            prompt = adaptive_pairwise_prompt(
                pilot, item_a, item_b, round_num=round_num, context=context
            )
            raw = self.client.complete(prompt)
            return _parse_winner(raw, item_a.id, item_b.id, round_num, comparisons)

        # -- Round 1: ceil(N/2) comparisons on a seeded shuffle ------------
        r1_pairs = design_adaptive_comparisons(item_ids, rng=rng)
        for a_id, b_id in r1_pairs:
            ask(item_map[a_id], item_map[b_id], 1)

        # -- Provisional fit ---------------------------------------------
        prov = BradleyTerryModel(item_ids)
        for c in comparisons:
            prov.add_comparison(
                c.winner_id,
                c.item_b_id if c.winner_id == c.item_a_id else c.item_a_id,
            )
        prov.fit()
        prov_ranking = [iid for iid, _ in prov.ranking()]

        # -- Round 2: uncertain adjacent pairs ---------------------------
        # design_adaptive_comparisons reshuffles on this second call, so its
        # round-1 block differs from the first call's and a few of those pairs
        # survive the filter below. That is what the CLI does; reproduced here
        # rather than corrected, so the two paths stay comparable.
        all_pairs = design_adaptive_comparisons(
            item_ids, provisional_ranking=prov_ranking, rng=rng
        )
        r1_set = {(min(a, b), max(a, b)) for a, b in r1_pairs}
        r2_pairs = [
            (a, b) for a, b in all_pairs if (min(a, b), max(a, b)) not in r1_set
        ]

        for a_id, b_id in r2_pairs:
            a_rank = prov_ranking.index(a_id) + 1
            b_rank = prov_ranking.index(b_id) + 1
            context = (
                f"Provisional ranking suggests item {a_id} is around "
                f"#{a_rank} and item {b_id} is around #{b_rank}."
            )
            ask(item_map[a_id], item_map[b_id], 2, context)

        # -- Final fit ----------------------------------------------------
        result = run_adaptive_pairwise(pilot, items, comparisons)
        ordered = [iid for iid, _ in result.final_ranking]
        strengths = {iid: float(s) for iid, s in result.final_ranking}

        return Ranking(
            method=self.name,
            pilot_id=pilot.id,
            ordered_ids=tuple(ordered),
            scores=strengths,
            artifacts={
                "n_comparisons": result.n_comparisons,
                "n_comparisons_full": len(item_ids) * (len(item_ids) - 1) // 2,
                "n_rounds": result.n_rounds,
                "fit_quality": round(result.fit_quality, 4),
                "consistency_note": result.consistency_note,
                "comparisons": [
                    {
                        "item_a": c.item_a_id,
                        "item_b": c.item_b_id,
                        "winner": c.winner_id,
                        "confidence": c.confidence,
                        "reason": c.reason,
                        "round": c.round_num,
                    }
                    for c in result.comparisons
                ],
            },
        )


def _parse_winner(
    raw: str,
    a_id: int,
    b_id: int,
    round_num: int,
    sink: List[PairwiseComparison],
) -> Optional[int]:
    """
    Parse one comparison reply and append it to `sink`.

    Parsing and failure handling mirror main.run_adaptive_pairwise_mode: fence
    stripping, winner read as 1 or 2, and a malformed reply logged and skipped
    rather than raised, so one bad comparison does not lose the whole pilot.
    """
    try:
        d = json.loads(
            raw.replace("```json", "").replace("```", "").strip()
        )
        winner_num = int(d["winner"])
        winner_id = a_id if winner_num == 1 else b_id
        sink.append(PairwiseComparison(
            item_a_id=a_id,
            item_b_id=b_id,
            winner_id=winner_id,
            confidence=d.get("confidence", "?"),
            reason=d.get("reason", ""),
            round_num=round_num,
        ))
        return winner_id
    except Exception as exc:  # noqa: BLE001 - mirrors main's handling
        print(f"  ⚠ Parse error: {exc} — skipping")
        return None
