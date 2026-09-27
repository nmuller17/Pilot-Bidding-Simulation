"""
Rank-All: one prompt per pilot, the LLM returns a ranked JSON array.

Adapter only. The prompt comes from `prompt_builder.oracle_line_prompt` and
the response goes through `main.parse_line_oracle_response`, exactly as the
CLI's line mode does.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from harness import Ranking
from models import Line, Pilot


class RankAll:
    """Method A — compare-all-at-once."""

    name = "rank_all"

    def __init__(self, client):
        self.client = client

    # ------------------------------------------------------------------
    def rank(
        self, pilot: Pilot, candidates: Sequence[Line], *, seed: int
    ) -> Ranking:
        from main import parse_line_oracle_response
        from prompt_builder import oracle_line_prompt

        lines = list(candidates)
        prompt = oracle_line_prompt(pilot, lines)
        raw = self.client.complete(prompt)
        parsed = parse_line_oracle_response(raw, len(lines))
        if parsed is None:
            raise ValueError(
                f"Rank-All response for pilot {pilot.id} did not parse"
            )

        ordered = ordered_ids_from_line_ranks(parsed, [ln.id for ln in lines])
        return Ranking(
            method=self.name,
            pilot_id=pilot.id,
            ordered_ids=ordered,
            scores=None,   # emits a rank directly, so it has no tie rate
            artifacts={
                "n_returned": len(parsed),
                "raw_response": raw,
                "reasons": {
                    str(r.line_id): r.short_reason for r in parsed
                },
                "eligible": {str(r.line_id): r.eligible for r in parsed},
            },
        )


def ordered_ids_from_line_ranks(
    parsed, candidate_ids: Sequence[int]
) -> Tuple[int, ...]:
    """
    Turn a list of LLMLineRank into an ordering of candidate ids, best first.

    Sorted by the rank the model stated, with the line id as a deterministic
    tiebreak for duplicate ranks. Any candidate the model omitted is appended
    in id order, so the result is always a permutation of `candidate_ids` and
    the metrics never silently score a short list.

    This is the one place where the harness is stricter than the pre-existing
    code, which fed the model's stated ranks straight into the Spearman
    calculation. For a well-formed response — a permutation of 1..N, which is
    what the prompt asks for — the two agree exactly; they can only diverge on
    a malformed response, where a positional ranking is the better-defined
    choice. The identity test covers both cases.
    """
    known = set(candidate_ids)
    seen: Dict[int, int] = {}
    for r in parsed:
        if r.line_id in known and r.line_id not in seen:
            seen[r.line_id] = r.rank

    ordered = [cid for cid, _ in sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))]
    ordered += [cid for cid in candidate_ids if cid not in seen]
    return tuple(ordered)
