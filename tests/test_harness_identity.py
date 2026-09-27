"""
Identity verification for the harness refactor (SPEC section 2).

The recorded summary statistics the spec asks to re-verify against are not in
this repo, and the model runs at temperature 1.0, so re-running the three
methods would not reproduce any fixed number anyway. Instead this pins down
the stronger property: given identical LLM output, the refactored methods send
byte-identical prompts and produce identical rankings to the pre-existing code
paths.

Run directly (no pytest needed):
    python tests/test_harness_identity.py
Or under pytest if it is installed.
"""

from __future__ import annotations

import json
import os
import random
import sys
import zlib
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

# The pre-existing progress prints contain non-ASCII characters (≤, ρ, ⚠).
# On Windows the console defaults to cp1252 and those raise UnicodeEncodeError
# from inside the code under test, which has nothing to do with the behaviour
# being verified. Force UTF-8 on this process's streams.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass

import metrics as M
from harness import Instance, Ranking
from llm_client import StubClient
from strategies.pairwise import PairwiseBT
from strategies.rank_all import RankAll, ordered_ids_from_line_ranks
from strategies.scoring import IndependentScoring, ranking_from_scored_lines

# One instance set, built once, shared by every test.
INSTANCE = Instance.build(n_pilots=5, n_lines=20, pairings_per_line=5)
PILOTS = list(INSTANCE.pilots)
LINES = list(INSTANCE.lines)
LINE_IDS = [ln.id for ln in LINES]


# ---------------------------------------------------------------------------
# Deterministic stubs
#
# Every stub answer is a pure function of the prompt text, so the adapter and
# the pre-existing code path receive exactly the same LLM output. Python's
# hash() is salted per process, so crc32 is used instead.
# ---------------------------------------------------------------------------

def _h(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def rank_all_response(line_ids: Sequence[int]) -> str:
    """A well-formed Rank-All reply: a permutation of 1..N in shuffled order."""
    ids = list(line_ids)
    random.Random(7).shuffle(ids)
    return json.dumps([
        {
            "lineId": lid,
            "rank": rank,
            "eligible": True,
            "shortReason": f"stub reason for {lid}",
            "pros": ["p"],
            "cons": ["c"],
        }
        for rank, lid in enumerate(ids, start=1)
    ])


def score_response(prompt: str) -> str:
    """0-100 score derived from the prompt, so repeat calls agree."""
    return json.dumps({
        "score": _h(prompt) % 101,
        "eligible": True,
        "shortReason": "stub",
        "pros": ["p"],
        "cons": ["c"],
    })


def winner_response(prompt: str) -> str:
    """Pick option 1 or 2 from the prompt hash."""
    return json.dumps({
        "winner": 1 if _h(prompt) % 2 == 0 else 2,
        "confidence": "high",
        "reason": "stub",
    })


# ---------------------------------------------------------------------------
# Rank-All
# ---------------------------------------------------------------------------

def test_rank_all_prompt_is_the_preexisting_prompt():
    from prompt_builder import oracle_line_prompt

    for pilot in PILOTS:
        client = StubClient([rank_all_response(LINE_IDS)])
        RankAll(client).rank(pilot, LINES, seed=1)
        assert len(client.prompts) == 1, "Rank-All must make exactly one call"
        assert client.prompts[0] == oracle_line_prompt(pilot, LINES), (
            f"Rank-All prompt changed for pilot {pilot.id}"
        )


def test_rank_all_ordering_matches_stated_ranks():
    """For a well-formed reply the harness ordering equals the model's ranks."""
    from main import parse_line_oracle_response

    raw = rank_all_response(LINE_IDS)
    parsed = parse_line_oracle_response(raw, len(LINES))
    assert parsed is not None

    preexisting = tuple(r.line_id for r in sorted(parsed, key=lambda r: r.rank))
    client = StubClient([raw])
    got = RankAll(client).rank(PILOTS[0], LINES, seed=1)

    assert got.ordered_ids == preexisting, (
        f"\n  harness:     {got.ordered_ids}\n  pre-existing: {preexisting}"
    )
    assert set(got.ordered_ids) == set(LINE_IDS)


def test_rank_all_malformed_reply_still_yields_a_permutation():
    """
    The documented divergence: on a reply with duplicate ranks and a missing
    line the pre-existing code fed the stated ranks straight into Spearman,
    while the harness falls back to a positional ordering. Asserted here so
    the difference is recorded rather than discovered later.
    """
    from models import LLMLineRank

    parsed = [
        LLMLineRank(line_id=3, rank=1, eligible=True, short_reason="", pros=[], cons=[]),
        LLMLineRank(line_id=7, rank=1, eligible=True, short_reason="", pros=[], cons=[]),
        LLMLineRank(line_id=5, rank=2, eligible=True, short_reason="", pros=[], cons=[]),
    ]
    ordered = ordered_ids_from_line_ranks(parsed, LINE_IDS)

    assert ordered[:3] == (3, 7, 5), "duplicate ranks break the id tiebreak"
    assert set(ordered) == set(LINE_IDS), "omitted lines must still appear"
    assert len(ordered) == len(LINE_IDS), "ordering must not contain duplicates"


# ---------------------------------------------------------------------------
# Independent Scoring
# ---------------------------------------------------------------------------

def test_scoring_prompts_are_the_preexisting_prompts():
    from prompt_builder import generate_line_anchor, independent_scoring_line_prompt

    pilot = PILOTS[0]
    client = StubClient([score_response])
    IndependentScoring(client, n_runs=1).rank(pilot, LINES, seed=1)

    expected = {
        independent_scoring_line_prompt(
            pilot, ln, anchor=generate_line_anchor(pilot)
        )
        for ln in LINES
    }
    assert set(client.prompts) == expected, "Independent Scoring prompts changed"
    assert len(client.prompts) == len(LINES)


def test_scoring_ordering_matches_run_scored_lines():
    """
    The adapter calls main.run_scored_lines, so this checks the hook wiring:
    driving the pre-existing aggregation directly with the same numbers must
    give the same ordering as driving it through the adapter.
    """
    import asyncio

    from main import parse_line_score_response, run_scored_lines
    from prompt_builder import generate_line_anchor, independent_scoring_line_prompt

    pilot = PILOTS[1]

    async def direct_score_fn(pilot_, line, call_idx, provider, model, sem):
        prompt = independent_scoring_line_prompt(
            pilot_, line, anchor=generate_line_anchor(pilot_)
        )
        async with sem:
            return parse_line_score_response(score_response(prompt), call_idx)

    direct = asyncio.run(run_scored_lines(
        [pilot], LINES, provider="", model="", n_runs=2,
        score_fn=direct_score_fn,
    ))
    expected = tuple(r.line.id for r in direct[pilot.id])

    got = IndependentScoring(StubClient([score_response]), n_runs=2).rank(
        pilot, LINES, seed=1
    )
    assert got.ordered_ids == expected, (
        f"\n  harness:     {got.ordered_ids}\n  pre-existing: {expected}"
    )
    assert got.scores is not None and len(got.scores) == len(LINES)


def test_scoring_tiebreak_path_still_fires():
    """
    Regression guard on the aggregation the hooks were added to: with scores
    that vary between runs, the unstable-and-overlapping branch must still
    trigger a pairwise tiebreak and still return a full permutation.
    """
    import asyncio

    from main import run_scored_lines

    pilot = PILOTS[2]
    calls = {"pairwise": 0}

    async def jittery_score_fn(pilot_, line, call_idx, provider, model, sem):
        async with sem:
            # Wide spread per run -> "unstable", with near-identical means so
            # adjacent intervals overlap.
            base = 50 + (line.id % 3)
            return {
                "score": base + (20 if call_idx % 2 else -20),
                "eligible": True,
                "short_reason": "stub",
                "pros": [],
                "cons": [],
            }

    async def counting_pairwise_fn(pilot_, line_a, line_b, provider, model, sem):
        async with sem:
            calls["pairwise"] += 1
            return line_a.id

    out = asyncio.run(run_scored_lines(
        [pilot], LINES, provider="", model="", n_runs=2,
        score_fn=jittery_score_fn, pairwise_fn=counting_pairwise_fn,
    ))
    ids = [r.line.id for r in out[pilot.id]]
    assert sorted(ids) == sorted(LINE_IDS), "aggregation lost or duplicated a line"
    assert calls["pairwise"] > 0, "unstable overlapping pairs no longer tiebreak"
    assert any(r.stability == "unstable" for r in out[pilot.id])


# ---------------------------------------------------------------------------
# Pairwise / Bradley-Terry
# ---------------------------------------------------------------------------

def test_pairwise_ordering_matches_cli_implementation():
    """
    Drive strategies.PairwiseBT and main.run_adaptive_pairwise_mode with the
    same seeded RNG and the same stubbed answers, and require an identical
    final ordering.
    """
    from main import run_adaptive_pairwise_mode
    from oracle import oracle_rank_lines

    pilot = PILOTS[0]
    seed = 4242

    got = PairwiseBT(StubClient([winner_response])).rank(pilot, LINES, seed=seed)

    cli = run_adaptive_pairwise_mode(
        [pilot],
        LINES,
        {pilot.name: oracle_rank_lines(pilot, LINES)},
        is_line=True,
        respond=lambda prompt, label: winner_response(prompt),
        rng=random.Random(seed),
    )
    expected = tuple(iid for iid, _ in cli[pilot.id].final_ranking)

    assert got.ordered_ids == expected, (
        f"\n  harness:     {got.ordered_ids}\n  pre-existing: {expected}"
    )
    assert got.artifacts["n_comparisons"] == cli[pilot.id].n_comparisons
    assert set(got.ordered_ids) == set(LINE_IDS)


def test_pairwise_prompts_are_the_preexisting_prompts():
    from prompt_builder import adaptive_pairwise_prompt

    pilot = PILOTS[0]
    client = StubClient([winner_response])
    PairwiseBT(client).rank(pilot, LINES, seed=99)

    line_by_id = {ln.id: ln for ln in LINES}
    # Every prompt must be reproducible from the pair it compares.
    for prompt in client.prompts:
        assert prompt.startswith(
            "You are evaluating which of two monthly flying lines is better"
        ), "pairwise prompt preamble changed"
        assert "== OPTION 1 — MONTHLY LINE ==" in prompt
        assert "== OPTION 2 — MONTHLY LINE ==" in prompt

    # Spot-check one pair renders byte-identically.
    a, b = line_by_id[LINE_IDS[0]], line_by_id[LINE_IDS[1]]
    assert adaptive_pairwise_prompt(pilot, a, b, round_num=1) != ""


# ---------------------------------------------------------------------------
# Metrics and oracle
# ---------------------------------------------------------------------------

def test_metrics_spearman_matches_evaluator():
    """metrics.spearman must equal evaluator._line_spearman, not approximate it."""
    from evaluator import _line_spearman
    from models import LLMLineRank

    oracle = INSTANCE.oracle_rankings()
    rng = random.Random(11)

    for pilot in PILOTS:
        ora_order = oracle[pilot.id]
        ora_ranked = _oracle_ranked(pilot)
        for _ in range(5):
            shuffled = list(LINE_IDS)
            rng.shuffle(shuffled)
            llm = [
                LLMLineRank(line_id=lid, rank=i, eligible=True,
                            short_reason="", pros=[], cons=[])
                for i, lid in enumerate(shuffled, start=1)
            ]
            assert M.spearman(shuffled, ora_order) == _line_spearman(llm, ora_ranked), (
                "metrics.spearman diverged from evaluator._line_spearman"
            )


def test_metrics_edge_cases():
    assert M.spearman([1, 2, 3], [1, 2, 3]) == 1.0
    assert M.spearman([1, 2, 3], [3, 2, 1]) == -1.0
    assert M.kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1.0
    assert M.kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    assert M.top3_accuracy([5, 1, 2, 3], [1, 9, 9]) == 1.0
    assert M.top3_accuracy([5, 6, 7, 1], [1, 9, 9]) == 0.0
    assert M.top3_set_overlap([1, 2, 3], [1, 2, 3]) == 1.0
    assert M.top3_set_overlap([4, 5, 6], [1, 2, 3]) == 0.0
    assert M.tie_rate(None) is None, "no scores must report n/a, not zero"
    assert M.tie_rate({1: 1.0, 2: 1.0, 3: 2.0}) == round(1 / 3, 4)
    assert M.tie_rate({1: 1.0, 2: 2.0, 3: 3.0}) == 0.0
    assert M.run_consistency([[1, 2, 3]]) is None, "one rep has no consistency"
    assert M.run_consistency([[1, 2, 3], [1, 2, 3]]) == 1.0


def test_oracle_is_unchanged():
    """
    Pins the oracle's output for this instance set. SPEC section 13 forbids
    modifying the oracle; this fails loudly if anything does.
    """
    oracle = INSTANCE.oracle_rankings()
    assert oracle[PILOTS[0].id] == (
        8, 1, 10, 17, 2, 20, 13, 4, 5, 6, 7, 3, 14, 9, 16, 15, 18, 11, 12, 19
    ), "oracle ranking changed for Dana Okafor"
    family = (20, 10, 3, 8, 16, 2, 14, 17, 1, 7, 11, 18, 4, 5, 6, 12, 13, 9, 15, 19)
    for pilot in PILOTS[1:]:
        assert oracle[pilot.id] == family, f"oracle changed for {pilot.name}"


def test_instance_matches_cli_construction():
    """The harness must rebuild exactly the lines the CLI's line mode builds."""
    from generator import ScenarioGenerator

    gen = ScenarioGenerator(1234567, 42)
    pilots = gen.build_pilots(5, "BOS")
    pairings = gen.build_pairings(100, pilots, "BOS", max_b767=round(100 * 0.32))
    lines = gen.build_lines(pairings, 20, 5)

    assert [p.name for p in pilots] == [p.name for p in PILOTS]
    assert [ln.id for ln in lines] == LINE_IDS
    for a, b in zip(lines, LINES):
        assert [p.id for p in a.pairings] == [p.id for p in b.pairings]


def _oracle_ranked(pilot):
    from oracle import oracle_rank_lines

    return oracle_rank_lines(pilot, LINES)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    print("Harness identity verification\n")
    sys.exit(_run_all())
