"""
The column-bid method (Steps 4-6 of the research note) and its runner
integration, driven by scripted LLM replies — no API calls.

Run directly:  python tests/test_pbs_pipeline.py
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass

import metrics as M
from harness import ExperimentRunner, Instance
from llm_client import StubClient
from oracle_responder import OracleResponder
from pbs_instance import PBSInstance
from strategies.column_bid import MODES, REGIMES, ColumnBid

INST = PBSInstance.build(n_pilots=5, n_pairings=40)
PILOTS = list(INST.pilots)
CANDS = INST.candidates()
# The pilot with the richest instructions (a firm one and a schedule-level one).
RICH = max(PILOTS, key=lambda p: len(INST.instructions[p.id]))


def _perfect():
    return StubClient([OracleResponder(INST)])


def _stage(prompt: str) -> str:
    return re.search(r"^Stage: (\w+)", prompt, re.M).group(1)


class Scripted:
    """Answers with the oracle, except for stages overridden by `overrides`."""

    def __init__(self, overrides):
        self.oracle = OracleResponder(INST)
        self.overrides = overrides

    def __call__(self, prompt):
        st = _stage(prompt)
        if st in self.overrides:
            out = self.overrides[st]
            return out(prompt) if callable(out) else out
        return self.oracle(prompt)


# ---------------------------------------------------------------------------
# A perfect LLM reproduces the oracle
# ---------------------------------------------------------------------------

def test_perfect_budget_bid_reproduces_the_oracle_in_every_mode_and_regime():
    for mode in MODES:
        for regime in REGIMES:
            m = ColumnBid(_perfect(), INST, mode, regime, "budget")
            for p in PILOTS:
                r = m.rank(p, CANDS, seed=3)
                x = INST.extra_metrics(r)
                assert r.ordered_ids == INST.oracle_ranking(p).ordered_ids, (m.name, p.id)
                assert x["selection_f1"] == 1.0 and x["weight_error"] == 0.0, (m.name, x)
                assert x["direction_accuracy"] in (1.0, None)
                assert r.artifacts["repaired"] == []


def test_rank_format_selects_perfectly_but_loses_weight_information():
    m = ColumnBid(_perfect(), INST, "separated", "R1", "rank")
    rhos = []
    for p in PILOTS:
        r = m.rank(p, CANDS, seed=3)
        x = INST.extra_metrics(r)
        assert x["selection_f1"] == 1.0
        assert abs(sum(r.artifacts["bid"]["weights"].values()) - INST.budget) < 1e-9
        rhos.append(M.spearman(r.ordered_ids, INST.oracle_ranking(p).ordered_ids))
    assert min(rhos) < 1.0


def test_call_counts_per_regime():
    expected = {("separated", "R1"): 2, ("separated", "R2"): 2, ("separated", "R3"): 3,
                ("joint", "R1"): 1, ("joint", "R2"): 1, ("joint", "R3"): 2}
    for (mode, regime), n in expected.items():
        client = _perfect()
        r = ColumnBid(client, INST, mode, regime).rank(RICH, CANDS, seed=1)
        assert r.artifacts["n_llm_calls"] == n == client.n_calls, (mode, regime)


# ---------------------------------------------------------------------------
# Information regimes
# ---------------------------------------------------------------------------

def test_selection_never_sees_the_trips_and_r1_weighting_neither():
    client = _perfect()
    ColumnBid(client, INST, "separated", "R1").rank(RICH, CANDS, seed=1)
    sel, wgt = client.prompts
    assert _stage(sel) == "selection" and _stage(wgt) == "weighting"
    for prompt in (sel, wgt):
        assert not re.search(r"^P\d+ \|", prompt, re.M), "trip rows leaked"
    assert "You do not see this month's trips" in wgt


def test_r2_shows_every_trip_on_the_selected_columns():
    client = _perfect()
    ColumnBid(client, INST, "separated", "R2").rank(RICH, CANDS, seed=1)
    wgt = client.prompts[1]
    rows = re.findall(r"^P(\d+) \|", wgt, re.M)
    assert sorted(map(int, rows)) == sorted(p.id for p in CANDS)
    header = next(l for l in wgt.splitlines() if l.startswith("trip | dates"))
    for k in INST.oracle_bid(RICH).weights:
        assert k in header


def test_r3_splits_the_budget_and_can_sample_trips():
    client = _perfect()
    r = ColumnBid(client, INST, "separated", "R3", alpha=0.25, sample_size=8).rank(RICH, CANDS, seed=5)
    wgt, inc = client.prompts[1], client.prompts[2]
    assert "Budget: 7.5" in wgt and "You do not see" in wgt
    assert _stage(inc) == "increment" and "Budget: 2.5" in inc
    assert len(re.findall(r"^P\d+ \|", inc, re.M)) == 8
    assert abs(sum(r.artifacts["bid"]["weights"].values()) - INST.budget) < 1e-9


def test_r4_runs_feedback_rounds_when_the_shortlist_is_poor():
    # A weighting reply that puts the whole budget on the oracle's least
    # important column produces a shortlist the oracle does not like, so the
    # feedback should trigger an adjust round.
    truth = INST.oracle_bid(RICH).weights

    def lopsided(prompt):
        keys = [k.strip() for k in re.search(r"^Columns to weight: (.*)$", prompt, re.M).group(1).split(",")]
        budget = float(re.search(r"^Budget: ([\d.]+)", prompt, re.M).group(1))
        worst = min(keys, key=lambda k: truth.get(k, 0.0))
        return json.dumps({"weights": {k: (budget if k == worst else 0.0) for k in keys}})

    client = StubClient([Scripted({"weighting": lopsided})])
    r = ColumnBid(client, INST, "separated", "R4", max_rounds=2).rank(RICH, CANDS, seed=1)
    hist = r.artifacts["feedback_history"]
    assert hist and r.artifacts["feedback_rounds"] >= 1, hist
    assert any(_stage(p) == "adjust" for p in client.prompts)
    assert any(v != "great" for _, v, _ in hist[0]["feedback"])
    # The oracle's adjust reply restores the oracle weights.
    assert INST.extra_metrics(r)["weight_error"] == 0.0


def test_no_priorities_variant_drops_the_block():
    client = _perfect()
    m = ColumnBid(client, INST, "separated", "R1", include_priorities=False)
    m.rank(RICH, CANDS, seed=1)
    assert m.name.endswith("/noprio")
    assert all("USUAL PRIORITIES" not in p for p in client.prompts)


def test_prompts_carry_the_instructions_and_the_calendar():
    client = _perfect()
    ColumnBid(client, INST, "separated", "R1").rank(RICH, CANDS, seed=1)
    for ins in INST.instructions[RICH.id]:
        assert ins.text in client.prompts[0]
    assert "Thu 1  Fri 2  Sat 3  Sun 4" in client.prompts[0]


# ---------------------------------------------------------------------------
# Schema violations: retry, then repair
# ---------------------------------------------------------------------------

def test_bad_weights_are_retried_with_the_violation_stated():
    replies = iter([json.dumps({"weights": {"tafb": 99}})])

    def once_bad(prompt):
        try:
            return next(replies)
        except StopIteration:
            return OracleResponder(INST)(prompt)

    client = StubClient([Scripted({"weighting": once_bad})])
    r = ColumnBid(client, INST, "separated", "R1").rank(RICH, CANDS, seed=1)
    assert r.artifacts["n_llm_calls"] == 3
    assert r.artifacts["repaired"] == []
    retry = client.prompts[2]
    assert "Your previous reply was not valid" in retry and "missing weights" in retry
    assert INST.extra_metrics(r)["weight_error"] == 0.0


def test_persistently_bad_weights_are_repaired_and_flagged():
    def always_off(prompt):
        keys = [k.strip() for k in re.search(r"^Columns to weight: (.*)$", prompt, re.M).group(1).split(",")]
        return json.dumps({"weights": {k: 1 for k in keys}})   # sums to len(keys), not 10

    client = StubClient([Scripted({"weighting": always_off})])
    r = ColumnBid(client, INST, "separated", "R1", max_retries=2).rank(RICH, CANDS, seed=1)
    assert r.artifacts["repaired"] == ["weighting"]
    assert r.artifacts["n_llm_calls"] == 1 + 3
    w = r.artifacts["bid"]["weights"]
    assert abs(sum(w.values()) - INST.budget) < 1e-9
    assert len(set(round(v, 9) for v in w.values())) == 1   # rescaled uniformly


def test_selection_errors_unknown_column_and_bad_exclusion():
    bad = json.dumps({"selected": [
        {"key": "tafb", "direction": -1, "hard_exclusion": True},     # continuous exclusion
        {"key": "touches_day_99", "direction": -1},                     # no such day
        {"key": "credit_pay", "direction": 1},
    ]})
    client = StubClient([Scripted({"selection": bad})])
    r = ColumnBid(client, INST, "separated", "R1", max_retries=1).rank(RICH, CANDS, seed=1)
    retry = client.prompts[1]
    assert "cannot be a hard exclusion" in retry and "'touches_day_99' is not a column" in retry
    assert r.artifacts["repaired"] == ["selection"]
    bid = r.artifacts["bid"]
    assert "touches_day_99" not in bid["weights"] and set(bid["weights"]) == {"tafb", "credit_pay"}


def test_joint_weights_must_sum_to_the_budget():
    def joint_off(prompt):
        d = json.loads(OracleResponder(INST)(prompt))
        for it in d["selected"]:
            if "weight" in it:
                it["weight"] *= 2
        return json.dumps(d)

    replies = [joint_off, OracleResponder(INST)]
    client = StubClient(replies)
    r = ColumnBid(client, INST, "joint", "R1").rank(RICH, CANDS, seed=1)
    assert r.artifacts["n_llm_calls"] == 2
    assert "sum to exactly 10" in client.prompts[1]
    assert r.ordered_ids == INST.oracle_ranking(RICH).ordered_ids


def test_unparseable_reply_raises_after_retries():
    client = StubClient(["I think tafb matters most."])
    try:
        ColumnBid(client, INST, "separated", "R1", max_retries=1).rank(RICH, CANDS, seed=1)
    except ValueError as exc:
        assert "selection" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------

def test_runner_on_a_pbs_instance_records_bid_metrics_and_cost():
    methods = [ColumnBid(_perfect(), INST, "separated", "R1"),
               ColumnBid(_perfect(), INST, "joint", "R3")]
    res = ExperimentRunner(INST).run(methods, n_reps=2)
    assert not res["failures"]
    assert len(res["runs"]) == 2 * len(PILOTS) * 2
    run = res["runs"][0]
    for key in ("selection_f1", "weight_error", "compliance_rate", "top10_overlap"):
        assert key in run["metrics"]
    assert run["cost"]["llm_calls"] == 2
    agg = {a["method"]: a for a in res["aggregate"]}
    assert agg["column_bid/separated/R1/budget"]["spearman"] == 1.0
    assert agg["column_bid/joint/R3/budget/a0.2"]["llm_calls"] == 2.0
    assert agg["column_bid/separated/R1/budget"]["consistency"] == 1.0
    meta = res["metadata"]
    assert meta["instance"]["kind"] == "pbs"
    rich = next(p for p in meta["pilots"] if p["id"] == RICH.id)
    assert [i["text"] for i in rich["monthly_instructions"]] == \
           [i.text for i in INST.instructions[RICH.id]]
    json.dumps(res, default=str)


def test_runner_still_runs_the_bid_line_methods():
    from strategies.rank_all import RankAll

    inst = Instance.build(n_pilots=2, n_lines=6, pairings_per_line=4)

    def reply(prompt):
        ids = [int(x) for x in re.findall(r"^LINE (\d+) —", prompt, re.M)]
        return json.dumps([{"lineId": i, "rank": n, "eligible": True, "shortReason": "",
                            "pros": [], "cons": []} for n, i in enumerate(ids, 1)])

    res = ExperimentRunner(inst).run([RankAll(StubClient([reply]))], n_reps=1)
    assert not res["failures"]
    assert res["metadata"]["instance"]["kind"] == "bid_line"
    assert res["aggregate"][0]["spearman"] is not None
    assert res["runs"][0]["cost"]["llm_calls"] == 1


def test_schedule_outcome_metric():
    p = RICH
    top = [q for q in CANDS if q.id in INST.oracle_ranking(p).ordered_ids[:3]]
    out = INST.schedule_outcome(p, top)
    scores = INST.oracle_ranking(p).scores
    assert abs(out["oracle_satisfaction"] - sum(scores[q.id] for q in top)) < 1e-4
    assert out["n_trips"] == 3


# ---------------------------------------------------------------------------
# LLM client: reply extraction
# ---------------------------------------------------------------------------

def test_anthropic_reply_skips_thinking_and_raises_on_refusal_or_truncation():
    from types import SimpleNamespace as NS
    from llm_client import LLMError, _anthropic_text

    ok = NS(stop_reason="end_turn", content=[NS(type="thinking", thinking=""),
                                             NS(type="text", text='{"a": '), NS(type="text", text="1}")])
    assert _anthropic_text(ok) == '{"a": 1}'
    for bad in (NS(stop_reason="refusal", stop_details=NS(category="cyber"), content=[]),
                NS(stop_reason="max_tokens", content=[NS(type="text", text='{"a"')]),
                NS(stop_reason="end_turn", content=[NS(type="thinking", thinking="")])):
        try:
            _anthropic_text(bad)
        except LLMError:
            pass
        else:
            raise AssertionError(f"expected LLMError for {bad.stop_reason}")


def test_client_sends_no_temperature_to_anthropic():
    from llm_client import LLMClient

    c = LLMClient()
    assert c.temperature is None and c.max_tokens == 16000
    assert c._openai_sampling() == {}
    assert LLMClient(provider="openai", temperature=0.2)._openai_sampling() == {"temperature": 0.2}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_bid_metrics():
    prf = M.selection_prf({"a", "b", "c"}, {"a", "b", "d", "e"})
    assert prf == {"precision": 0.6667, "recall": 0.5, "f1": 0.5714}
    assert M.selection_prf(set(), set())["f1"] == 1.0
    assert M.selection_prf({"a"}, set())["f1"] == 0.0
    assert M.weight_error({"a": 7, "b": 3}, {"a": 7, "b": 3}, 10) == 0.0
    assert M.weight_error({"a": 10}, {"b": 10}, 10) == 1.0
    assert M.weight_error({"a": 7, "b": 3}, {"a": 5, "c": 5}, 10) == 0.5
    assert M.direction_accuracy({"a": 1, "b": -1}, {"a": 1, "b": 1}) == 0.5
    assert M.direction_accuracy({"a": 1}, {"b": 1}) is None
    assert M.top_k_overlap([1, 2, 3, 4], [2, 1, 9, 8], 2) == 1.0
    assert M.top_k_overlap([1, 2, 3, 4], [2, 1, 9, 8], 4) == 0.5
    assert M.rate([True, None, False, True]) == 0.6667
    assert M.rate([None]) is None


# ---------------------------------------------------------------------------

def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
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
    sys.exit(_run_all())
