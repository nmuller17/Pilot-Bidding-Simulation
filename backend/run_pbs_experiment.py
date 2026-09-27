"""
run_pbs_experiment.py
---------------------
Run column-selection / column-weighting bids (research note, sections 3-5) on a
PBS instance and write every run, per-pilot and aggregate metric to JSON.

Variants are written mode:regime:format, e.g.

    python run_pbs_experiment.py --variants separated:R1:budget joint:R1:budget \\
        separated:R2:budget separated:R3:budget separated:R4:budget --reps 3

    # every variant, no API calls: a scripted responder answers from the oracle
    python run_pbs_experiment.py --all-variants --dry-run

LLM settings default to the Parley Anthropic route configured in .env (see
llm_client.py). The PBS solver (Step 7) is not part of this script.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

from harness import ExperimentRunner, LLMSettings
from llm_client import LLMClient, StubClient
from paths import RESULTS_DIR
from pbs_instance import PBSInstance
from strategies.column_bid import MODES, REGIMES, WEIGHT_FORMATS, ColumnBid

SHOWN = [
    ("spearman", "rho"),
    ("kendall_tau_b", "tau"),
    ("top5_overlap", "top5"),
    ("selection_f1", "selF1"),
    ("weight_error", "wErr"),
    ("direction_accuracy", "dir"),
    ("compliance_rate", "compl"),
    ("consistency", "stab"),
    ("tie_rate", "ties"),
    ("llm_calls", "calls"),
]


def _parse_variant(text: str):
    parts = text.split(":")
    if len(parts) != 3 or parts[0] not in MODES or parts[1] not in REGIMES \
            or parts[2] not in WEIGHT_FORMATS:
        raise argparse.ArgumentTypeError(
            f"variant '{text}' must be mode:regime:format with mode in {MODES}, "
            f"regime in {REGIMES}, format in {WEIGHT_FORMATS}"
        )
    return tuple(parts)


def _print_table(aggregate) -> None:
    head = f"{'method':44s}" + "".join(f"{label:>7s}" for _, label in SHOWN)
    print(head)
    print("-" * len(head))
    for row in aggregate:
        cells = ""
        for key, _ in SHOWN:
            v = row.get(key)
            cells += f"{'—':>7s}" if v is None else f"{v:7.2f}"
        print(f"{row['method']:44s}{cells}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variants", nargs="+", type=_parse_variant,
                    default=[("separated", "R1", "budget")])
    ap.add_argument("--all-variants", action="store_true",
                    help="every mode x regime x format combination")
    ap.add_argument("--pilots", type=int, default=5)
    ap.add_argument("--pairings", type=int, default=40)
    ap.add_argument("--month", default="2026-10", help="bid month, YYYY-MM")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--top-k", type=int, default=10, help="shortlist size for compliance")
    ap.add_argument("--pilot-seed", type=int, default=1234567)
    ap.add_argument("--pairing-seed", type=int, default=42)
    ap.add_argument("--instruction-seed", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.2, help="R3 informed budget share")
    ap.add_argument("--sample-size", type=int, default=None, help="R3: show only this many trips")
    ap.add_argument("--shortlist", type=int, default=5, help="R4 shortlist size")
    ap.add_argument("--rounds", type=int, default=2, help="R4 feedback rounds")
    ap.add_argument("--no-priorities", action="store_true",
                    help="omit the profile's usual-priorities block from the prompts")
    ap.add_argument("--keep-prompts", action="store_true", help="store prompts in the output")
    ap.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "anthropic"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=None,
                    help="OpenAI route only; Claude 5 models reject sampling parameters")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--dry-run", action="store_true",
                    help="answer every prompt from the oracle; no API calls")
    ap.add_argument("--on-error", choices=["raise", "record"], default="record")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "pbs_results.json"))
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    year, month = map(int, args.month.split("-"))
    inst = PBSInstance.build(
        n_pilots=args.pilots, n_pairings=args.pairings, bid_month=date(year, month, 1),
        pilot_seed=args.pilot_seed, pairing_seed=args.pairing_seed,
        instruction_seed=args.instruction_seed, budget=args.budget, top_k=args.top_k,
    )

    model = args.model or os.getenv(
        "ANTHROPIC_MODEL" if args.provider == "anthropic" else "OPENAI_MODEL", "claude-sonnet-5")
    settings = LLMSettings(args.provider, model, args.temperature, args.max_tokens)

    variants = ([(m, r, f) for m in MODES for r in REGIMES for f in WEIGHT_FORMATS]
                if args.all_variants else args.variants)
    methods = []
    for mode, regime, fmt in variants:
        if args.dry_run:
            from oracle_responder import OracleResponder
            client = StubClient([OracleResponder(inst)])
        else:
            client = LLMClient.from_settings(settings)
        methods.append(ColumnBid(
            client, inst, mode, regime, fmt, alpha=args.alpha, sample_size=args.sample_size,
            shortlist_size=args.shortlist, max_rounds=args.rounds,
            include_priorities=not args.no_priorities, keep_prompts=args.keep_prompts,
        ))

    print(f"PBS instance: {len(inst.pilots)} pilots, {len(inst.pairings)} pairings, "
          f"{len(inst.library.columns)} columns, bid month {inst.library.month_start:%B %Y}")
    for p in inst.pilots:
        texts = [("[firm] " if i.firm else "") + i.text for i in inst.instructions[p.id]]
        print(f"  #{p.seniority} {p.name} ({p.family_status}, {p.age}): "
              + ("; ".join(texts) if texts else "no instructions"))
    print(f"LLM: {'dry run (oracle responder)' if args.dry_run else settings.as_dict()}")
    print(f"Running {len(methods)} variant(s) x {len(inst.pilots)} pilots x {args.reps} reps\n")

    runner = ExperimentRunner(inst, settings)
    results = runner.run(methods, n_reps=args.reps, on_error=args.on_error)
    results["metadata"]["dry_run"] = args.dry_run
    ExperimentRunner.write(results, args.out)

    _print_table(results["aggregate"])
    if results["failures"]:
        print(f"\n{len(results['failures'])} run(s) failed:")
        for f in results["failures"][:10]:
            print(f"  {f['method']} pilot {f['pilot_id']} rep {f['rep']}: {f['error']}")
    repaired = sum(1 for r in results["runs"] if r["artifacts"].get("repaired"))
    if repaired:
        print(f"\n{repaired} run(s) needed a repaired reply (see artifacts.repaired).")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
