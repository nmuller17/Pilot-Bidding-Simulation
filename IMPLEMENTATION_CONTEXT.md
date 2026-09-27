# IMPLEMENTATION_CONTEXT — LLM column bids → robust per-pilot pairing ranking

Context file for whoever (human or coding agent) continues the implementation of the research note
*"From Bid-Line Ranking to Preferential Bidding: LLM-Generated Pilot Bids with Separated Column Selection
and Weighting"* (N. Muller, 23 Sep 2026) in this repo. Read it fully before touching code.

Last updated: 26 Sep 2026 · repo HEAD `b0d992e` (main) · 65/65 tests pass.

---

## 0. Goal and scope

**Primary goal:** a working, *robust* method that turns (pilot profile θ_p + monthly instructions ι_p^t)
into one ranking r_p of the month's pairings per pilot, through an interpretable LLM bid (z_p, w_p).

| Note step | In scope now? |
|---|---|
| 1 Pilot data + monthly instructions | yes |
| 2 Feature library K, matrix Φ | yes |
| 3 Oracle in bid form (z\*, w\*) | yes |
| 4 Selection stage | yes |
| 5 Weighting stage, regimes R1–R4, budget vs rank | yes |
| 6 Pairing ranking (eq. 2 + hard exclusions) | yes |
| 7 PBS solve (seniority, lexicographic) | **later** |
| 8 Bid-line baseline (existing PoC, used as benchmark) | **later** — keep that code intact |
| 9 Evaluation §5 | yes, except the schedule-outcome row (needs Step 7) |
| §6 Preferences inside the CPP | **later** |

Core equations (note §2), implemented in `bid.py`:

- Bid: z_p ∈ {0,1}^|K|, w_p ≥ 0, Σ_k w_pk = B (B = 10), w_pk ≤ B·z_pk.
- Score: s_p(j) = Σ_k z_pk · w_pk · σ_k · φ̃_jk, with φ̃ min-max normalised over J, σ_k ∈ {−1,+1}.
- Ranking: sort by s_p(j) desc; firm-instruction pairings go to an *avoid* tier at the bottom;
  pairings the pilot is not qualified for are *ineligible*.

---

## 1. Repository architecture

### 1.1 Branch state — do this first

- `main` (checked out locally) is **flat**: every module sits at the repo root.
- The restructure into `backend/ frontend/ data/` is **already committed and pushed** on
  `origin/method-d-indicator-ranking` (commit `6fc2776`, one commit on top of `b0d992e`). It moves every
  module into `backend/`, adds `backend/paths.py` (PROJECT_ROOT, FRONTEND_DIR, DATA_DIR, RESULTS_DIR,
  ENV_FILE), moves `pilot_bidding_poc.html` → `frontend/`, `results.html` → `data/results/`,
  `env.example` → `.env.example`, `requirements-backend.txt` → `backend/requirements.txt`, fixes test
  imports, and deletes `models/pilot.py`.
- **Action (on Windows, not from the Linux VM):**
  ```bash
  git fetch origin
  git checkout main
  git merge --ff-only origin/method-d-indicator-ranking   # fast-forward, main has nothing new
  python -m pytest -q                                      # expect 65 passed
  git push origin main
  ```
  `models/pilot.py` was already removed from the working tree (see §1.4), so git will show it as deleted
  before the merge; the fast-forward deletes it too, so there is no conflict. If git complains, run
  `git checkout -- models/pilot.py` first, then merge.

- Line endings: files are CRLF on disk. Seen from a Linux shell without `core.autocrlf`, every file
  looks "modified" (the diff is CR-only). Add a `.gitattributes` with `* text=auto eol=lf` (or keep
  `core.autocrlf=true` on Windows) so this noise never reaches a commit.

### 1.2 Module map (paths after the merge: prefix `backend/`)

**A. PBS column-bid pipeline — the code to work on**

| Module | Role | Note step |
|---|---|---|
| `generator.py` | Synthetic pilots (`ScenarioGenerator`, archetypes, OracleWeights) and pairings with calendar dates | 1, instance |
| `instructions.py` | Monthly instruction generator, ground-truth `Effect`s, `compliance()` checks | 1 |
| `features.py` | `FeatureLibrary` (K: name, description, σ, type), `FeatureMatrix` (Φ, min-max) | 2 |
| `bid.py` | `Bid`, `oracle_bid`, `base_oracle_columns`, `score_pairings`, `rank_pairings`, `roc_weights`, `normalise_to_budget` | 3, 6 |
| `bid_prompts.py` | Selection / weighting / joint / feedback prompts | 4, 5 |
| `strategies/column_bid.py` | `ColumnBid`: modes `separated`/`joint` × regimes `R1..R4` × formats `budget`/`rank`; JSON parsing, retry, flagged repair | 4, 5 |
| `pbs_instance.py` | `PBSInstance`: pilots + pairings + instructions + Φ + oracle; `extra_metrics` (§5) | 9 |
| `metrics.py` | Spearman, Kendall τ-b, top-k, selection P/R/F1, weight error, direction accuracy, compliance, oracle satisfaction | 9 |
| `oracle_responder.py` | Answers LLM prompts from the oracle (`--dry-run`) | test tool |
| `harness.py` | `ExperimentRunner`, `Instance`, `Ranking`, `LLMSettings` (shared with bid-line) | runner |
| `llm_client.py` | `LLMClient` (Anthropic/OpenAI, retries), `StubClient` | runner |
| `run_pbs_experiment.py` | CLI entry point | runner |
| `models.py` | Dataclasses (`Pilot`, `Pairing`, `Leg`, `Line`, …) + schedule legality | shared |

**B. Legacy bid-line PoC — Step 8 benchmark. Keep, do not extend, do not import from new code.**

`main.py` (CLI, 1.8k lines), `prompt_builder.py`, `oracle.py`, `evaluator.py` (Bradley–Terry),
`allocator.py`, `strategies/{rank_all,scoring,pairwise}.py`, `llm_api.py`, `proxy_server.py`,
`pilot_bidding_poc.html` (UI), `results.html` (old bid-line output).

**Coupling to cut (A must not depend on B except `models.py`, `harness.py`, `generator.py`):**

1. `bid_prompts.py` imports `prompt_builder._priority_list` and `_pilot_profile` → replace (WP1).
2. `strategies/__init__.py` eagerly imports the three bid-line strategies, so importing
   `strategies.column_bid` loads bid-line code → make `__init__.py` empty (or lazy).
3. `harness.py` lazily imports `oracle.oracle_rank_lines` (bid-line only; fine as long as it stays lazy).
4. `llm_api.py` duplicates `llm_client.py`; only the bid-line UI uses it. Leave it until Step 8, then
   port `proxy_server.py` to `llm_client` and delete `llm_api.py`.

### 1.3 Hygiene

- `.claude/settings.local.json` is tracked but is machine-local and stale (it references
  `models/pilot.py`). `git rm --cached .claude/settings.local.json` and add `.claude/` to `.gitignore`.
- Results go to `data/results/` (after the merge `RESULTS_DIR` is the default output); results JSON
  should be committed only for runs cited in a write-up.
- `.gitignore` also needs `.pytest_cache/`.

### 1.4 Cleanup already done (26 Sep)

- Deleted `models/pilot.py` and the empty `models/` folder: a stale copy of `Pilot`/`OracleWeights`
  shadowed by `models.py` (a `.py` module wins over a namespace folder, so nothing imported it).
  The restructure commit deletes it too. Tests: 65 passed before and after.
- Nothing else was deleted: every other file is either part of the pipeline (A) or the Step 8
  benchmark (B).

---

## 2. Status per note step

| Step | Status | Gap blocking a *robust* ranking |
|---|---|---|
| 1 Pilots + instructions | Done: seeded profiles; instructions for days/weekends off (soft/firm), evening home, long/short trips, layover want/avoid, no early reports, one layover in X, n free weekends; ground-truth effects + compliance | Only 5 pilots, 4 of them family archetype, all dual-qualified (qualification never tested) → WP3 |
| 2 Feature library | Done: 16 fixed columns + `touches_day_d`, `away_evening_d`, `layover_CODE`, `sched_layover_CODE`, `sched_free_weekends_n` (~120 columns) | Φ is almost one-dimensional (see below) and 8 fixed columns are constant → WP2 |
| 3 Oracle in bid form | Done: base weights rescaled to B; soft instructions take a capped share (≤ 60 %); firm → exclusions; schedule-level → prefs | Base weights nearly cancel for family pilots (credit 40 vs TAFB+hotel 48) → ranking flips under small weight errors → WP2 |
| 4–5 Selection / weighting | Done: separated + joint, R1–R4, budget + rank (ROC), R3 α and sampling, R4 rounds | **Prompt tells the LLM a different priority order than the oracle uses** → WP1. No aggregation over repeated bids → WP4. R4 feedback comes from the oracle ranking (leaks ordinal info) → WP5 |
| 6 Score + ranking | Done: tiers eligible / avoid / ineligible, per-pilot σ | — |
| 9 Metrics | Done except schedule outcome | ρ inflated by the avoid tier, no confidence intervals, no stability metric → WP5 |
| Real LLM runs | **None yet** (only `--dry-run`: budget variants ρ = 1.00, rank variants 0.72, R3-rank 0.86) | → WP6 |

Measured problem with Φ: the generator sets `nights_away = num_legs − 1`, so `hotel_nights`,
`calendar_days`, `num_legs` correlate at 1.00 and `tafb`, `per_diem`, `credit_pay` at 0.93–1.00.
Perturbing the oracle weights (lognormal σ = 0.5, 200 draws) gives ρ mean / p10 of 0.68 / −0.17
(Sam Rivera), 0.75 / 0.22 (Morgan Kim), 0.97 / 0.94 (Dana Okafor, mid-career). Under this collinearity
weight error and selection metrics are not identifiable either: very different bids give the same
ranking.

---

## 3. Work packages (do them in this order)

Each WP ends with `python -m pytest -q` green and a commit.

### WP0 — Merge the restructure and fix hygiene (≈ 30 min)

Do §1.1 and §1.3. Make `strategies/__init__.py` empty and fix any import that relied on it
(`tests/test_harness_identity.py` imports the bid-line strategies by module path, so it keeps working).

**Done when:** `main` has `backend/ frontend/ data/`, tests pass, `git status` clean on Windows.

### WP1 — Align the prompt with the oracle (highest priority, small)

*Why:* `_priority_list` (from the bid-line prompt) tells family and mid-career pilots that TAFB is #1
and credit pay #4, while the oracle gives `credit_pay` the largest weight (40 / 57 of 100). It also
mentions per diem, start day and destinations, which the oracle ignores. A faithful LLM is scored as
wrong, so every current metric partly measures this mismatch.

*What:*
1. In `bid_prompts.py`, add `profile_priorities(pilot) -> str` built from
   `bid.base_oracle_columns(pilot)`: list the base columns **in decreasing weight order**, each with an
   explicit direction ("prefers **higher** credit pay", "prefers **shorter** TAFB", "prefers **fewer** /
   **more** hotel nights", "prefers **later** report") and a coarse strength word
   (e.g. dominant / important / minor from the share), **not** the exact numbers (otherwise weighting
   is copied, not inferred).
2. Add `profile_text(pilot)` in `bid_prompts.py` (age, family status, qualifications, seniority,
   hourly pay) and remove the import from `prompt_builder`.
3. Keep `--no-priorities` as the prior-only ablation (LLM infers from θ_p alone).
4. Optional third level `--priorities exact` (numeric shares shown) as an upper-bound control.

*Tests:* for each archetype, (a) every column named in the priority text is in `base_oracle_columns`,
(b) the order matches the weight order, (c) the text contains no `per diem`, `start day`,
`destination`.

*Done when:* dry-run numbers unchanged; the three priority levels (none / ordinal / exact) are
selectable from the CLI.

### WP2 — Decorrelated, realistic candidate set (the main modelling fix)

*Why:* the ranking can only be robust if Φ has several independent directions; today it is ≈ one
axis ("trip length").

*What (in `generator.py`, new function, keep the old one for the bid-line benchmark):*
1. Build pairings duty by duty: 1–4 duties, each duty 1–4 legs → legs and nights vary independently.
2. Layover length varies (10 h–30 h), so TAFB ≠ f(nights).
3. Report times spread over 04:30–14:00, releases up to 23:30; include red-eyes (≈ 10 %) and late
   releases so those columns are no longer constant.
4. Credit = max(block, duty guarantee, TAFB/3.5) per duty/pairing (Barnhart & Vaze pay-and-credit
   rule) so credit is not a linear function of length.
5. Start days spread over the whole month, weekends included; 150–300 pairings; several layover cities
   with a few repeats (needed by layover instructions).
6. **Preferred route:** an adapter `load_pairings(path)` that reads real pairings exported by the CPP
   pipeline (JSON/CSV: id, legs with times/airports/aircraft, duties, block, credit, TAFB, base) and
   assigns calendar dates. The synthetic generator stays as fallback and for tests.

*Diagnostics (new `backend/diagnostics.py`, CLI):* correlation matrix of continuous columns, list of
constant columns, number of principal components for 90 % variance, and the weight-perturbation test
(lognormal σ = 0.5 on oracle weights, 200 draws → mean and p10 ρ per pilot).

*Done when:* max |corr| between the base oracle columns ≤ 0.7; no constant fixed column except by
design; ≥ 4 components for 90 % variance; perturbation p10 ρ ≥ 0.5 for every archetype. If an
archetype still fails, adjust its base weights so no two opposing columns nearly cancel, and document
it.

### WP3 — Diversify pilots

*What:* ≥ 15 pilots, the three archetypes balanced (family / mid-career / early-career), ≥ 30 %
B737-only (so eligible sets differ and qualification filtering is actually tested), optional second
base. Keep the existing profile schema and pay table (2023 Delta contract) of the bid-line PoC.
Instruction generator: 1–4 instructions per pilot, ≥ 1 firm for ~30 % of pilots, ≥ 1 schedule-level
for ~30 %.

*Done when:* `PBSInstance.describe()` reports the mix; tests cover a B737-only pilot (no B767 pairing
ranked eligible).

### WP4 — Robustness layer (core of "robust ranking")

*Why:* one LLM call per pilot gives one noisy bid; the note's Stability metric needs repeated bids,
and PBS needs one final ranking.

*What (new `backend/robust.py` + a `ConsensusColumnBid` wrapper around `ColumnBid`):*
1. Draw K independent bids per pilot (default K = 5, same inputs, fresh calls, trip order shuffled
   when trips are shown: R2/R3/R4).
2. Aggregate:
   - selection: column kept if selected in ≥ ⌈K/2⌉ bids;
   - direction σ_k: majority vote (ties → profile default);
   - weights: per-column median over the K bids (0 when not selected), then `normalise_to_budget`;
   - exclusions / schedule prefs: majority vote.
3. Consensus ranking r_p = `rank_pairings` with the aggregated bid. Comparison aggregator: Borda
   count over the K rankings.
4. Per-pairing uncertainty: rank interval [p10, p90] of j's rank across the K bids; flag pairings
   whose interval crosses the top-k boundary.
5. Store all K bids, the aggregate and intervals in `ranking.artifacts`.

*New metrics (`metrics.py`):* run-to-run ρ over the K rankings; mean rank-interval width; top-k
stability (share of the consensus top-k present in ≥ 80 % of draws); perturbation stability of the
aggregated bid (same test as WP2).

*Done when:* dry-run with a noisy stub (oracle bid + lognormal noise) shows consensus ρ ≥ single-bid
ρ and narrower intervals as K grows (K = 1, 3, 5, 9).

### WP5 — Metric fixes

1. Report ρ on the **core set**: eligible pairings not in the avoid tier (`rho_core`), next to the
   full ρ. The avoid tier sits at the bottom of both rankings and inflates ρ.
2. Bootstrap 95 % CIs (resample pilots × reps, 2 000 draws) for every aggregated metric.
3. Report Kendall τ-b and top-5 / top-10 overlap with ρ.
4. R4: mark as **upper bound** in outputs (feedback is built from the oracle rank). Add a weaker
   feedback mode: per shortlist item only like / neutral / dislike + the instruction it violates, no
   ordinal information.
5. Log per run: model id, provider, max tokens, temperature (if any), seeds, K, git commit, prompts
   when `--keep-prompts`.

### WP6 — Real LLM runs

```bash
# smoke test: 1 pilot, 1 rep, cheapest variant
python backend/run_pbs_experiment.py --variants separated:R1:budget --pilots 1 --reps 1 --keep-prompts
# inspect prompts + raw answers, then the main grid
python backend/run_pbs_experiment.py --all-variants --pilots 15 --pairings 200 --reps 3 --keep-prompts
# ablations
python backend/run_pbs_experiment.py --all-variants --no-priorities ...
```

(Before WP0 the script is `run_pbs_experiment.py` at the root, output `pbs_results.json`.)
Variant syntax: `mode:regime:format`, mode ∈ {separated, joint}, regime ∈ {R1, R2, R3, R4},
format ∈ {budget, rank}. Estimate cost first: calls ≈ pilots × reps × K × (2 for separated, 1 for
joint) × (1 + R4 rounds).

Outputs to `data/results/pbs_<YYYYMMDD>_<model>_<tag>.json` + a short summary table (Markdown) of
the §5 metrics with CIs, per variant and per archetype.

### WP7 — Later (not now)

- Step 7: PBS solver, sequential lexicographic first (optimise pilot 1, fix score, next pilot), using
  s_p(j) or r_p as g_ip; schedule-level indicators ψ as extra yes/no terms; schedule-outcome metric
  `oracle_satisfaction` by seniority.
- Step 8: run the bid-line PoC on the same instance as benchmark.
- §6: leg-level column selection/weighting inside the CPP (Quesnel et al. 2020 style bonuses).

---

## 4. Research questions ↔ experiments

| Question (note §3) | Comparison | Main metric |
|---|---|---|
| Does separating selection and weighting help? | `separated:*` vs `joint:*`, same regime/format | selection F1, weight error, rho_core |
| Does seeing the pairings help weighting? | R1 vs R2 vs R3 (α ∈ {0.1, 0.2, 0.3}, with/without sampling) | rho_core, compliance |
| Does interaction help? | R4 (weak feedback) vs R1; R4 (oracle feedback) = upper bound | rho_core per round, calls |
| Budget vs ordinal bid? | `*:budget` vs `*:rank` | rho_core, weight error |
| How much comes from the stated priorities? | ordinal priorities vs `--no-priorities` vs exact | all |
| Is the ranking robust? | K = 1 vs consensus K = 5 | run-to-run ρ, interval width, perturbation p10 |

---

## 5. Definition of done — "robust per-pilot ranking"

On the WP2/WP3 instance (≥ 15 pilots, ≥ 150 pairings), with a real model and consensus K = 5, for
the best variant:

- rho_core ≥ 0.8 (95 % CI reported), top-10 overlap ≥ 0.7;
- firm-instruction compliance = 100 %, soft ≥ 80 %;
- run-to-run ρ ≥ 0.9; perturbation p10 ρ ≥ 0.5 for every archetype;
- all runs reproducible from the logged config (model id, seeds, commit).

These thresholds are targets to discuss with the supervisor, not fixed by the note.

---

## 6. Working rules

- Do not modify the bid-line modules (§1.2 B) except to cut coupling; they are the Step 8 benchmark.
- Tests never call an API: use `StubClient` / `OracleResponder`.
- Every new metric gets a unit test on a hand-computed case.
- Keep the oracle as a reference, not as ground truth about real pilots (it is not observed
  behaviour); say so in any write-up.
- Run git from Windows, not from a Linux shell on the mounted folder (line endings, lock files).
