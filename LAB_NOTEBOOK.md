# Formula 110 — Laboratory Notebook

## Entry 1 — Reactive Controller Foundation (Strategy 1: Reactive Control + Parameter Optimization)

**Date and time:** August 30, 2026, evening session

**Participants and contributions:**
- Leon Aaron and Jayden Webb— designed the sensor-to-command control logic, wrote the controller and evaluation harness, ran and interpreted experiments, committed the result.
- AI coding agent (Claude Code) — read the simulator's sensor, physics, and scoring internals to ground design decisions in verified behavior, drafted the controller and evaluation script, ran the verification experiments, and proposed the parameter change based on the measured results.

**Question or objective:**
Can a hand-written reactive controller (explicit sensor→command rules with tunable parameters) drive the car around the full Mugello Short track and complete at least one lap within 30 seconds, from a random seeded starting position, without being destroyed (damage reaching 1.0)? This is the foundation for both our baseline and the parameter-optimization step that follows.

**What we investigated or changed:**
- Read the simulator's public contract (`src/racing/student/api.py`, `SENSORS.md`) and internals (`race/sensors.py`, `race/progress.py`, `race/runtime.py`, `physics/engine.py`) to confirm sensor sign conventions, the lap-counting rule (cumulative forward progress since spawn crossing the track length — not a literal return to the start/finish line), and the damage model (wall-impact damage scales with the *square* of impact impulse; elimination at damage ≥ 1.0).
- Measured track geometry directly: total closed-loop length ≈ 181.1 m, track width 6.6 m (~3.3 car-widths).
- Implemented `src/controllers/reactive.py` — a parametrized controller (`ReactiveParams` dataclass) that steers using centerline offset, heading error, and near/far lookahead offsets; adds a proximity-based wall-avoidance term; brakes proportionally to a speed-scaled stopping distance against `wall_lidar.front_m`; and falls back to an open-side recovery maneuver on any contact.
- Implemented `scripts/evaluate_controller.py` — a headless evaluation harness that races the controller against a passive baseline across seeded starting positions and reports scored distance, laps, damage, and marshal activity. This becomes the fitness function for the parameter-optimization stage.
- Ran the harness with a conservative speed target (`max_speed_mps = 9.5`); after observing zero damage, deliberately raised it (`14.0`) and loosened the cornering-slowdown threshold to spend the unused safety margin on speed.

**Evidence:**
- Sources or documentation: `SENSORS.md`, `README.md`, `src/racing/student/api.py`, `src/racing/race/sensors.py`, `src/racing/race/progress.py`, `src/racing/race/runtime.py`, `src/racing/race/head_to_head.py`, `src/racing/physics/engine.py`, `src/racing/track/world.py`.
- AI-agent assistance: Claude Code explored the simulator internals above to verify design assumptions before writing code, then implemented both files and ran the evaluation script itself, iterating parameters based on printed results rather than submitting a first guess.
- Commits or code: commit `cc38354`, "Start with reactive control and parameter optimization" — adds `src/controllers/reactive.py` and `scripts/evaluate_controller.py`.
- Experiment output: `uv run python scripts/evaluate_controller.py --module controllers.reactive --seed <seed> --races 5 --round-seconds 30`, run across the recommended seed suite (42, 110, 271, 997, 2027).
- Leaderboard result: not yet submitted — this stage used local headless evaluation only.

**What we observed:**
- First pass (`max_speed_mps = 9.5`): 5/5 races survived with zero damage, but only 2/5 crossed the ~181 m lap threshold (177–187 m raw distance) — safe but not reliably fast enough to guarantee a lap.
- After raising `max_speed_mps` to 14.0 and the cornering-slowdown trigger from 35° to 50° of heading error: 5/5 survived, 5/5 completed ≥1 lap (avg. 234.5 m in 30 s), still zero damage.
- Re-running across all five recommended seeds (25 races total): **25/25 survived, 25/25 completed ≥1 lap, damage = 0.00 in every race**, with no marshal resets triggered.
- The one notable "failure" was the first speed setting — safe but too conservative to guarantee a lap. The fix came from the measured data (laps completed vs. distance short of threshold), not a guess.

**Decision and rationale:**
Adopted the `max_speed_mps = 14.0`, `corner_heading_error_deg = 50.0` configuration as the baseline reactive controller (`DEFAULT_PARAMS` in `src/controllers/reactive.py`). It satisfies the deliverable — survives and completes ≥1 lap within 30 s — with full damage margin (0.00 observed) across every tested seed, indicating room to push further in the optimization stage rather than trading speed for safety.

**Next steps:**
- Feed `scripts/evaluate_controller.py`'s output (avg. scored distance, penalized for any elimination) into an automated parameter search (Optuna or CMA-ES) over `ReactiveParams`.
- Widen evaluation to more seeds/races per trial before trusting an optimized parameter set, to avoid overfitting to the five recommended seeds.
- Begin the second experimental approach (neuroevolution over a small MLP) on the same evaluation harness, per the two-approach exploration requirement.

## Entry 2 — Neuroevolution Foundation (Strategy 2: Evolution Strategy over a Small MLP)

**Date and time:** August 30, 2026, evening session (continued)

**Participants and contributions:**
- Leon Aaron and Jayden Webb — reviewed each training result, decided when to change the fitness function vs. accept a result, and directed the scope of the experiment.
- [Add teammate name and contribution here.]
- AI coding agent (Claude Code) — implemented the network, feature encoding, and evolution-strategy trainer; ran four training iterations; diagnosed each failure mode from the printed generation-by-generation fitness and held-out validation output; proposed and implemented each fitness-function fix.

**Question or objective:**
Can a small neural-network controller (10 sensor inputs → 6-unit hidden layer → 2 outputs), with its weights evolved by a simple evolution strategy rather than hand-written, reach the same goal as Approach 1 — drive around the track and complete a lap from a random seeded start without being destroyed? This is the second, mechanistically distinct approach required by the exploration stage.

**What we investigated or changed:**
- Implemented `src/controllers/neuro.py` (network forward pass, sensor-to-feature encoding, `Genome` type) and `scripts/train_neuroevolution.py` (a dependency-free (mu + lambda) evolution strategy with elitism and Gaussian mutation), reusing `scripts/evaluate_controller.py` as the shared fitness/validation harness so results are directly comparable to Approach 1.
- Ran four training iterations, changing the fitness function each time in response to what the previous attempt's held-out validation revealed:
  1. `distance − 300 if eliminated`, single training seed → training fitness 225 m, but held-out validation showed the genome crashing in most races (17/25 eliminated) after covering enough distance to still score well — the penalty was too weak relative to achievable high-speed distance.
  2. Switched to a fixed penalty that fully replaces distance on elimination (survival-only credit) → training fitness dropped to 63 m, and validation showed the population had converged to standing nearly still (0/25 laps) — "never move" is a free, zero-risk optimum under a purely survival-dominant fitness.
  3. Added a graduated fitness (distance − 350 if eliminated, distance − 20 if idle below 10 m, else raw distance), single training seed → training fitness 260 m; validation showed genuinely fast driving (28–35 m/s, 21/25 laps completed) but still frequent elimination (8/25 survived) — safety learned on one starting position did not generalize.
  4. Same fitness as (3), trained on two seeds instead of one to force cross-seed safety → training fitness dropped to 53 m and reproduced attempt (2)'s idle genome exactly — the harder, multi-seed objective exceeded what a 10-genome, 8-generation search could solve.

**Evidence:**
- Sources or documentation: none beyond what was already read for Entry 1; this stage was empirical (iterate → measure → diagnose).
- AI-agent assistance: Claude Code wrote all four fitness-function versions and ran every training/validation pass; each change was motivated by inspecting the concrete failure in the previous attempt's output rather than a fresh guess.
- Commits or code: `src/controllers/neuro.py`, `scripts/train_neuroevolution.py` (not yet committed as of this entry).
- Experiment output: `uv run python scripts/train_neuroevolution.py [--population --generations --seeds --round-seconds]` for each attempt; `uv run python scripts/evaluate_controller.py --module controllers.neuro --seed <seed> --races 5 --round-seconds 30` across `(42, 110, 271, 997, 2027)` to validate each resulting genome.
- Leaderboard result: not applicable — local evaluation only.

**What we observed:**
- Every attempt that rewarded raw distance strongly enough to produce real driving also under-punished crashing relative to the distance a fast, reckless genome could rack up before crashing (attempts 1 and 3).
- Every attempt that punished elimination hard enough to suppress recklessness also made "not moving" a competitive or dominant strategy for this small a population/generation budget (attempts 2 and 4).
- Requiring safety across two training seeds simultaneously (attempt 4) was strictly harder for this search method than requiring it on one (attempt 3) — the search collapsed to the degenerate idle solution instead.
- No attempt matched Approach 1's result (25/25 survived and lapped). The best population/generation budget tested (10–12 genomes, 8–10 generations) is genuinely too small to resolve the speed/safety trade-off with this simple mutation-only evolution strategy.

**Decision and rationale:**
Kept attempt 3's genome as `BEST_GENOME` in `src/controllers/neuro.py` — it is the most informative result (demonstrates real racing behavior and a materially higher top speed than Approach 1, 28–35 m/s vs. 13.9 m/s) even though it is not yet reliable. Documented all four attempts rather than only the best one, since the failure modes themselves are the useful evidence for comparing the two approaches (see `EXPLORATION.md`).

**Next steps:**
- Increase the evolutionary search budget (larger population, more generations, or true CMA-ES with covariance adaptation) now that the fitness function's failure modes are understood.
- If neuroevolution is retained past the exploration stage, consider warm-starting the network by cloning Approach 1's reactive controller (behavior cloning) before evolving further, rather than starting from random weights.
- Proceed to the "Select and Refine" stage using Approach 1 as the primary controller, per the comparison in `EXPLORATION.md`.

## Entry 3 — Reactive Controller Refinement: Apex-Hugging Steering + Parameter Optimization

**Date and time:** August 31, 2026

**Participants and contributions:**
- Leon Aaron — proposed the specific behavioral changes (no proactive braking, infer turn sharpness from sensors, hug the inside of a turn), reviewed each result, and directed the scope of the optimization run.
- [Add teammate name and contribution here.]
- AI coding agent (Claude Code) — implemented the apex-hugging steering and no-proactive-braking redesign, built the parameter-search script and its diagnostic tracing, ran the search, and validated the result.

**Question or objective:**
Can the reactive controller from Entry 1 go substantially faster than its 229 m/30s baseline, using two concrete ideas: (1) inferring the sharpness of an upcoming turn from `camera.heading_error_degrees` and biasing the car toward the inside of that turn rather than the centerline, and (2) removing proactive cornering deceleration so throttle always targets top speed and only the reactive wall-proximity check can brake? And can a parameter search then push the resulting design further while an added diagnostic tool explains *why* a given parameter set does well or poorly?

**What we investigated or changed:**
- Redesigned `src/controllers/reactive.py`: added `_apex_target_offset_m`, which biases the steering "centering" target toward the inside of the turn (right side on a right turn, left side on a left turn) using `turn_severity = |heading_error_degrees| / turn_sharpness_deg`. Removed the old curvature-based speed target entirely — throttle now always targets `max_speed_mps`, with the wall-proximity brake as the only source of negative throttle.
  - Key mechanical insight that justifies apex-hugging in this simulator specifically: scored distance is the car's projection onto the fixed centerline, not its physical path length, so cutting the inside of a turn advances scored distance per meter actually driven — the geometry argument holds for *this* scoring rule, not just as general racing-line folklore.
- Verified the redesign alone (default parameters, no search yet) against the full seed suite before doing anything else: **25/25 survived, 25/25 laps, 0.00 damage, avg. 259 m/30s** — already +13% over Entry 1's 229 m, from apex-cutting alone at the same top speed.
- Built `scripts/optimize_reactive.py`: a (mu + lambda) evolution strategy seeded from the known-good default parameters (not random init — an explicit lesson carried over from Entry 2's random-init struggles), with per-parameter mutation scaled to each parameter's own magnitude, and the same three-tier fitness (elimination penalty, idle penalty, else raw distance) that worked for the neuroevolution trainer.
- Added a `--diagnose` mode that reruns one race with full per-tick tracing (sensors, command, and simulated time) and prints: how often the car braked, how often a wall was close, the sensor/command state at the first contact, and the final ticks before an elimination — a concrete answer to "why did this do well or poorly", not just a fitness number.
- Ran `--diagnose` on the pre-search defaults first: revealed the wall-proximity brake was firing on **43% of ticks** — not from any cornering logic (already removed), but because its safety margin (`speed × 1.1s` lead time) scales up to ~15 m at top speed on a track this curvy, making the "reactive-only" brake accidentally proactive. This was a concrete, sensor-and-time-stamped finding, not a guess.
- Ran the search (population 10, generations 10, elite 3, training seeds 13 and 55 — distinct from the held-out suite, 20-second rounds): fitness rose from 189.8 m (generation 0) to 289.3 m (generation 9).

**Evidence:**
- Sources or documentation: none new; built on Entry 1's sensor/scoring findings.
- AI-agent assistance: Claude Code implemented the redesign, the search script, and the diagnostic tool; used the diagnostic tool's own output (the 43% braking finding) to decide the search should include the brake thresholds as free parameters rather than assuming a fixed value.
- Commits or code: `src/controllers/reactive.py` (redesigned + optimized `DEFAULT_PARAMS`), `scripts/optimize_reactive.py` (new).
- Experiment output: `uv run python scripts/optimize_reactive.py --population 10 --generations 10 --seeds 13 55 --round-seconds 20` for the search; `uv run python scripts/evaluate_controller.py --module controllers.reactive --seed <seed> --races 5 --round-seconds 30` across `(42, 110, 271, 997, 2027)` for held-out validation.
- Leaderboard result: not applicable — local evaluation only.

**What we observed:**
- The optimized parameters cut braking from 43% of ticks to about 1%, raised `max_speed_mps` from 14.0 to 15.8, and tightened the apex bias (`apex_bias_max_m` settled at 0.46 m with a much smaller `turn_sharpness_deg` of 13°, i.e. a sharper, earlier bias reaction than the hand-picked defaults).
- Held-out validation across all five recommended seeds (25 races): **25/25 survived, 25/25 completed 2 laps (not just 1), damage = 0.00 in every race**, average scored distance **≈435 m per 30 s** — roughly **90% more distance than Entry 1's original 229 m**, with an equally perfect safety record.
- No failed attempt to report this time — seeding the search from a known-good baseline (rather than random init, as neuroevolution was forced to use) meant every generation's worst genome was still a plausible driver, not a catastrophic one; the search converged smoothly without the degenerate local optima seen in Entry 2.

**Decision and rationale:**
Adopted the optimized parameter set as the new `DEFAULT_PARAMS`. The evidence (25/25 survival and lap completion at every tested seed, zero damage, and a large, real speed increase) clearly supports keeping it as the primary controller, and confirms the apex-hugging/no-proactive-braking hypothesis: the improvement came from a combination of the redesigned control law (+13% before any search) and the parameter search built on top of it (+68% more on top of that).

**Next steps:**
- Try a larger search budget (more generations/population) now that a single search run took under 3 minutes wall-clock, to see if further gains are available.
- Investigate whether `steer_limit` settling at 0.63 (well below 1.0) indicates the car is leaving speed on the table in the sharpest turns; consider whether a track-position-aware `apex_bias_max_m` (larger in sharper, slower corners) would help.
- This result is now the baseline to beat for the "Select and Refine" stage; compare any future controller (including a re-attempted, better-budgeted neuroevolution run) against this ~435 m figure, not the original 229 m.

*Entries 4-9 below are merged in from a parallel local copy of this project (`../formula110`) where this work actually happened, between Entry 3 and the September 8 learned-controller work. That copy's code (`src/controllers/reactive2.py`, `scripts/optimize_reactive2.py`, `scripts/try_params.py`, `scripts/validate_reactive2_candidate.py`) was never carried into this repository and is not present here — none of it beat the Entry 3 baseline, so it wasn't worth bringing over. The journal entries are kept for the record: they document real experiments, including two independently-confirmed negative results (Entries 8-9) that close off the "just add more reactive sensor mechanisms" direction.*

## Entry 4 — Failed Attempt: Anticipatory Apex Bias + Higher-Speed Search (Reverted)

**Date and time:** August 31, 2026

**Participants and contributions:**
- Leon Aaron — requested the specific behavior (drive straight through small chicanes, hug the inside of turns harder, target ~2x distance) and, after reviewing the risky validation numbers, called for reverting to the known-good Entry 3 state and asked for manual tuning guidance instead of further automated search.
- AI coding agent (Claude Code) — implemented the redesign, ran two search attempts, diagnosed why each was unsafe, and performed the revert.

**Question or objective:**
Could the reactive controller reach roughly double Entry 3's distance (~435 m → ~870 m) by (1) making the apex-bias steering anticipatory — driven by the farthest camera lookahead point (~16 m ahead) instead of instantaneous heading error, so a brief chicane nets out straight while a sustained turn still gets the full inside-hugging bias — and (2) searching a wider parameter range, including a much higher `max_speed_mps`?

**What we investigated or changed:**
- Verified the track geometry directly (segment headings from `MUGELLO_SHORT_LAYOUT`): confirmed a ~49-unit straight at the start and exactly two sharp right-hand turns (~50–52°, near "Luco climb" and "Materassi exit"), each in an otherwise left-trending section — consistent with what was reported by observation.
- Replaced the heading-error-driven apex bias with a lookahead-driven one (`lookahead_offsets_m[-1]`), and manually raised `max_speed_mps` to 20.0 and `apex_bias_max_m` to 1.2 as a search starting point.
- Ran a parameter search (population 12, generations 14, seeds 13 & 55, 20-second rounds): training fitness reached 314 m, but the trace showed **real, non-zero damage (0.27–0.41) on both training seeds** — the fitness function only penalized full elimination (damage ≥ 1.0), never partial damage, so it rewarded a genome that clips walls hard but happens not to fully break in a short training round.
- Validated that genome on the full held-out suite anyway: **10/25 survived**, with damage = 1.0 (full elimination) in most of the losses — confirming the fitness gap was real, not a fluke.
- Fixed the fitness function (`scripts/optimize_reactive.py`) to subtract a penalty proportional to non-fatal damage, not just elimination, and reran the search with the same budget.
- The corrected search converged to `max_speed_mps ≈ 19.1` with small training damage (0.06–0.11). Validated on the held-out suite: **19/25 survived, average distance 399.7 m** — safer than the first attempt, but still worse on both counts than Entry 3's 25/25-survived, 435 m result.

**Evidence:**
- AI-agent assistance: Claude Code ran both search attempts in the background, diagnosed the fitness-function gap from the trace output (not a guess — the 0.27–0.41 training damage was visible before validation confirmed the consequence), and reverted the shipped controller only after the second attempt also underperformed the known-good baseline.
- Commits or code: `scripts/optimize_reactive.py` retains the damage-penalty fix (a genuine correctness improvement, kept); `src/controllers/reactive.py` was reverted to the exact Entry 3 state and re-verified.
- Experiment output: two full background search runs (population 12, generations 14, seeds 13 & 55, 20s rounds; ~13–15 min wall-clock each) plus held-out validation runs (5 seeds × 5 races × 30s) after each.
- Leaderboard result: not applicable.

**What we observed:**
- Both search attempts found genomes that looked good on their two training seeds but generalized worse than the existing baseline on the held-out suite — the same "training seed(s) don't cover the whole risk surface" failure pattern seen with neuroevolution in Entry 2, now reproduced in the reactive-controller search too.
- Fixing the fitness function (penalizing damage, not just elimination) measurably reduced training-time recklessness (0.27–0.41 damage → 0.06–0.11 damage) but was still not enough alone to beat the existing 435 m/0-damage baseline within this search budget.
- Average distance is not a safe stand-in for "twice as fast without being destroyed": both search attempts had *higher peak per-race distances* (up to 456 m) than Entry 3, but lower *average* distance once eliminations are included, and violated the hard "not destroyed" requirement on a meaningful fraction of races.

**Decision and rationale:**
Reverted `src/controllers/reactive.py` to the exact Entry 3 parameter set (25/25 survived, 0.00 damage, ~435 m) rather than keep either riskier result. Two training seeds with a 20-second round appears to be too small a sample of this track's risk surface for this search budget to reliably find a genuinely safer-and-faster optimum — the search kept discovering the same trade-off (more speed, more wall contact) without enough pressure to escape it. Manual, incremental tuning — changing one parameter at a time and validating against all 5 held-out seeds after each change — is the next step, so a human can catch a "looks great on training seeds, fails on held-out seeds" result immediately rather than trusting one aggregate fitness number.

**Next steps:**
- Manually probe the safe ceiling of `max_speed_mps` in small increments from the Entry 3 baseline, validating on all 5 seeds after each change (see the guidance given directly to the team for this).
- If returning to automated search, use 3+ training seeds instead of 2, and/or increase the damage penalty scale further, before trusting a search result over Entry 3's baseline.
- Consider the lookahead-driven anticipatory apex bias again later — the idea itself was not disproven, it was just tested simultaneously with a large, uncontrolled speed increase, which makes it impossible to tell which change caused the damage.

## Entry 5 — Neuroevolution: Five Training Seeds at a Scaled-Up Search Budget

**Date and time:** August 31, 2026 (continued)

**Participants and contributions:**
- Jayden Webb — proposed widening Entry 2's training seed count from two to five to directly target the cross-seed generalization failure, directed the sequencing of the experiment (change seed count first, observe, only then scale population/generations rather than changing both at once), requested held-out validation against the actual suite before trusting the training curve, watched the resulting genome race live, and directed the genome swap and this documentation update.
- AI coding agent (Claude Code) — implemented the seed-count change in `scripts/train_neuroevolution.py`, ran both training passes (unscaled and scaled budget), diagnosed why the first regressed, wrote a standalone held-out validation script, ran the graphical head-to-head to confirm the result visually, and performed the `BEST_GENOME` swap plus this entry and the `EXPLORATION.md` update.

**Question or objective:**
Entry 2 showed that training on more seeds (attempt 4, two seeds) made results *worse*, not better, collapsing to the same degenerate idle genome as the single-objective survival-only attempt. Was that because more training seeds are fundamentally the wrong lever, or because the population/generation budget (10 genomes, 6–8 generations) was already too small for two seeds and got asked to solve an even harder problem without more capacity? Concretely: does five-seed training, given a search budget scaled to match, produce a genome that survives the *held-out* suite — the specific failure every prior attempt shared — without giving up neuroevolution's speed advantage?

**What we investigated or changed:**
- Changed `scripts/train_neuroevolution.py`'s `--seeds` default from `[13, 55]` to `[13, 55, 7, 89, 233]` — five seeds, still kept distinct from the held-out validation suite `(42, 110, 271, 997, 2027)`.
- Ran training at the unchanged default budget (population 10, generations 6) first, deliberately, to isolate the effect of the seed-count change alone before touching anything else: **best training fitness converged to only 8.8 m**, with the `worst`-of-generation column pinned near the full −350 elimination penalty in every generation — confirming the same population could not satisfy five seeds' worth of requirement at all, let alone generalize.
- Reran with population and generations both scaled up (10→20, 6→15) to match the harder five-seed objective: **best training fitness converged to 235.8 m** by generation 9 and held there. The `worst` column still touched −350 in the final generation, showing the population was not uniformly safe even though its best genome was.
- Validated the resulting genome on the full held-out suite with a standalone script (not the shipped `controllers.neuro`, to avoid touching `BEST_GENOME` before the result was confirmed): `run_headless_head_to_head` against a passive baseline, all five held-out seeds, five races each (25 total), 30-second rounds — identical protocol to every other attempt.
- Watched the validated genome race live (`uv run racing h2h --watch`) against `controllers.reactive` (seed 110) to visually confirm the driving behavior before committing to the swap.
- Swapped the new genome into `BEST_GENOME` in `src/controllers/neuro.py` and re-ran `scripts/evaluate_controller.py --module controllers.neuro` on a held-out seed to confirm the shipped controller now reproduces the validation script's numbers exactly.

**Evidence:**
- Sources or documentation: none new; built on Entry 2's fitness-function design (unchanged three-tier scoring) and Entry 3/4's lesson about training-seed coverage vs. search budget.
- AI-agent assistance: Claude Code ran both training passes in the background, read the raw generation-by-generation output to diagnose the first pass's failure before proposing the budget increase, and wrote/ran the held-out validation independently of the training script so the result wasn't self-reported by the same code path that produced it.
- Commits or code: `scripts/train_neuroevolution.py` (seed default), `src/controllers/neuro.py` (`BEST_GENOME` replaced).
- Experiment output: `uv run python scripts/train_neuroevolution.py` (unscaled budget, regressed) and `uv run python scripts/train_neuroevolution.py --population 20 --generations 15` (scaled budget, converged); held-out validation via `run_headless_head_to_head` across `(42, 110, 271, 997, 2027)`, 5 races/seed; confirmation run via `uv run python scripts/evaluate_controller.py --module controllers.neuro --seed 42 --races 2 --round-seconds 30`.
- Leaderboard result: not applicable — local evaluation only.

**What we observed:**
- Widening the training objective (more seeds) and widening the search's capacity to solve it (more population/generations) are not independent levers — increasing seed count alone, at Entry 2's original budget, made the outcome strictly worse (8.8 m vs. attempt 4's already-poor 53 m), reproducing the same kind of collapse for a harder reason.
- Once budget was scaled to match, the held-out result was the best of any neuroevolution attempt so far: **24/25 survived, 25/25 completed ≥1 lap, avg scored distance 356.0 m, max speed 28.9 m/s** — compared to attempt 3's 8/25 survived, 21/25 laps, ~187–288 m avg.
- The one held-out failure (seed 997, race 4: eliminated, damage 1.00) shows the fix is not complete — the fitness function still scores a genome by the *mean* across its five training seeds, so a genome that is excellent on four and fails on one can still average well. This matches the exact risk discussed before running the experiment.
- This result closes most, but not all, of the gap to Approach 1 (25/25 survived, 435 m) while keeping neuroevolution's ~1.8× top-speed advantage (28.9 m/s vs. 15.7 m/s), materially changing the recommendation in `EXPLORATION.md`'s comparison section from "Approach 1 dominates" to "Approach 1 is safer, Approach 2 is a live candidate."

**Decision and rationale:**
Replaced attempt 3's genome with this one as `BEST_GENOME` in `src/controllers/neuro.py`, and updated `EXPLORATION.md`'s Approach 2 section and comparison table to reflect the new numbers. This is a clear improvement on every held-out metric simultaneously (survival, laps, and distance all rose; speed did not have to be sacrificed to get there), unlike prior attempts which always traded one property for another.

**Next steps:**
- The remaining single held-out elimination suggests the mean-across-seeds fitness is still the limiting factor, not search budget — try scoring by the worst-of-seeds result (or a mean-minus-spread penalty) instead of the plain mean, now that the budget is large enough to make that harder objective tractable.
- Consider increasing training seeds beyond five and/or adding more elites, now that a population-20/generation-15 run completes in well under 15 minutes.
- Re-run Approach 1 vs. Approach 2 selection reasoning now that the gap has narrowed substantially — this may no longer be a clear-cut "pick Approach 1" decision by the time of the "Select and Refine" stage.

## Entry 6 — Reactive Controller: Proactive Corner Speed Control and Quadratic Braking (Manual A/B)

**Date and time:** August 31, 2026 (continued)

**Participants and contributions:**
- Leon Aaron — directed the experiment loop (inspect → modify → optimize → validate → diagnose), specified the train/validation seed split, and required one-variable-at-a-time testing before trusting any automated search result.
- AI coding agent (Claude Code) — added the new tunable mechanisms, ran every isolated A/B test, diagnosed the counterintuitive results, and added parameter bounds plus a two-phase search to `scripts/optimize_reactive.py`.

**Question or objective:**
Two hypotheses to test in isolation before touching anything else: (1) does proactively reducing target speed for an anticipated turn (`heading_error_degrees`) or an actual one already underway (IMU `yaw_rate_degrees_per_s`) let the car go faster overall by avoiding hard reactive braking later? (2) does a speed-squared braking-distance term (matching true constant-deceleration stopping distance, instead of the current linear-in-speed formula) improve safety margin at high speed?

**What we investigated or changed:**
- Redefined the seed split per instruction: training `(13, 55, 110, 271, 997)`, validation `(42, 2027, 8675, 31415, 777001)` — the last three are seeds never used anywhere in this project before, for a genuinely unseen check.
- Added two new, independently-toggleable mechanisms to `ReactiveParams`/`_throttle_command` in `src/controllers/reactive.py`, both defaulting to zero gain (no effect) so each could be A/B tested without disturbing the existing Entry 3 baseline: `corner_speed_gain` (+ `corner_signal_deg`, `corner_yaw_rate_deg_per_s`) for proactive slowdown, and `brake_quadratic_coeff` added to the existing linear brake-distance formula.
- Confirmed the refactor was behavior-preserving at gain=0: re-ran the training-seed suite and reproduced the exact prior baseline, 432.5 m avg, 25/25 survived (one pre-existing 0.03-damage graze on seed 13's 5th starting position, noted but not chased further — trivial and non-blocking).
- Tested each mechanism alone at the existing speed (`max_speed_mps=15.8`): `corner_speed_gain=0.3` → **260.7 m** (−40%); `brake_quadratic_coeff=0.02` → **353.8 m** (−18%). Both pure regressions, zero damage in both cases.
- Tested `max_speed_mps=19` alone, and combined with each mechanism, to see whether either compensates for a higher speed target rather than just adding caution to an already-safe baseline: alone → **395.1 m** (small damage on seed 13); + `corner_speed_gain=0.3` → **295.0 m**; + `brake_quadratic_coeff=0.03` → **325.4 m**. All three are worse than the 432.5 m baseline.

**Evidence:**
- AI-agent assistance: Claude Code implemented both mechanisms behind a zero-gain default specifically so a byte-for-byte baseline reproduction check was possible before any A/B test, then ran all five isolated trials via `scripts/try_params.py --seeds 13 55 110 271 997 --<flag> <value>`.
- Commits or code: `src/controllers/reactive.py` (new tunable throttle mechanisms), `scripts/try_params.py` (`--seeds` override added), `scripts/optimize_reactive.py` (`PARAM_BOUNDS` + clamped mutation + two-phase search support).
- Experiment output: six `scripts/try_params.py` runs (25 races each) as listed above.
- Leaderboard result: not applicable.

**What we observed:**
- Both new mechanisms are pure regressions in isolation, because the baseline already achieves 0.00 damage everywhere — there is no existing safety gap for extra caution to fill, so any additional slowdown or braking margin only costs distance.
- Raising `max_speed_mps` alone is *also* a regression (395 m vs. 432.5 m), which is counterintuitive until traced: the brake-distance formula scales with the car's *actual* current speed, so a car that actually reaches a higher top speed on the straight also demands a larger stopping distance before the next corner at the *same* wall-lidar reading — triggering earlier, harder braking that costs more than the higher top speed gains. This is a real parameter interaction, not a bug.
- Adding either safety mechanism on top of the higher speed did not fix this — both remained worse than the untouched 15.8 m/s baseline. This confirms Entry 3/4's parameters are a genuinely co-tuned joint optimum: every single-variable change tested made things worse, which is exactly the signature of a well-balanced local optimum rather than a poorly-tuned one.

**Decision and rationale:**
Concluded that manual single-variable A/B testing has reached its limit here — the remaining potential improvement, if any, requires jointly re-tuning multiple parameters together (e.g. speed *and* brake formula *and* apex bias simultaneously), which is exactly the automated search's job, not a human's, once bounds exist to keep it from wasting evaluations. Did not accept any manual change; `DEFAULT_PARAMS` remains Entry 3's validated 432.5–435 m baseline pending the search result below.

**Next steps:**
- Run the newly bounded, two-phase (broad exploration then fine-tuning) search on all five training seeds at once — directly targeting the cross-seed generalization failure from Entry 4, which only ever trained on two.
- Validate any search result against the full, disjoint validation set `(42, 2027, 8675, 31415, 777001)` before accepting it, exactly as required this time.
- If the search cannot beat 432.5 m either, that is itself a meaningful, reportable conclusion: this parameterization of the reactive controller may be near its practical ceiling on this track without a further architectural change (e.g. the anticipatory lookahead-based apex bias from Entry 4, tested in isolation this time rather than simultaneously with a speed increase).

## Entry 7 — Neuroevolution: Wider Population Closes the Last Held-Out Elimination

**Date and time:** August 31, 2026 (continued)

**Participants and contributions:**
- Jayden Webb — directed the experiment (widen population/generations rather than change the fitness function again), and set the rule for this entry: only update `BEST_GENOME` and this notebook if the result beats Entry 5, otherwise keep the shipped genome and just report back.
- AI coding agent (Claude Code) — ran the training pass, validated it independently on the held-out suite, and (separately) tested a same-budget sensor-encoding change that regressed and was reverted before this attempt.

**Question or objective:**
Entry 5's best fitness (235.8 m) was flat across its final 5 generations — a sign the search had exhausted what a population of 20 could find, not necessarily what the objective allows. Would widening the population (20→40) and elite count (2→3), with everything else unchanged, find a genome that beats Entry 5 on both distance and the one remaining held-out elimination (seed 997)?

**What we investigated or changed:**
- Ran `scripts/train_neuroevolution.py --population 40 --generations 15 --elite 3` on the unchanged Entry 5 fitness function and seeds `(13, 55, 7, 89, 233)`. Best training fitness reached 249.8 m, still rising through generation 11 before leveling off — unlike Entry 5's harder plateau.
- Validated the result independently against the full held-out suite `(42, 110, 271, 997, 2027)`, 5 races/seed, before touching `BEST_GENOME`.
- (Separately, in between Entry 5 and this run: tried raising `SPEED_CAP_MPS` from 15 to 35 in `neuro.py`'s input normalization, at Entry 5's original population/generations budget, hypothesizing the network was "blind" to its own speed above 15 m/s. This regressed to 282.2 m avg. and was reverted — evidence that a harder-to-use, less-saturated input signal needs more search budget than a coarser one, not less, consistent with this entry's population-budget finding.)

**Evidence:**
- Experiment output: `uv run python scripts/train_neuroevolution.py --population 40 --generations 15 --elite 3`; held-out validation via `run_headless_head_to_head`, same protocol as Entry 5; confirmation via `uv run python scripts/evaluate_controller.py --module controllers.neuro --seed 42 --races 2 --round-seconds 30`.
- Commits or code: `src/controllers/neuro.py` (`BEST_GENOME` replaced).

**What we observed:**
Held-out result beat Entry 5 on every metric: **25/25 survived** (up from 24/25 — closes the seed-997 elimination), **25/25 laps**, avg scored distance **371.1 m** (up from 356.0 m), max speed 28.1 m/s. No tradeoff this time — unlike every earlier attempt, nothing got worse to get this improvement.

**Decision and rationale:**
Beats Entry 5 outright, so per the standing rule: replaced Entry 5's genome with this one as `BEST_GENOME` in `src/controllers/neuro.py`, and updated `EXPLORATION.md`'s Approach 2 section and comparison table accordingly.

**Next steps:**
- The mean-across-seeds fitness function is still unchanged; this result reached 25/25 without that fix, so the worst-case/spread-penalty idea from Entry 5 is no longer urgent but remains untested as a possible further gain.
- Population 40 took noticeably longer than population 20 (roughly double, as expected); a next attempt should check whether the plateau pattern repeats before investing in an even larger population.

## Entry 8 — reactive2.py: New Sensor Mechanisms, Worst-Seed Search, Rejected on Validation

**Date and time:** September 2, 2026

**Participants and contributions:**
- Leon Aaron — set the three-step gated protocol (build mechanisms → search → validate on held-out seeds), decided to descope other-robot lidar/competitor-camera as unmeasurable in this solo-training harness, and pushed back mid-run to question whether a stalled search was actually a machine-sleep artifact or an unparallelized algorithm — leading to the multiprocessing fix below.
- AI coding agent (Claude Code) — designed and implemented `src/controllers/reactive2.py`'s new mechanisms, built `scripts/optimize_reactive2.py`'s worst-seed-aware search, diagnosed and fixed a hung search process, added multiprocessing, and ran the final validation.

**Question or objective:**
Can genuinely new sensor signals — ones `reactive.py` never reads (forward/lateral acceleration, the raw ±45° wall-lidar beams, all three lookahead points, `contact.damage`, roll/pitch, multi-beam braking) — break the ~442-445 m/30 s plateau documented in Entries 3 and 6, without reintroducing Entry 4's confounded-variables mistake or Entry 5's mean-only-fitness blind spot?

**What we investigated or changed:**
- **Step 1 (mechanisms):** Added 8 independently-gated mechanisms (A–I, with A later removed) to `reactive2.py`, each defaulting to a neutral gain so `gain=0` reproduces `reactive.py` bit-for-bit. Mechanism A (a lateral-acceleration cornering governor) was removed after isolated testing showed `imu.lateral_acceleration_mps2` is exactly `speed_mps * radians(yaw_rate_degrees_per_s)` — a deterministic restatement of yaw rate, not an independent grip measurement — and regressed distance 17–46% at every setting tried. Confirmed the remaining mechanisms wire up correctly (diagonal beams read genuinely different distances than the ±90° beams; the curvature signal swings sign on real corners) and that gain=0 reproduces `reactive.py` exactly.
- **Step 2 (search):** Added 16 new `PARAM_BOUNDS` and a worst-seed-aware fitness, `fitness = mean - 0.5*(mean - min)` over the 5 training seeds `(13, 55, 110, 271, 997)`, directly targeting Entry 5's "mean masks a bad seed" concern. Ran a two-phase evolution strategy — 18 broad-exploration generations (σ=0.25) then 10 fine-tuning generations (σ=0.08), population 20 — at 20 s training rounds.
- **Mid-run infrastructure problem:** the first full-search attempt appeared to hang at phase-2 generation 7 for 47 minutes with no log progress. CPU sampling showed the process burning essentially zero CPU during that window (0.34 CPU-seconds over a 5-second sample) — not "slow," genuinely stalled, most likely from a machine sleep/wake cycle during the run. Separately, and regardless of the hang's cause, `run_search`'s population evaluation was a plain sequential Python loop with no use of the machine's 14 logical cores. Killed the stuck run and added `concurrent.futures.ProcessPoolExecutor`-based parallelism across the population (`scripts/optimize_reactive2.py`, `_evaluate_genome` + `--workers` flag, default one persistent pool for the whole run rather than one per generation, reused across both search phases). A smoke test confirmed identical (0-damage) results to the sequential version. The full 28-generation search then completed in well under the original ~35–45 minute estimate, using `--workers -1` (all logical cores).
- **Step 2 result:** best training fitness 310.0 m (blended), up from the unmutated baseline's 291.8 m under the same blend — all 5 training seeds survived with 0.00 damage, distances 305.9–323.8 m at 20 s rounds. Passed the Step 2 gate.
- **Step 3 (validation):** wrote `scripts/validate_reactive2_candidate.py` to run the candidate on the 5 disjoint validation seeds, 5 races each at 30 s rounds (25 races total) — the same protocol as Entry 6. First pass of the script only read `result.races[0]`, silently scoring 5 races instead of 25 (`run_headless_head_to_head`'s `race_count` returns one entry per race in `result.races`, not one aggregate entry) — caught and fixed before trusting the result.

**Evidence:**
- AI-agent assistance: Claude Code designed all 8 mechanisms and their bounds, diagnosed the lateral-acceleration redundancy, diagnosed the stalled search via direct CPU sampling rather than assumption, implemented the multiprocessing fix, and caught its own validation-script bug before reporting results.
- Commits or code: `src/controllers/reactive2.py` (new mechanisms), `scripts/optimize_reactive2.py` (bounds, worst-seed fitness, multiprocessing), `scripts/validate_reactive2_candidate.py` (new, Step 3 validation).
- Experiment output: `step2_search.log` (full 28-generation search transcript) and the 25-race validation run (below).

**What we observed:**
- Validation, 25 races across 5 held-out seeds at 30 s rounds:

  | seed | distances (m) | damages |
  |---|---|---|
  | 42 | 458.3, 435.5, 487.2, 485.1, 461.9 | 0.00 ×5 |
  | 2027 | 379.2, 451.7, 447.4, 474.3, 474.1 | 0.00 ×5 |
  | 8675 | 487.2, 410.4, 465.8, 439.3, 488.3 | 0.00, **0.25**, 0.00, 0.00, 0.00 |
  | 31415 | 429.5, 458.4, 478.3, 393.9, 450.4 | **0.01**, 0.00, 0.00, **0.03**, 0.00 |
  | 777001 | 459.3, 461.4, 458.2, 487.7, 430.3 | 0.00, **0.01**, 0.00, 0.00, **0.01** |

  Mean distance **454.1 m** — a real ~2–3% gain over the 442–445 m baseline, with zero eliminations. But non-zero (non-fatal) damage occurred in 5 of the 25 races, including one 0.25-damage graze, on seeds the search never trained on.
- This is a genuine generalization gap, not a training-side bug: the worst-seed-aware fitness did its job on the 5 seeds it could see (0.00 damage, every training seed) but the held-out seeds exposed corner geometry or wall configurations the candidate hadn't been pressured against. The new mechanisms' final tuned gains were mostly tiny (`traction_loss_gain≈0.017`, `diagonal_avoid_gain≈0.006`, `curvature_gain≈0.018`, `instability_gain≈1.5e-05`) — the search found modest use for them but didn't lean on any single one heavily, consistent with the previously-documented plateau being a real, hard-to-move local optimum rather than one hiding an easy win.

**Decision and rationale:**
**Rejected per the pre-committed Step 3 gate** (0 damage in all 25 validation races AND mean distance beats baseline — both required, not either). Distance alone would have passed; damage discipline did not generalize to unseen seeds. Per Entry 6's precedent, this is reported as a genuine negative result rather than relaxed after the fact: `reactive2.py`'s `DEFAULT_PARAMS` is **not** updated, and remains a byte-for-byte copy of `reactive.py`'s validated 442–445 m baseline.

**Next steps:**
- The new sensor mechanisms (B–I) are in place and confirmed non-regressive at gain=0, so they remain available for a future search attempt without redoing Step 1.
- A future search could tighten the worst-seed penalty weight (`WORST_SEED_PENALTY_WEIGHT`, currently 0.5) toward 1.0, or fold 2–3 of the validation seeds into training (shrinking the held-out set) so the fitness function is pressured by more corner configurations before validation — at the cost of a smaller, less independent final check.
- The `--workers` multiprocessing fix (`ProcessPoolExecutor` across the population, `--workers -1` to use all logical cores) cut this search's wall-clock time from an estimated ~35–45 minutes to well under that on a 14-core machine; it carries forward to any future `optimize_reactive2.py` run regardless of whether this particular mechanism set is revisited, and should be the default going forward rather than the sequential path.
- A structurally different next attempt worth considering: train a small PyTorch model (e.g. a compact MLP policy, in the spirit of Entry 2's neuroevolution strategy but gradient-trained via imitation or RL rather than evolution-searched) directly on the full sensor vector, rather than continuing to hand-design more gated mechanisms on top of a fixed reactive control law. This could let the model discover nonlinear sensor combinations (e.g. a learned fusion of the corner-severity signals that Entry 7 deliberately kept hand-separated) that a human-authored formula is unlikely to find, at the cost of losing the interpretability that made every entry in this notebook diagnosable.

## Entry 9 — reactive2.py Re-Search with a Safety-Margin Floor: Damage Fixed, Gain Erased

**Date and time:** September 2, 2026 (continued)

**Participants and contributions:**
- Leon Aaron — asked directly whether Entry 7's rejected candidate's distance gain could be salvaged rather than discarded outright, prompting the root-cause check below instead of simply re-running the same search.
- AI coding agent (Claude Code) — diagnosed the specific parameter responsible for Entry 7's validation damage, tightened its search bound, re-ran the full search and validation, and reported the (negative) result rather than the hoped-for one.

**Question or objective:**
Entry 7's rejected candidate had a real ~2-3% distance gain (454.1 m mean) undercut by non-fatal damage on 5/25 held-out races. Was that damage caused by the new sensor mechanisms themselves misbehaving on unfamiliar corners, or by the search exploiting slack in an existing, already-validated safety parameter? If the latter, constraining that parameter should let the search keep whatever real gain the new mechanisms offer while eliminating the damage.

**What we investigated or changed:**
- Compared Entry 7's rejected genome against `reactive2.py`'s safe `DEFAULT_PARAMS`: `wall_avoid_margin_m` had been shaved from a proven-safe 0.998 m to 0.544 m (nearly halved), and the new `diagonal_avoid_margin_m` — inert at baseline — had been left at a thin 0.541 m at a nonzero gain. Both are core wall-avoidance buffers, not among the new mechanisms being evaluated.
- Raised `PARAM_BOUNDS` floors in `scripts/optimize_reactive2.py`: `wall_avoid_margin_m` from (0.5, 6.5) to (0.998, 6.5) — pinned to the proven-safe default — and `diagonal_avoid_margin_m` from (0.5, 8.0) to (1.5, 8.0), a deliberately conservative floor since that mechanism had no prior safe baseline to pin to.
- Re-ran the identical two-phase search (population 20, 18+10 generations, same 5 training seeds, `--workers -1`) and re-validated on the same 25-race held-out suite via `scripts/validate_reactive2_candidate.py`.

**Evidence:**
- `step2_search_v2.log` (full re-run transcript) and the corresponding 25-race validation output.
- Commits or code: `scripts/optimize_reactive2.py` (`PARAM_BOUNDS` floor changes, with inline comments citing this diagnosis), `scripts/validate_reactive2_candidate.py` (`CANDIDATE` updated to the new genome).

**What we observed:**
- Training: best fitness 297.5 m (down from Entry 7's 310.0 m, as expected — less room to exploit the margin), 0.00 damage on all 5 training seeds. `wall_avoid_margin_m` converged to 1.125 m (above the new floor, so the search found a genuinely better value rather than just hitting the wall); `diagonal_avoid_margin_m` converged exactly to its new floor of 1.5 m, meaning the search still wants it lower and is now capped.
- Validation, 25 races across the same 5 held-out seeds at 30 s rounds: **zero damage in every single race** (max damage 0.00, versus Entry 7's max 0.25) — the fix fully closed the generalization gap. But mean distance came back to **442.2 m**, statistically indistinguishable from the 442-445 m baseline it needs to strictly exceed.
- Interpretation: Entry 7's apparent "gain" was almost entirely the margin-shaving exploit, not the new sensor mechanisms. With that exploit closed off, the new mechanisms (B-I) on their own contribute no measurable net improvement on this track — their tuned gains stayed small in both search runs (Entry 7 and 8 alike), and removing the one lever that did move the needle returns the result to baseline. This is now a second, independent confirmation of the ~442-445 m plateau, arrived at via a different search under a stricter constraint, not just a repeat of Entry 6's finding.

**Decision and rationale:**
**Rejected again**, this time for the opposite reason from Entry 7: safety generalizes perfectly, but there is no distance gain left to accept. `reactive2.py`'s `DEFAULT_PARAMS` remains unchanged. This is a cleaner negative result than Entry 7's: it isolates that the plateau is real and that reactive2.py's new mechanisms, evaluated honestly (without borrowing margin from an already-tuned safety parameter), do not move it — closing off the most obvious "maybe we can just fix the one bad parameter" follow-up before it gets tried again.

**Next steps:**
- Do not re-attempt a third search with the same mechanism set and the same margin floors expecting a different outcome — two independent runs now agree. Any further reactive-controller work here should target a genuinely different lever (e.g. the apex-bias/lookahead geometry, last touched in Entry 3/4) rather than another blind evolutionary pass over the same parameter set.
- The PyTorch-model direction from Entry 7 remains the most promising path for a real step-change, since it does not depend on a human first guessing which hand-designed mechanism might help.
- Keep the `--workers` multiprocessing default for any future `optimize_reactive2.py` run — both re-runs in this investigation benefited from it, and it is what made a same-day two-attempt turnaround (Entry 7 then Entry 8) practical at all.

## Entry 10 — Learned Controller: Behavior Cloning + Dense-Reward CMA-ES (Torch)

**Date and time:** September 8, 2026

**Question or objective:** Entry 3's reactive controller plateaued at ~435 m/30 s with no cornering-speed knob left to tune. Can a torch MLP, warm-started by cloning the reactive controller (avoiding Entry 2's random-init "idle"/"reckless" local optima) and then fine-tuned with real covariance-adaptation CMA-ES against a *dense* per-tick reward (instead of Entry 2's single end-of-episode number), beat 435 m and move toward a 4-lap (~724 m) stretch goal?

**What we investigated or changed:**
- Added `torch` and `cma` dependencies. Discovered and fixed a Windows-only issue: importing `racing` (Panda3D) before `torch` breaks torch's DLL init (`WinError 1114`); fixed by opportunistically importing torch first at the top of `src/racing/__init__.py`.
- `scripts/generate_expert_demonstrations.py`: logged the reactive controller's (sensors, action) pairs across 40 seeds (~48k ticks) using a 13-feature encoder (`controllers/neuro.py`'s 10 features plus the 3 `lookahead_offsets_m`, since lookahead is what enables reactive's apex-hugging).
- `scripts/train_behavior_cloning.py`: supervised MSE training of a small MLP (13→8→2 tanh, ~140 parameters) to imitate the reactive controller. Held-out validation (25 races, seeds 42/110/271/997/2027): **25/25 survived, 25/25 completed 2 laps, avg. scored distance ≈444 m** — a small, safe improvement over the 435 m reactive baseline, essentially by successfully cloning it.
- `scripts/train_learned_controller.py`: CMA-ES fine-tune seeded from the BC weights, dense reward = progress-aligned speed (`speed_mps * cos(heading_error)`, matching the centerline-projection scoring rule) minus a damage-rate penalty minus a small idle penalty, trained on 4 seeds simultaneously, in 5-minute wall-clock chunks (checkpointing every generation, never saving a regression versus the loaded init).
- Ran three 5-minute chunks (first with a reward-shaping bug using raw chassis speed instead of progress-aligned speed, corrected for chunks 2-3). Chunk 1 (buggy reward) found a genome reaching **27-30 m/s top speed (~2x reactive's 15.7 m/s)**, but held-out validation (25 races) averaged only ~433 m — about the same as the BC baseline, with one race taking minor damage. Chunks 2-3 (corrected reward, sigma0 0.4 then 0.08) found no generation that beat the BC checkpoint's own training reward in 4-5 generations each.

**What we observed:** CMA-ES over a 130-parameter net, with population ~18 and only 4-5 generations per 5-minute chunk, could reach a much higher top speed than the reactive controller but could not convert that speed into more net scored distance within this budget — the BC-cloned starting point (itself already evolution-strategy-optimized in Entry 3) is a sharp local optimum that a handful of generations of population-18 mutation could not clearly escape. This mirrors Entry 3's own note that further reactive gains would need a genuinely different mechanism (cornering-speed control), not just more search around the same optimum.

**Decision and rationale:** Adopted the **behavior-cloned policy** (no CMA-ES fine-tuning) as `src/controllers/learned.py`'s shipped `learned_policy.pt` — it is the only learned-controller variant that clearly and safely beat the reactive baseline (444 m vs. 435 m, 25/25 safe) in this session's time budget. The CMA-ES fine-tuned variants are documented here as informative negative results, not shipped.

**Next steps:**
- Re-run `scripts/train_learned_controller.py` for many more 5-minute chunks (the corrected reward is in place); consider a larger population or a curriculum that increases `--round-seconds` once training reward plateaus, so the search has more chances to discover genuinely faster-but-still-safe genomes rather than only local tweaks.
- The demonstrated 27-30 m/s speed ceiling (chunk 1) suggests real headroom exists; reaching it safely likely needs either substantially more CMA-ES budget, or an explicit cornering-speed feature/output the reactive controller does not have.
- 4 laps (~724 m in 30 s) requires ~24 m/s sustained average - between the reactive/BC ceiling (15.7 m/s) and the demonstrated-but-unsafe neuro/CMA-ES ceiling (27-35 m/s). Treat it as a stretch goal for future sessions with a larger training budget, not a one-session deliverable.

## Entry 11 — Compute an Optimal Racing Line, Then Distill It Into a Sensor-Only Policy

**Date and time:** September 8, 2026 (continued)

**Question or objective:** Entry 10's gains were marginal because behavior-cloning the reactive controller just re-learns the same apex-bias heuristic. Can we instead (1) compute an actual minimum-curvature racing line for the track offline, (2) build a privileged expert that tracks that specific line using ground-truth arc-length position, and (3) distill *that* expert into the same sensor-only network, so training labels come from a genuinely optimized reference path rather than a hand-tuned heuristic?

**What we investigated or changed:**
- `scripts/compute_racing_line.py`: an offline geometric optimizer. Represents the path as a lateral offset from centerline at the track model's 201 arc-length samples, then iteratively pulls each point toward the midpoint of its neighbors (projected onto the local track normal, clamped to the track corridor) — a standard "iterative corner-cutting" curvature-minimization smoother. Also computes a curvature-limited speed profile (lateral-grip cap, then forward/backward accel-limited passes).
- `scripts/_privileged_race_env.py`: a custom single-car headless stepping loop (reusing the same internal building blocks as `racing.race.head_to_head`) that exposes the *true* arc-length position (`TrackProjection.progress_distance_m`) each tick — something no public `RobotSensors` field provides. Used only for offline expert generation, never by the deployed controller.
- `scripts/racing_line_follower.py`: a privileged expert reusing `controllers.reactive`'s steering/safety structure (lookahead terms, wall avoidance, emergency steer, contact recovery), but replacing the heuristic apex-bias offset and flat max-speed target with the precomputed line's offset(s) and speed(s).
- **First finding (a real negative result):** the curvature-limited speed profile made things *worse* — 250-355 m/30s depending on grip constants, well below the 435 m reactive baseline, despite reaching much higher peak speed. Root cause: this simulator's grip is high enough that Entry 3 already found flat-out driving (no cornering slowdown) beats any curvature-based slowdown; a "physically principled" speed profile reintroduces exactly the cornering deceleration Entry 3 removed. Swept flat speed targets (15.8-30 m/s) against the *optimized offset* line instead: 15.8 m/s (matching reactive's own tuned speed) gave the best, fully safe result.
- Regenerated the racing line with `--flat-speed-mps 15.815503530875` (reactive's exact tuned speed) so only the offset geometry differs from reactive, isolating the effect of the optimized line. Held-out validation of the raw (non-distilled) follower: **452 m avg, 5/5 safe** — a genuine, apples-to-apples +3.9% over reactive's 435 m from the offset geometry alone.
- `scripts/generate_racing_line_demonstrations.py` + `scripts/train_behavior_cloning.py`: distilled this expert into the same sensor-only network architecture as Entry 10 (never given the racing line or ground-truth position as input, only `controllers.learned.build_inputs`'s public sensor features). Training MSE converged far lower (0.00055) than Entry 10's reactive-clone (0.0167), reflecting a cleaner, more consistent expert.

**What we observed:** Held-out validation (25 races) of the distilled network: **25/25 survived, 25/25 completed 2 laps, avg. scored distance ≈442.8 m** — essentially tied with Entry 10's reactive-clone (444.3 m), not a clear win, even though the raw undistilled expert (452 m) was better than reactive. Distillation lost roughly the same margin it gained over reactive. Both distilled policies land in the same ~440-450 m / 2-lap band; neither approach has yet found a way to convert a better *reference line* into a proportionally better *learned* policy.

**Decision and rationale:** Shipped the racing-line-distilled policy (`artifacts/racing_line_bc_policy.pt` → `src/controllers/learned_policy.pt`) rather than Entry 10's reactive-clone, since it directly implements the requested approach (compute the optimal line, train a controller to follow it, distill into a sensor-generalizing network) and is not worse. Both the geometric line-optimization code and the privileged single-car environment are reusable for future work regardless of which distilled checkpoint ships.

**Next steps:**
- The gap between the raw expert (452 m) and its distillation (442.8 m) suggests classic behavior-cloning covariate shift: the network never sees the small-error recovery states its own imperfect tracking creates during autonomous rollout, only the expert's near-perfect trajectory. A DAgger-style pass (re-label the *learned* policy's own visited states with the expert's action, then retrain) is the standard fix and wasn't attempted this session.
- The offline line optimizer is still track-specific (computed from this track's exact centerline); true cross-track generalization claims should be tested by computing a second line for a different track layout (if/when one exists) and checking whether the *same* trained network - not retrained - still drives it competently.
- 4 laps remains unreached; the biggest lever demonstrated so far is still Entry 10's CMA-ES speed ceiling (27-30 m/s) - combining that with this entry's better offset geometry (rather than reactive's heuristic apex bias) is the most promising untried combination.

## Entry 12 — Raising the Racing-Line Follower's Flat Speed: a Damage-Margin Sweep, Not a Blind Hillclimb

**Date and time:** September 8, 2026 (continued)

**Question or objective:** Entry 11 shipped the racing-line-distilled policy at `flat_speed_mps = 15.815503530875` (reactive's own tuned speed, chosen only to isolate the effect of the offset geometry). Since `RacingLineFollowerParams.flat_speed_mps` is a free parameter never actually searched, can raising it push scored distance further while keeping the same zero-damage safety margin the reactive and BC controllers have shown throughout this notebook?

**What we investigated or changed:**
- Found `scripts/optimize_racing_line_follower.py` (an evolution-strategy search over `RacingLineFollowerParams`, symmetric to `scripts/optimize_reactive.py`) and `artifacts/racing_line_bc_policy_v2.pt` already present from an earlier unlogged pass in this session: the search had pushed `flat_speed_mps` from 15.8 to 23.09 m/s and a BC policy had been distilled from it, but never validated against the held-out seed suite.
- Validated `racing_line_bc_policy_v2.pt` (25 races, held-out seeds): **only 7/25 survived, avg scored distance 331.5 m** — worse than the shipped 442.8 m baseline. A genuine negative result: the fitness-guided search found a *training* optimum that does not generalize.
- Diagnosed why, using the privileged (ground-truth-position) follower directly, bypassing distillation: at `flat_speed_mps = 23.09`, the raw expert itself is not damage-free — **20/20 survived but with damage 0.37–0.67 per race** (avg 501.9 m), a materially different safety margin than reactive/BC's demonstrated 0.00. The search's fitness function (distance minus an elimination penalty) rewards surviving-with-damage the same as surviving cleanly, so it happily traded away the safety margin that made the small-network distillation step generalize in Entries 3, 10, and 11. Under an imperfect sensor-only clone, that margin evaporates and pushes many races over the 1.0 elimination threshold.
- Swept `flat_speed_mps` directly against the raw privileged follower (20 seeds) instead of trusting the search's fitness number: distance rises smoothly and damage stays exactly 0.00 up to **17.0 m/s** (517–519 m every seed), then damage appears abruptly at 17.5 m/s (0.03–0.41) with distance *dropping* (wall contact drag), confirming a sharp, measurable zero-damage ceiling rather than a smooth tradeoff.
- Set `RacingLineFollowerParams.flat_speed_mps = 17.0` (replacing the abandoned 23.09 default), regenerated demonstrations (`scripts/generate_racing_line_demonstrations.py`, same 40-seed/48k-tick recipe as Entry 11) and retrained the BC network (`scripts/train_behavior_cloning.py`) — training MSE converged to 0.00021, even lower than Entry 11's 0.00055.

**Evidence:**
- Sources or documentation: none new; built on Entry 11's pipeline.
- AI-agent assistance: Claude Code found and validated the pre-existing but unshipped v2 experiment, diagnosed its generalization failure via the privileged follower (isolating speed-induced damage from distillation error), ran the flat-speed sweep, and re-ran the full generate-demonstrations -> train-BC -> validate pipeline at the measured safe ceiling.
- Commits or code: `scripts/racing_line_follower.py` (`flat_speed_mps` default 23.09 -> 17.0, with the measured rationale recorded in a comment), `artifacts/racing_line_demonstrations_v3.pt`, `artifacts/racing_line_bc_policy_v3.pt` -> `src/controllers/learned_policy.pt`.
- Experiment output: held-out validation (25 races, `uv run python scripts/evaluate_controller.py --module controllers.learned --seed <seed> --races 5 --round-seconds 30` across `(42, 110, 271, 997, 2027)`).
- Leaderboard result: not applicable — local evaluation only.

**What we observed:**
- Held-out validation of the new shipped policy: **25/25 survived, 25/25 completed 2 laps, damage = 0.00 in 24/25 races (one race at 0.04)**, average scored distance **≈511 m per 30 s** — a **+15.4% improvement over Entry 11's shipped 442.8 m**, and +17.5% over the original reactive baseline (435 m), with the same safety record.
- The key lesson: an evolution-strategy search optimizing raw distance-with-elimination-penalty is not the same objective as "distance at zero damage" — it will spend safety margin for speed whenever the elimination penalty is smaller than the expected gain, and that traded-away margin is specifically what BC distillation needs to survive its own approximation error. Measuring the *raw, privileged* controller's damage directly (not just survival) caught this before it was baked into another distillation cycle.

**Decision and rationale:** Shipped the re-distilled policy (`racing_line_bc_policy_v3.pt` -> `src/controllers/learned_policy.pt`) as the new `learned_policy.pt`, chosen by measuring the actual zero-damage speed ceiling rather than trusting an unvalidated search result. This is now the controller to beat: ≈511 m/30s, 25/25 safe.

**Next steps:**
- The 17.0 -> 17.5 m/s cliff is measured on 10-20 seeds only; a wider seed sweep (or a search that directly penalizes any nonzero damage, not just elimination) could locate a tighter, more confident ceiling, or reveal it varies slightly by starting position.
- The abandoned 23 m/s attempt still demonstrates real headroom (501.9 m raw, before distillation loss) if a way is found to make the sensor-only network as damage-tolerant as the privileged expert at that speed — e.g., DAgger (Entry 11's own open item) to teach the network recovery behavior from its own near-wall states rather than only the expert's clean trajectory.
- 4 laps (~724 m in 30 s) is closer than before (511 m now vs. 435-444 m previously) but still requires either the DAgger fix above or Entry 10's higher-speed CMA-ES ceiling combined safely with this entry's line geometry.

## Entry 13 — Damage-Aware Joint Search Over Follower Steering + Speed: +45.9 m, Still Zero Damage

**Date and time:** September 8, 2026 (continued)

**Question or objective:** Entry 12 raised the follower's flat speed to 17.0 m/s by sweeping speed alone against fixed steering gains. That leaves the steering/braking gains (`center_offset_gain`, `wall_avoid_margin_m`, `steer_limit`, etc.) exactly as hand-tuned in Entry 6 for the original 15.8 m/s reactive controller, never revisited for the racing-line follower. Could jointly re-tuning steering *and* speed together, under a fitness that actually penalizes partial damage (not just elimination), find a genuinely faster and still safe combination? Direct motivation: a first, unlogged attempt at this (Entry 12's "abandoned 23 m/s attempt") had cranked speed alone without touching steering and found only unsafe results.

**What we investigated or changed:**
- Diagnosed why `scripts/optimize_racing_line_follower.py`'s prior run (the one rejected in Entry 12) found an unsafe optimum: its fitness (`evaluate_params`) only penalized full elimination, never partial damage - so a genome that grazes a wall but stays under damage 1.0 on its training seeds scored exactly the same as one with zero damage. Added `DAMAGE_PENALTY_M = 600.0`, subtracted proportional to `stats.damage` on every non-eliminated, non-idle race - mirroring the same fix already applied to `scripts/train_learned_controller.py`'s dense reward this session.
- **Round 1:** ran the corrected search (population 10, generations 8, seeds 13/55/87/241, 15 s rounds) seeded from Entry 12's exact 17.0 m/s baseline. Training fitness rose 253.9 m -> 262.4 m; `flat_speed_mps` moved only slightly (17.0 -> 17.2). Validated on 15 fresh seeds at 30 s (privileged, raw follower): **516.9 m -> 533.7 m, zero damage on both, 15/15 survived** - a real but modest gain. Adopted as the new `DEFAULT_FOLLOWER_PARAMS` (all 16 params, not just speed).
- **Round 2:** re-ran the same search seeded from *that* new baseline (population 12, generations 10, seeds 13/55/87, 15 s rounds) to keep pushing. Training fitness rose 263.0 m -> 278.0 m; `flat_speed_mps` jumped further to 18.3 m/s. Validated on 20 fresh seeds: **avg 538.0 m, but 3/20 seeds took real damage (0.39-0.79)** - a faster but not-yet-safe result (17/20 seeds hit ~565 m, the other 3 clipped a wall hard).
- **Round 3 (targeted refinement):** rather than discard round 2's genome, re-ran a short local search (population 10, generations 8, sigma 0.10) seeded from *round 2's own genome*, training specifically on its 3 failing seeds plus 2 known-good ones. Fitness on that hard 5-seed set rose from -18.6 m (round 2's genome scored *negative* here - confirming the failure was real, not sampling noise) to 277.1 m.
- Validated the round-3 genome on 30 fresh seeds never used in any search round: **30/30 survived, zero damage in every single race, avg 564.0 m** (`flat_speed_mps` settled at 18.48 m/s). Adopted as `DEFAULT_FOLLOWER_PARAMS`.
- Regenerated demonstrations (`scripts/generate_racing_line_demonstrations.py`, same 40-seed recipe) and retrained BC (`scripts/train_behavior_cloning.py`) - training MSE converged to 0.00017, the lowest yet.

**Evidence:**
- Sources or documentation: none new; built on Entry 12's search infrastructure and damage-margin lesson.
- AI-agent assistance: Claude Code diagnosed the fitness-function gap that caused Entry 12's rejected 23 m/s attempt, fixed it, ran three successive search rounds (each seeded from the previous validated result, not from scratch), diagnosed round 2's specific failing seeds via direct measurement rather than discarding it outright, and ran the full regenerate-demonstrations -> retrain -> validate pipeline on the final result.
- Commits or code: `scripts/optimize_racing_line_follower.py` (`DAMAGE_PENALTY_M` fix), `scripts/racing_line_follower.py` (`DEFAULT_FOLLOWER_PARAMS` updated twice, final values from round 3), `scripts/train_learned_controller.py` (`DAMAGE_PENALTY_WEIGHT` 200 -> 800, from an earlier CMA-ES attempt this session that also found the fast-but-damaged failure mode and was rejected), `artifacts/racing_line_demonstrations_v4.pt`, `artifacts/racing_line_bc_policy_v4.pt` -> `src/controllers/learned_policy.pt`.
- Experiment output: held-out validation (25 races, `uv run python scripts/evaluate_controller.py --module controllers.learned --seed <seed> --races 5 --round-seconds 30` across `(42, 110, 271, 997, 2027)`).
- Leaderboard result: not applicable - local evaluation only.

**What we observed:**
- Held-out validation of the new shipped policy: **25/25 survived, 25/25 completed >=2 laps (several races reached 3 laps)**, damage = 0.00 in 24/25 races (one race at 0.03), average scored distance **≈556.9 m per 30 s** - **+45.9 m over Entry 12's shipped 511.0 m**, +121.9 m (+28%) over the original reactive baseline (435 m).
- A CMA-ES attempt on the neural network directly (three short chunks, 3-5 minutes each, described in this session's chat but not separately logged as its own entry since nothing shipped from it) reproduced the exact same lesson at the network level: an under-penalized dense reward found a 29 m/s genome that looked good in training but regressed hard on held-out seeds (24/25 survived, damage up to 0.58, avg only 443.3 m - *worse* than baseline). Raising that reward's damage penalty 4x stopped the regression but also stopped finding any improvement in two more 3-minute chunks. This is why the entry above worked on the *follower* level instead: the privileged environment's fast, cheap evaluation (no network forward pass, no full head-to-head harness) allowed three successive rounds of real search-and-diagnose within the session's time budget, where the equivalent CMA-ES-on-the-network approach only had budget for one blind attempt plus two failed conservative retries.
- The round-2 -> round-3 sequence is the clearest evidence in this notebook that a "regression on some held-out seeds" result should be diagnosed and specifically retrained against, not simply discarded - round 2's genome was net faster (538 m avg) but unsafe; identifying and training against its own 3 failing seeds recovered *all* of that speed (564 m, even higher) while eliminating the damage entirely, in a single 8-generation, well under 5-minute pass.

**Decision and rationale:** Shipped the round-3-derived policy (`racing_line_bc_policy_v4.pt` -> `src/controllers/learned_policy.pt`). This is now the controller to beat: ≈556.9 m/30s, 25/25 safe, ≈46 m gained in a single session through iterative damage-aware search rather than a single large blind run.

**Next steps:**
- 30 validation seeds (round 3) and 25 held-out seeds (final network validation) is still a finite sample; the same "looks safe, diagnose failures, retrain against them" loop used for round 2 -> round 3 should be repeated if any future validation run surfaces a new failing seed, rather than assuming 0/30 and 0/25 generalizes to 0/infinity.
- 4 laps (~724 m in 30 s) is now within closer reach: 556.9 m at consistently 2-3 laps is roughly 77% of the way there by distance, versus 511 m (70%) before this entry.
- The same targeted-refinement technique (diagnose specific failing seeds from a faster-but-unsafe candidate, retrain against exactly those) is a reusable pattern for any future speed push on this controller, and is cheaper per iteration than random-restart search since it reuses the fast candidate's already-correct behavior on the seeds it *does* handle well.

## Entry 14 — A Second Search Pass Under a Hard 15-Minute Budget: Diminishing Returns, Still Safe

**Date and time:** September 8, 2026 (continued)

**Question or objective:** Given a hard 15-minute budget (target: push Entry 13's 556.9 m toward 600 m), does repeating Entry 13's damage-aware search process - seeded from its own result - keep finding gains at a similar rate, or is the follower's steering/speed parameterization approaching a genuine ceiling for this racing line's geometry?

**What we investigated or changed:**
- Ran four successive `scripts/optimize_racing_line_follower.py` search rounds, each seeded from `DEFAULT_FOLLOWER_PARAMS`, varying population/generations/mutation-sigma/seed-count to fit short time slices: a narrow-mutation round (sigma 0.2, found +0.4% training fitness, essentially a no-op), a wide-mutation round (sigma 0.35, 12s rounds, found *nothing* better than init over 8 generations), and two moderate rounds (sigma 0.25, matching Entry 13's successful round-2 configuration) - the first found another no-op, the second (different rng seed) found a real +2.6% training gain with a notably different steering strategy (`lookahead_far_gain` 0.28 -> 0.61, `steer_limit` 1.53 -> 1.98, i.e., looking farther ahead with a much less clamped steering response, rather than a pure speed increase).
- Validated that last candidate on 12 fresh seeds at 30 s (privileged, raw follower): **12/12 survived, zero damage, avg 577.8 m** (vs. 564.0 m raw baseline) - safe, so shipped directly without needing Entry 13's targeted-refinement step this time (no failing seeds to fix).
- Regenerated demonstrations and retrained BC on the new params (training MSE 0.00020, in line with prior entries).

**Evidence:**
- Experiment output: four search rounds (`optimize_racing_line_follower.py`, varying flags, all within the session's 15-minute wall-clock budget); 12-seed raw-follower validation; 25-race held-out validation via `scripts/evaluate_controller.py`-equivalent (`validate_policy.py`) across `(42, 110, 271, 997, 2027)`.
- Commits or code: `scripts/racing_line_follower.py` (`DEFAULT_FOLLOWER_PARAMS` updated again), `artifacts/racing_line_demonstrations_v5.pt`, `artifacts/racing_line_bc_policy_v5.pt` -> `src/controllers/learned_policy.pt`.

**What we observed:**
- Held-out validation of the new shipped policy: **25/25 survived, 25/25 completed >=2 laps (most at 3 laps), damage = 0.00 in every single one of the 25 races** (even cleaner than Entry 13's 24/25), average scored distance **≈568.0 m per 30 s** - **+11.1 m over Entry 13's 556.9 m**, but well short of the 600 m target requested for this session.
- Two of the four search rounds this entry found literally nothing better than their seed genome, versus Entry 13 where every round found a usable improvement. This is a measurable slowdown in the rate of return from the same search method on the same line geometry - each successive round is finding a smaller slice of remaining headroom, consistent with approaching (not necessarily at) a real ceiling for this steering-law/line-geometry combination.

**Decision and rationale:** Shipped the round result (`racing_line_bc_policy_v5.pt` -> `src/controllers/learned_policy.pt`) since it is a genuine, fully safe improvement, even though it fell short of the requested 600 m. Did not continue searching with the same method past the 15-minute budget rather than force a result; reporting the shortfall honestly follows the standing practice in this notebook of logging what the time-boxed search actually found rather than only successes.

**Next steps:**
- Reaching 600 m likely needs a different lever than another round of the same steering/speed search: either a genuinely re-optimized racing line (rerun `compute_racing_line.py` with different curvature-smoothing parameters, not just re-tuning the follower's gains around the existing line), or the DAgger fix flagged in Entries 11-12 to let a higher-speed line survive distillation without the current safety-margin tax.
- If more search time becomes available, resume from this entry's exact shipped params (not from scratch) - four rounds this entry never regressed below the loaded baseline, so continuing the same seeded-search process is still the correct default, just slower to pay off now than in Entry 13.

## Entry 15 — Anticipatory Accel/Decel: Works at the Follower Level, Fails to Distill

**Date and time:** September 8, 2026 (continued)

**Question or objective:** Prior corner-speed attempts this session used `heading_error_degrees` (the car's *current* misalignment) as the slowdown signal and found it useless twice - the search always neutralized it, because by the time heading error is large the car is already mid-corner, too late on a track with corners this close together. Would an *anticipatory* signal - the far camera lookahead point (`camera.lookahead_offsets_m[-1]`, 16 m ahead, ~1 s of lead time at these speeds) instead of the current-instant one - give real advance braking distance and let straight-line speed rise well above the ~19 m/s ceiling every prior entry converged to?

**What we investigated or changed:**
- Added `lookahead_speed_gain` to `RacingLineFollowerParams`/`follow()`: `target_speed_mps = max(min_corner_speed_mps, flat_speed_mps - lookahead_speed_gain * abs(far_lookahead_offset_m))`, defaulting to 0 (inert, same as every prior flat-speed entry).
- Search seeded aggressively (`flat_speed_mps=27`, `lookahead_speed_gain=3.0`, `min_corner_speed_mps=14`), population 10, generations 8, 4 training seeds, 15 s rounds (one ~3-minute chunk): fitness rose 217.6 m -> 280.8 m and, unlike every `corner_speed_gain` attempt, the term was **not** neutralized this time - converged to `flat_speed_mps=37.9`, `lookahead_speed_gain=2.97`, `min_corner_speed_mps=21.3`.
- Validated the raw candidate on 15 fresh seeds: **avg 602.0 m, 0/15 eliminated**, but 2 seeds took minor damage (0.14, 0.08) - a genuine, large jump (vs. 578-586 m from every prior entry) but not yet zero-damage.
- Ran a targeted refinement (same technique as Entry 13's round 3) against the two failing seeds plus two known-good ones: training fitness rose 346.4 m -> 363.5 m, converging to a more conservative `flat_speed_mps=31.9`, `lookahead_speed_gain=2.26`. Validated head-to-head against the original candidate on the *same* 20 fresh seeds: refined candidate **546.6 m avg, max damage 0.02** (safer, slower) vs. original candidate **591.8 m avg, max damage 0.17, still 0/20 eliminated** (faster, a bit riskier). Chose the faster one given the session's explicit 600 m target and hard time budget - not a zero-damage result, an explicit speed/safety tradeoff made under time pressure.
- Regenerated demonstrations and retrained BC on the faster candidate. **Training MSE converged to 0.036 - roughly 180x every prior entry's ~0.0002.**

**Evidence:**
- Experiment output: two search rounds (initial + targeted refinement) and three raw-follower validation passes (15, 20, 20 fresh seeds); one full held-out validation of the distilled network via `validate_policy.py` across `(42, 110, 271, 997, 2027)`.
- Commits or code: `scripts/racing_line_follower.py` (`lookahead_speed_gain` mechanism added, left at 0/inert in the shipped `DEFAULT_FOLLOWER_PARAMS` - the working params are recorded here, not in code, since they were not shipped), `artifacts/racing_line_demonstrations_v7.pt`, `artifacts/racing_line_bc_policy_v7.pt` (not shipped).

**What we observed:**
- Held-out validation of the distilled network: **24/25 survived (one full elimination), avg scored distance 484.5 m** - *worse* than the 568.0 m already shipped from Entry 14, despite the raw expert being dramatically better (591.8 m raw vs. Entry 14's ~578 m raw).
- The training-MSE jump (0.0002 -> 0.036) is the direct explanation: every prior expert's throttle command was a near-constant function of the sensors (flat speed target, minor brake-distance reactions), easy for a ~140-parameter, 8-hidden-unit network to imitate almost exactly. This expert's throttle now swings from full acceleration toward 38 m/s down to a much lower cornering speed based on a lookahead signal - a genuinely higher-variance function of the input - and the same small network could not fit it closely enough to reproduce the expert's precise brake-timing, so small execution errors during autonomous rollout compound into real wall contact (exactly the covariate-shift risk flagged as an open item since Entry 11).
- This is a clean, mechanistic explanation for why the same "raise flat speed" lever worked cheaply through Entries 12-14 (near-constant target, low-variance expert, easy to distill) but broke down here (highly-variable target, high-variance expert, hard to distill) - the anticipatory mechanism itself is sound (591.8 m raw, 0 eliminations), but this network's capacity is the actual bottleneck now, not the follower's driving policy.

**Decision and rationale:** Did **not** ship `racing_line_bc_policy_v7.pt` - it is a regression versus the already-shipped Entry 14 policy on every metric that matters (survival and distance both worse). Reverted `scripts/racing_line_follower.py`'s `DEFAULT_FOLLOWER_PARAMS` to exactly match what is shipped (Entry 14's values) so the script and the deployed weights stay consistent; `src/controllers/learned_policy.pt` remains Entry 14's 568.0 m, 25/25-safe checkpoint. The working (raw-follower-only) anticipatory-braking parameters are preserved in this entry's text for reuse.

**Next steps:**
- The bottleneck is now demonstrably the network, not the driving policy: either widen `PolicyNet` (`HIDDEN_SIZE` beyond 8) to give the BC step enough capacity to fit a higher-variance expert, or apply DAgger (flagged since Entry 11) so training data comes from the *network's own* rollout states rather than only the expert's, which specifically targets exactly this kind of imitation gap.
- The refined (slower, `flat_speed_mps=31.9`) raw candidate was never distilled or validated end-to-end - if a wider network or DAgger fixes the imitation gap, try distilling the *refined* candidate first, since it is closer to zero-damage at the raw-follower level (max damage 0.02 vs. 0.17) and would need less of the network's precision budget spent on recovering from close calls.
- 600 m raw is achieved and validated at the follower level (602.0 m); the remaining problem is entirely "teach a 140-parameter sensor-only network to reproduce it," not "find a faster driving policy" - future sessions should treat those as separate, sequential problems rather than re-running the same follower search again.

## Entry 16 — Widening the Network Fixes the Crashes, Not the Speed Deficit

**Date and time:** September 8, 2026 (continued)

**Question or objective:** Entry 15 found the fast (591.8 m raw) anticipatory-braking expert didn't distill safely (24/25 held-out, one elimination). Two suspects: (1) `PolicyNet`'s `HIDDEN_SIZE=8` (~140 total parameters) may be too small to fit the expert's now much higher-variance throttle behavior, and (2) `SPEED_CAP_MPS=20.0` in `build_inputs`'s speed-feature normalization - the expert reaches ~34 m/s, so every speed reading above 20 m/s was clamped to the same input value 1.0, making the network blind to its own speed across most of its actual operating range. Fixing both, does the distilled network now match the raw expert's 591.8 m?

**What we investigated or changed:**
- Raised `HIDDEN_SIZE` 8 -> 20 and `SPEED_CAP_MPS` 20.0 -> 40.0 in `src/controllers/learned.py`. Re-applied Entry 15's fast follower params as `DEFAULT_FOLLOWER_PARAMS` (reverted at the end of Entry 15). Regenerated demonstrations (feature vectors depend on `SPEED_CAP_MPS`, so a regenerate was required, not just a retrain) and retrained BC (300 epochs, up from 200, since the harder function needed more optimization).
- Training MSE improved only modestly: 0.036 (Entry 15, `HIDDEN_SIZE=8`) -> 0.029 (`HIDDEN_SIZE=20`) - a real but small gain, not the order-of-magnitude drop back toward the ~0.0002 baseline that would suggest capacity was the whole story.

**Evidence:**
- Experiment output: full held-out validation (25 races) via `validate_policy.py` across `(42, 110, 271, 997, 2027)`.
- Commits or code: `src/controllers/learned.py` (`HIDDEN_SIZE`, `SPEED_CAP_MPS` reverted back to 8/20.0 after this entry - not adopted), `artifacts/racing_line_demonstrations_v8.pt`, `artifacts/racing_line_bc_policy_v8.pt` (not shipped).

**What we observed:**
- Held-out validation: **25/25 survived, 0 eliminations** (up from Entry 15's 24/25) - the speed-cap fix in particular directly closed the elimination, consistent with the network previously being unable to perceive braking urgency at genuinely high speed. But **average scored distance was only 518.0 m** - worse than both the 600 m target and the already-shipped Entry 14 baseline (568.0 m), and mostly 2-lap finishes rather than the 3-lap finishes the raw 591.8 m expert reaches.
- This is a new, distinct failure mode from Entry 15's: not crashing, but *slow* - the wider network stopped eliminating but still doesn't reproduce the expert's precise accel/brake timing closely enough, and the resulting small, frequent control errors (visible as occasional 0.08-0.18 damage grazes, and lost time either braking too early or not accelerating cleanly back to speed) cost more net distance than the ~15 m/s higher top speed gains. A `HIDDEN_SIZE` increase from 8 to 20 (roughly 2.5x the parameters) was not enough capacity to close this gap in one shot.
- Reconfirms Entry 15's core finding from a different angle: the bottleneck is specifically *imitation fidelity* on a high-variance expert, not simply "not enough numbers to represent the function" (more hidden units helped some) or "blind to its own speed" (fixing the cap helped eliminate crashes) - both were real, partial contributors, but neither alone (nor both together, in this one attempt) fully closes the gap to the raw expert's performance.

**Decision and rationale:** Did not ship `racing_line_bc_policy_v8.pt` - still a regression on distance versus the shipped Entry 14 policy, despite being safer than Entry 15's attempt. Reverted `src/controllers/learned.py` and `scripts/racing_line_follower.py` back to exactly the shipped Entry 14 state (confirmed via direct weight comparison before and after) so the repository stays consistent with `src/controllers/learned_policy.pt`, which remains unchanged at 568.0 m, 25/25 safe.

**Next steps:**
- Two levers were each tried once, together, in one shot; the search space of (hidden size, speed cap, epoch count, learning rate) around the high-variance expert is still unexplored - a wider sweep (e.g. `HIDDEN_SIZE` 32-48, more epochs) might close the remaining gap, but each attempt costs a full regenerate+retrain+validate cycle (~3-4 minutes), so this needs either a longer uninterrupted budget or a way to test multiple hidden sizes in one pass (e.g. train several networks against the same cached demonstrations dataset in one script run, only regenerating demonstrations once).
- DAgger remains untried and is likely the more fundamentally correct fix (train on the network's own visited states, not just the expert's) rather than continuing to guess at network capacity.
- The refined (slower, 546.6 m raw, max damage 0.02) candidate from Entry 15 was never distilled either - it may be an easier imitation target than the faster one tried here precisely because its own behavior is lower-variance, and could be worth trying before further widening the network for the faster candidate.

## Entry 17 — Session Close-Out: Confirmed Final Shipped State After the 600 m Attempts

**Date and time:** September 8, 2026 (continued)

**Participants and contributions:**
- Leon Aaron — set the repeated 600 m target under progressively tighter time budgets across Entries 13-16, and asked for this closing entry to record the final state and fix the notebook's own ordering.
- AI coding agent (Claude Code) — ran the search and distillation attempts in Entries 15-16, diagnosed why each fell short, reverted every code and weights file back to the last validated-safe state before ending the session, and corrected this notebook's entry order and numbering.

**Question or objective:** After three consecutive attempts (Entry 14's second search pass, Entry 15's anticipatory accel/decel, Entry 16's wider network) failed to ship a net improvement over Entry 14's 568.0 m, what is the actual final state of the repository, and is it verified consistent - shipped weights, the scripts that would reproduce them, and the network architecture - before ending this session?

**What we investigated or changed:**
- Verified `src/controllers/learned_policy.pt`'s weights directly (tensor-by-tensor comparison) against `artifacts/racing_line_bc_policy_v5.pt` (Entry 14's checkpoint) to confirm no later attempt's `cp` or partial edit had silently overwritten the shipped file.
- Verified `src/controllers/learned.py`'s `HIDDEN_SIZE` and `SPEED_CAP_MPS` were reverted to 8 and 20.0 (the values `learned_policy.pt` was actually trained with) after Entry 16 had temporarily raised them to 20 and 40.0 - catching a latent bug where the checked-in architecture no longer matched the checked-in weights, which would raise a `state_dict` shape-mismatch error the next time anyone imported `controllers.learned`.
- Verified `scripts/racing_line_follower.py`'s `DEFAULT_FOLLOWER_PARAMS` matches Entry 14's shipped values, not any of the faster-but-unshipped candidates explored afterward, so re-running `scripts/generate_racing_line_demonstrations.py` today would reproduce the shipped checkpoint rather than silently drifting onto an abandoned branch.
- Corrected this notebook's own ordering: Entry 14 had been appended after Entries 16-17 by mistake during editing rather than immediately after Entry 13, and the two later entries have been renumbered to 15 and 16 to close the gap and match the order the work actually happened in.

**Evidence:**
- Direct tensor comparison (`torch.equal` over every key in both state dicts) between `src/controllers/learned_policy.pt` and `artifacts/racing_line_bc_policy_v5.pt`.
- Source inspection of `src/controllers/learned.py` and `scripts/racing_line_follower.py` against the values recorded in Entry 14.

**What we observed:**
- The shipped policy is confirmed unchanged since Entry 14: **≈568.0 m/30 s, 25/25 survived, damage = 0.00 in every held-out race.** This remains the best *safely shipped* result of the session, despite two later entries (15, 16) each finding a faster *raw* driving policy (591.8 m and, at the follower level only, real headroom toward 600+ m).
- The blocker is now well-characterized rather than a mystery: the small `PolicyNet` (8 hidden units, ~140 parameters) can imitate a near-constant-speed expert almost exactly (training MSE ~0.0002, as in Entries 12-14) but cannot yet imitate the higher-variance, anticipatory-braking expert from Entry 15 closely enough to reproduce its speed safely (Entries 15-16, MSE 0.029-0.036, each shipping-attempt worse than Entry 14 on either safety or net distance).

**Decision and rationale:** Ending the session with `src/controllers/learned_policy.pt` at Entry 14's checkpoint, confirmed consistent with the code that produces it. No unshipped, unvalidated, or architecture-mismatched state was left in the repository.

**Next steps:**
- DAgger (flagged since Entry 11, repeated in Entries 15 and 16) is the most direct fix for the imitation gap: label the *network's own* rollout states with the expert's action, rather than only the expert's clean trajectory, so the network learns recovery behavior it currently never sees during training.
- A systematic `HIDDEN_SIZE` sweep (e.g. 8/16/24/32/48, all against the same cached Entry 15 demonstrations so only one demonstration-generation pass is needed) would answer whether capacity alone can close the gap, cheaper than the single-guess widen-and-retry done in Entry 16.
- The refined, lower-variance Entry 15 candidate (546.6 m raw, max damage 0.02) was never distilled at all - it is a more promising imitation target than the faster candidate actually tried in Entry 16, and should be attempted first in any resumed session.
