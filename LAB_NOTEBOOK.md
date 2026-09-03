# Formula 110 — Laboratory Notebook

## Entry 1 — Reactive Controller Foundation (Strategy 1: Reactive Control + Parameter Optimization)

**Date and time:** August 30, 2026, evening session

**Participants and contributions:**
- Leon Aaron and Jayden Webb— designed the sensor-to-command control logic, wrote the controller and evaluation harness, ran and interpreted experiments, committed the result.
- AI coding agent (Claude Code) — read the simulator's sensor, physics, and scoring internals to g

**Question or objective:**
Can a hand-written reactive controller (explicit sensor→command rules with tunable parameters) drive the car around the full track and complete at least one lap within 30 seconds, from a random seeded starting position, without being destroyed (damage reaching 1.0)? This is the foundation for both our baseline and the parameter-optimization step that follows.

**What we investigated or changed:**
- Read the simulator's public contract (`src/racing/student/api.py`, `SENSORS.md`) and internals (`race/sensors.py`, `race/progress.py`, `race/runtime.py`, `physics/engine.py`) to confirm sensor sign conventions, the lap-counting rule (cumulative forward progress since spawn crossing the track length — not a literal return to the start/finish line), and the damage model where car is eliminated at damage ≥ 1.0.
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

## Entry 7 — reactive2.py: New Sensor Mechanisms, Worst-Seed Search, Rejected on Validation

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

## Entry 8 — reactive2.py Re-Search with a Safety-Margin Floor: Damage Fixed, Gain Erased

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

## Entry 7 — Neuroevolution: Wider Population Closes the Last Held-Out Elimination

**Date and time:** August 31, 2026 (continued)

**Participants and contributions:**
- Jayden Webb — directed the experiment (widen population/generations rather than change the fitness function again), and set the rule for this entry: only update `BEST_GENOME` and this notebook if the result beats Entry 5, otherwise keep the shipped genome and just report back.
- [Add teammate name and contribution here.]
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
