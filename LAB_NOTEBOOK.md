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

## Entry 4 — Learned Controller: Behavior Cloning + Dense-Reward CMA-ES (Torch)

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

## Entry 5 — Compute an Optimal Racing Line, Then Distill It Into a Sensor-Only Policy

**Date and time:** September 8, 2026 (continued)

**Question or objective:** Entry 4's gains were marginal because behavior-cloning the reactive controller just re-learns the same apex-bias heuristic. Can we instead (1) compute an actual minimum-curvature racing line for the track offline, (2) build a privileged expert that tracks that specific line using ground-truth arc-length position, and (3) distill *that* expert into the same sensor-only network, so training labels come from a genuinely optimized reference path rather than a hand-tuned heuristic?

**What we investigated or changed:**
- `scripts/compute_racing_line.py`: an offline geometric optimizer. Represents the path as a lateral offset from centerline at the track model's 201 arc-length samples, then iteratively pulls each point toward the midpoint of its neighbors (projected onto the local track normal, clamped to the track corridor) — a standard "iterative corner-cutting" curvature-minimization smoother. Also computes a curvature-limited speed profile (lateral-grip cap, then forward/backward accel-limited passes).
- `scripts/_privileged_race_env.py`: a custom single-car headless stepping loop (reusing the same internal building blocks as `racing.race.head_to_head`) that exposes the *true* arc-length position (`TrackProjection.progress_distance_m`) each tick — something no public `RobotSensors` field provides. Used only for offline expert generation, never by the deployed controller.
- `scripts/racing_line_follower.py`: a privileged expert reusing `controllers.reactive`'s steering/safety structure (lookahead terms, wall avoidance, emergency steer, contact recovery), but replacing the heuristic apex-bias offset and flat max-speed target with the precomputed line's offset(s) and speed(s).
- **First finding (a real negative result):** the curvature-limited speed profile made things *worse* — 250-355 m/30s depending on grip constants, well below the 435 m reactive baseline, despite reaching much higher peak speed. Root cause: this simulator's grip is high enough that Entry 3 already found flat-out driving (no cornering slowdown) beats any curvature-based slowdown; a "physically principled" speed profile reintroduces exactly the cornering deceleration Entry 3 removed. Swept flat speed targets (15.8-30 m/s) against the *optimized offset* line instead: 15.8 m/s (matching reactive's own tuned speed) gave the best, fully safe result.
- Regenerated the racing line with `--flat-speed-mps 15.815503530875` (reactive's exact tuned speed) so only the offset geometry differs from reactive, isolating the effect of the optimized line. Held-out validation of the raw (non-distilled) follower: **452 m avg, 5/5 safe** — a genuine, apples-to-apples +3.9% over reactive's 435 m from the offset geometry alone.
- `scripts/generate_racing_line_demonstrations.py` + `scripts/train_behavior_cloning.py`: distilled this expert into the same sensor-only network architecture as Entry 4 (never given the racing line or ground-truth position as input, only `controllers.learned.build_inputs`'s public sensor features). Training MSE converged far lower (0.00055) than Entry 4's reactive-clone (0.0167), reflecting a cleaner, more consistent expert.

**What we observed:** Held-out validation (25 races) of the distilled network: **25/25 survived, 25/25 completed 2 laps, avg. scored distance ≈442.8 m** — essentially tied with Entry 4's reactive-clone (444.3 m), not a clear win, even though the raw undistilled expert (452 m) was better than reactive. Distillation lost roughly the same margin it gained over reactive. Both distilled policies land in the same ~440-450 m / 2-lap band; neither approach has yet found a way to convert a better *reference line* into a proportionally better *learned* policy.

**Decision and rationale:** Shipped the racing-line-distilled policy (`artifacts/racing_line_bc_policy.pt` → `src/controllers/learned_policy.pt`) rather than Entry 4's reactive-clone, since it directly implements the requested approach (compute the optimal line, train a controller to follow it, distill into a sensor-generalizing network) and is not worse. Both the geometric line-optimization code and the privileged single-car environment are reusable for future work regardless of which distilled checkpoint ships.

**Next steps:**
- The gap between the raw expert (452 m) and its distillation (442.8 m) suggests classic behavior-cloning covariate shift: the network never sees the small-error recovery states its own imperfect tracking creates during autonomous rollout, only the expert's near-perfect trajectory. A DAgger-style pass (re-label the *learned* policy's own visited states with the expert's action, then retrain) is the standard fix and wasn't attempted this session.
- The offline line optimizer is still track-specific (computed from this track's exact centerline); true cross-track generalization claims should be tested by computing a second line for a different track layout (if/when one exists) and checking whether the *same* trained network - not retrained - still drives it competently.
- 4 laps remains unreached; the biggest lever demonstrated so far is still Entry 4's CMA-ES speed ceiling (27-30 m/s) - combining that with this entry's better offset geometry (rather than reactive's heuristic apex bias) is the most promising untried combination.

## Entry 6 — Raising the Racing-Line Follower's Flat Speed: a Damage-Margin Sweep, Not a Blind Hillclimb

**Date and time:** September 8, 2026 (continued)

**Question or objective:** Entry 5 shipped the racing-line-distilled policy at `flat_speed_mps = 15.815503530875` (reactive's own tuned speed, chosen only to isolate the effect of the offset geometry). Since `RacingLineFollowerParams.flat_speed_mps` is a free parameter never actually searched, can raising it push scored distance further while keeping the same zero-damage safety margin the reactive and BC controllers have shown throughout this notebook?

**What we investigated or changed:**
- Found `scripts/optimize_racing_line_follower.py` (an evolution-strategy search over `RacingLineFollowerParams`, symmetric to `scripts/optimize_reactive.py`) and `artifacts/racing_line_bc_policy_v2.pt` already present from an earlier unlogged pass in this session: the search had pushed `flat_speed_mps` from 15.8 to 23.09 m/s and a BC policy had been distilled from it, but never validated against the held-out seed suite.
- Validated `racing_line_bc_policy_v2.pt` (25 races, held-out seeds): **only 7/25 survived, avg scored distance 331.5 m** — worse than the shipped 442.8 m baseline. A genuine negative result: the fitness-guided search found a *training* optimum that does not generalize.
- Diagnosed why, using the privileged (ground-truth-position) follower directly, bypassing distillation: at `flat_speed_mps = 23.09`, the raw expert itself is not damage-free — **20/20 survived but with damage 0.37–0.67 per race** (avg 501.9 m), a materially different safety margin than reactive/BC's demonstrated 0.00. The search's fitness function (distance minus an elimination penalty) rewards surviving-with-damage the same as surviving cleanly, so it happily traded away the safety margin that made the small-network distillation step generalize in Entries 3-5. Under an imperfect sensor-only clone, that margin evaporates and pushes many races over the 1.0 elimination threshold.
- Swept `flat_speed_mps` directly against the raw privileged follower (20 seeds) instead of trusting the search's fitness number: distance rises smoothly and damage stays exactly 0.00 up to **17.0 m/s** (517–519 m every seed), then damage appears abruptly at 17.5 m/s (0.03–0.41) with distance *dropping* (wall contact drag), confirming a sharp, measurable zero-damage ceiling rather than a smooth tradeoff.
- Set `RacingLineFollowerParams.flat_speed_mps = 17.0` (replacing the abandoned 23.09 default), regenerated demonstrations (`scripts/generate_racing_line_demonstrations.py`, same 40-seed/48k-tick recipe as Entry 5) and retrained the BC network (`scripts/train_behavior_cloning.py`) — training MSE converged to 0.00021, even lower than Entry 5's 0.00055.

**Evidence:**
- Sources or documentation: none new; built on Entry 5's pipeline.
- AI-agent assistance: Claude Code found and validated the pre-existing but unshipped v2 experiment, diagnosed its generalization failure via the privileged follower (isolating speed-induced damage from distillation error), ran the flat-speed sweep, and re-ran the full generate-demonstrations -> train-BC -> validate pipeline at the measured safe ceiling.
- Commits or code: `scripts/racing_line_follower.py` (`flat_speed_mps` default 23.09 -> 17.0, with the measured rationale recorded in a comment), `artifacts/racing_line_demonstrations_v3.pt`, `artifacts/racing_line_bc_policy_v3.pt` -> `src/controllers/learned_policy.pt`.
- Experiment output: held-out validation (25 races, `uv run python scripts/evaluate_controller.py --module controllers.learned --seed <seed> --races 5 --round-seconds 30` across `(42, 110, 271, 997, 2027)`).
- Leaderboard result: not applicable — local evaluation only.

**What we observed:**
- Held-out validation of the new shipped policy: **25/25 survived, 25/25 completed 2 laps, damage = 0.00 in 24/25 races (one race at 0.04)**, average scored distance **≈511 m per 30 s** — a **+15.4% improvement over Entry 5's shipped 442.8 m**, and +17.5% over the original reactive baseline (435 m), with the same safety record.
- The key lesson: an evolution-strategy search optimizing raw distance-with-elimination-penalty is not the same objective as "distance at zero damage" — it will spend safety margin for speed whenever the elimination penalty is smaller than the expected gain, and that traded-away margin is specifically what BC distillation needs to survive its own approximation error. Measuring the *raw, privileged* controller's damage directly (not just survival) caught this before it was baked into another distillation cycle.

**Decision and rationale:** Shipped the re-distilled policy (`racing_line_bc_policy_v3.pt` -> `src/controllers/learned_policy.pt`) as the new `learned_policy.pt`, chosen by measuring the actual zero-damage speed ceiling rather than trusting an unvalidated search result. This is now the controller to beat: ≈511 m/30s, 25/25 safe.

**Next steps:**
- The 17.0 -> 17.5 m/s cliff is measured on 10-20 seeds only; a wider seed sweep (or a search that directly penalizes any nonzero damage, not just elimination) could locate a tighter, more confident ceiling, or reveal it varies slightly by starting position.
- The abandoned 23 m/s attempt still demonstrates real headroom (501.9 m raw, before distillation loss) if a way is found to make the sensor-only network as damage-tolerant as the privileged expert at that speed — e.g., DAgger (Entry 5's own open item) to teach the network recovery behavior from its own near-wall states rather than only the expert's clean trajectory.
- 4 laps (~724 m in 30 s) is closer than before (511 m now vs. 435-444 m previously) but still requires either the DAgger fix above or Entry 4's higher-speed CMA-ES ceiling combined safely with this entry's line geometry.
