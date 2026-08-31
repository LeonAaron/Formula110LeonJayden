# Formula 110 — Laboratory Notebook

## Entry 1 — Reactive Controller Foundation (Strategy 1: Reactive Control + Parameter Optimization)

**Date and time:** August 30, 2026, evening session

**Participants and contributions:**
- Leon Aaron — designed the sensor-to-command control logic, wrote the controller and evaluation harness, ran and interpreted experiments, committed the result.
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
- Leon Aaron — reviewed each training result, decided when to change the fitness function vs. accept a result, and directed the scope of the experiment.
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
