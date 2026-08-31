# Formula 110 — Exploration of Two Experimental Approaches

Per the assignment's exploration stage, we investigated two approaches that differ in mechanism, not just parameters: **reactive control with parameter optimization** (hand-written rules, tuned) and **neuroevolution** (a learned policy, evolved). Both are evaluated with the same headless harness (`scripts/evaluate_controller.py`) and the same fitness definition — scored distance, heavily penalized on elimination — so results are comparable.

## Approach 1: Reactive Control + Parameter Optimization

**Hypothesis:** The simulator's processed sensors (`camera.heading_error_degrees`, `camera.center_offset_m`, `wall_lidar`) already encode steering error and wall proximity directly. A small set of explicit rules connecting these to throttle/steer should produce a controller that survives and completes laps almost immediately, and tuning its parameters (by hand or by search) should recover most of the achievable speed without needing training infrastructure.

**Minimum experiment:** A parametrized rule-based controller (`ReactiveParams`: steering gains for centerline/heading/lookahead + wall-avoidance, a speed-scaled braking distance against `wall_lidar.front_m`, and a curvature-based speed target) evaluated headless, then manually adjusted once based on measured slack (zero damage) rather than guesswork.

**Evaluation:** Seed suite `(42, 110, 271, 997, 2027)`, 5 races per seed (25 total), 30-second rounds. Metrics: survival rate, lap-completion rate, scored distance, damage, max speed.

**Status: implemented and evaluated.**

| Metric | Result |
| --- | --- |
| Survival rate | 25/25 races |
| Lap-completion rate (≥1 lap in 30s) | 25/25 races |
| Damage | 0.00 in every race |
| Avg. scored distance | ~229–234 m per seed (≈1.3 laps) |
| Max speed reached | 13.9 m/s |
| Development effort | Low — one session, no training loop |
| Remaining risk | Parameters are hand-tuned, not searched; likely leaving speed on the table |

Files: `src/controllers/reactive.py`, `scripts/evaluate_controller.py`. Full experimental log: `LAB_NOTEBOOK.md`, Entry 1.

## Approach 2: Neuroevolution (Evolution Strategy over a Small MLP)

**Hypothesis:** A small neural network (10 normalized sensor inputs → 6-unit hidden layer → 2 outputs, ~80 weights) evolved directly against race distance can discover throttle/steer behavior without hand-specifying rules, and — because it optimizes the actual scoring objective end-to-end — may find a faster or more consistent policy than hand-written rules, at the cost of needing many simulated races to train.

**Minimum experiment:** A dependency-free (mu + lambda) evolution strategy: a population of random-weight genomes, evaluated by scored distance with a penalty on elimination, with elitism and Gaussian mutation refilling each next generation. Kept intentionally simple — no covariance adaptation, no external library — to bound development effort for a first pass.

**Evaluation:** Trained on seeds distinct from the recommended suite (fast, 20-second rounds); validated on the held-out suite `(42, 110, 271, 997, 2027)`, 5 races per seed, 30-second rounds — identical protocol to Approach 1 for a direct comparison.

**Status: implemented and trained; not yet reliable.** Four training runs were needed to understand the fitness landscape, each revealing a distinct failure mode:

| Attempt | Fitness design | Training result | Held-out validation (25 races) |
| --- | --- | --- | --- |
| 1 | Distance − 300 if eliminated | Converged to 225 m | 8/25 survived, 18/25 laps — **fast but reckless**: crashing far enough to still out-score cautious genomes |
| 2 | Distance replaced by fixed −50 if eliminated (survival-only) | Converged to 63 m | 20/25 survived, **0/25 laps**, ~0 m avg — discovered that standing still is a free, zero-risk "safe" strategy |
| 3 | Distance − 350 if eliminated, − 20 if idle (< 10 m) | Converged to 260 m (single training seed) | 8/25 survived, 21/25 laps — fast (28–35 m/s) and usually completes a lap, but still crashes in most races |
| 4 | Same fitness as #3, trained on two seeds instead of one | Converged to 53 m | Reproduced attempt #2's idle genome exactly — the wider requirement (safe on two starting points) overwhelmed this population/generation budget |

Attempt 3's genome is kept as `BEST_GENOME` in `src/controllers/neuro.py` — it is the most genuinely "racing" result, even though unreliable.

| Metric | Result (Attempt 3, held-out suite) |
| --- | --- |
| Survival rate | 8/25 races |
| Lap-completion rate | 21/25 races |
| Avg. scored distance | ~187–288 m per seed |
| Max speed reached | 28–35 m/s (2–2.5× the reactive controller's 13.9 m/s) |
| Development effort | High — four training iterations, each requiring a fitness-function diagnosis |
| Remaining risk | Fitness shaping is unresolved; this minimal (mu+lambda) search has not found a genome that is both fast and consistently safe |

Files: `src/controllers/neuro.py`, `scripts/train_neuroevolution.py`.

## Comparison Summary

| | Approach 1: Reactive | Approach 2: Neuroevolution |
| --- | --- | --- |
| Survival rate (held-out suite) | 25/25 | 8/25 |
| Lap-completion rate | 25/25 | 21/25 |
| Max speed | 13.9 m/s | 28–35 m/s |
| Development effort | Low — one session | High — four training iterations to diagnose reward shaping |
| Interpretability | High (readable rules) | Low (opaque weights) |
| Remaining risk | Hand-tuned, not searched — likely leaving speed on the table | Unresolved fitness shaping; no genome yet found that is both fast and reliably safe |

**Reading the evidence:** Approach 1 satisfies the core goal (drive the full track, every seed, no damage) with low effort. Approach 2 has clearly demonstrated a materially higher speed ceiling but has not yet converged on a policy that is also reliably safe — its two failure modes (reckless-but-fast vs. safe-but-static) sit on opposite ends of a speed/safety trade-off that a five-generation, ten-genome search cannot yet resolve. This is the strongest evidence for selecting **Approach 1 as the primary controller**, while treating Approach 2's demonstrated speed ceiling as motivation for a refinement idea worth retaining: a hybrid where the reactive controller's braking/wall-avoidance logic acts as a safety floor under a learned component chasing extra speed.
