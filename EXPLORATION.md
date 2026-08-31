# Formula 110 — Exploration of Two Experimental Approaches

Per the assignment's exploration stage, we investigated two approaches that differ in mechanism, not just parameters: **reactive control with parameter optimization** (hand-written rules, tuned) and **neuroevolution** (a learned policy, evolved). Both are evaluated with the same headless harness (`scripts/evaluate_controller.py`) and the same fitness definition — scored distance, heavily penalized on elimination — so results are comparable.

## Approach 1: Reactive Control + Parameter Optimization

**Hypothesis:** The simulator's processed sensors (`camera.heading_error_degrees`, `camera.center_offset_m`, `wall_lidar`) already encode steering error and wall proximity directly. A small set of explicit rules connecting these to throttle/steer should produce a controller that survives and completes laps almost immediately, and tuning its parameters (by hand or by search) should recover most of the achievable speed without needing training infrastructure.

**Minimum experiment:** A parametrized rule-based controller (`ReactiveParams`: steering gains for centerline/heading/lookahead + wall-avoidance, a speed-scaled braking distance against `wall_lidar.front_m`) evaluated headless, then manually adjusted once based on measured slack (zero damage) rather than guesswork.

**Evaluation:** Seed suite `(42, 110, 271, 997, 2027)`, 5 races per seed (25 total), 30-second rounds. Metrics: survival rate, lap-completion rate, scored distance, damage, max speed.

**Status: implemented, redesigned, and parameter-optimized.**

The controller went through two stages of improvement past the initial hand-tuned baseline:

1. **Redesign** (apex-hugging steering + no proactive cornering brake): steering now biases toward the inside of an upcoming turn — inferred from `heading_error_degrees` — rather than the raw centerline, since the simulator scores centerline *projection*, so cutting a turn's inside advances scored distance per meter driven. Throttle always targets top speed; the only source of braking is the reactive wall-proximity check.
2. **Parameter search** (`scripts/optimize_reactive.py`): a (mu + lambda) evolution strategy seeded from the redesigned defaults, with a diagnostic tracer that reports per-tick sensor/command state to explain *why* a parameter set does well or poorly (e.g. it revealed the pre-search defaults were braking on 43% of ticks — not from cornering logic, but because the safety margin scaled too conservatively with speed).

| Stage | Survival | Laps | Damage | Avg. scored distance | Max speed |
| --- | --- | --- | --- | --- | --- |
| Original hand-tuned baseline | 25/25 | 25/25 (≥1) | 0.00 | ~229 m | 13.9 m/s |
| + Apex-hugging redesign (no search yet) | 25/25 | 25/25 (≥1) | 0.00 | ~259 m (+13%) | 13.9 m/s |
| + Parameter search | 25/25 | 25/25 (2 laps) | 0.00 | **~435 m (+90%)** | 15.7 m/s |

| Metric | Result |
| --- | --- |
| Development effort | Low-to-moderate — one redesign pass, one ~3-minute search run |
| Remaining risk | `steer_limit` settled at 0.63 (well under the 1.0 max), suggesting there may be more speed available in the sharpest corners |

Files: `src/controllers/reactive.py`, `scripts/evaluate_controller.py`, `scripts/optimize_reactive.py`. Full experimental log: `LAB_NOTEBOOK.md`, Entries 1 and 3.

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

| | Approach 1: Reactive (optimized) | Approach 2: Neuroevolution |
| --- | --- | --- |
| Survival rate (held-out suite) | 25/25 | 8/25 |
| Lap-completion rate | 25/25 (2 laps each) | 21/25 (1 lap) |
| Avg. scored distance | ~435 m | ~187–288 m per seed |
| Max speed | 15.7 m/s | 28–35 m/s |
| Development effort | Low-to-moderate — one redesign pass, one ~3-minute search run | High — four training iterations to diagnose reward shaping |
| Interpretability | High (readable rules) | Low (opaque weights) |
| Remaining risk | `steer_limit` below max suggests unclaimed speed in sharp corners | Unresolved fitness shaping; no genome yet found that is both fast and reliably safe |

**Reading the evidence:** Approach 1 now leads on every metric that matters except raw top speed — it covers more distance overall (435 m vs. Approach 2's best-case ~290 m) *and* does so with perfect reliability. Approach 2's neuroevolution still demonstrates a materially higher physical speed ceiling (28–35 m/s vs. 15.7 m/s) that Approach 1 has not reached, but it has not converged on a policy that is also safe. Both approaches improved via the same lesson — seed a search from a known-good baseline rather than random initialization — which explains why Approach 1's search converged smoothly (worst-of-generation was never catastrophic) while Approach 2's random-init runs repeatedly found degenerate optima. This is decisive evidence for selecting **Approach 1 as the primary controller**. Approach 2's speed ceiling remains motivation for the hybrid idea noted earlier: let a learned component chase Approach 2's demonstrated top speed under Approach 1's safety floor.
