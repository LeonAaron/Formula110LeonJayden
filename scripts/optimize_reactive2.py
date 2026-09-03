"""Parameter-search and diagnostic tool for controllers.reactive2.ReactiveParams.

Optimization: a simple (mu + lambda) evolution strategy, seeded from the
current DEFAULT_PARAMS (a known-safe baseline) rather than random init, since
a from-scratch random search over throttle/steer gains risks wasting most of
its budget on unstable genomes (see the neuroevolution experiment's random-init
struggles in LAB_NOTEBOOK.md, Entry 2). Each parameter is perturbed by noise
scaled to its own magnitude, so one shared step size works across gains,
angles, and distances of very different scale.

Fitness: a blend of the mean and the worst per-seed score (see
WORST_SEED_PENALTY_WEIGHT below), where each per-seed score is distance minus
a large penalty on elimination, minus a small penalty for going nowhere,
minus a penalty proportional to non-fatal damage. The damage term matters: an
earlier version only penalized full elimination, and the search found a
genome that clipped walls hard on every corner but technically survived its
short training rounds, then got eliminated in most held-out validation races.
Penalizing damage directly (not just death) closed that gap. The worst-seed
blend addresses a related, later-discovered gap (LAB_NOTEBOOK.md, Entry 5):
averaging across seeds can still let a genome that's excellent on most seeds
and eliminated on one look good on average — blending in the worst seed's
score pushes the search toward genomes that are safe on every seed, not just
safe on average, without going as far as pure worst-case fitness (which risks
collapsing the search to a degenerate idle optimum, as also seen in Entry 2).

Diagnostics: --diagnose reruns one race with full per-tick tracing (sensors +
command, including reactive2.py's new sensor signals: lateral/forward
acceleration, the raw diagonal wall-lidar beams, roll/pitch, the curvature
signal, and whether the instability override fired) and prints, in order:
  - a summary (distance, laps, damage, marshal activity, max speed)
  - how often the car braked and how close it got to walls
  - how often the new reactive2.py mechanisms were active
  - the sensor/command state at the moment of first wall contact, if any
  - the last several ticks before the race ended, if the car was eliminated

This is meant to answer "why did this genome do well or poorly", using
concrete sensor values, the throttle/steer output, and the simulated time at
which the event happened - not just a fitness number.

Usage:
    uv run python scripts/optimize_reactive2.py --population 20 --generations 18 --phase2-generations 10
    uv run python scripts/optimize_reactive2.py --diagnose --seed 110
"""

from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, replace
from functools import partial
from statistics import mean

from controllers.reactive2 import DEFAULT_PARAMS, ReactiveParams, drive
from racing import HeadToHeadTeamRaceStats, RobotCommand, RobotSensors, run_headless_head_to_head

ELIMINATION_PENALTY_M = 350.0
IDLE_DISTANCE_M = 10.0
IDLE_PENALTY_M = 20.0
DAMAGE_PENALTY_SCALE_M = 300.0

# Blend factor between mean and worst-seed per-seed score: fitness =
# (1 - w) * mean + w * min. w=0.5 directly targets Entry 5's "mean masks a
# single bad seed" concern without repeating Entry 2/5's opposite failure
# (pure worst-case fitness collapsing the search to a degenerate idle
# optimum): a genome cannot win by being spectacular on 4/5 seeds and
# mediocre on the 5th, but it also isn't purely hostage to whichever single
# seed the mutation operator happens to disturb most that generation.
WORST_SEED_PENALTY_WEIGHT = 0.5

PARAM_NAMES: tuple[str, ...] = tuple(field.name for field in fields(ReactiveParams))
Genome = tuple[float, ...]

# Sensible physical/behavioral bounds per parameter, so mutation cannot waste
# evaluations on nonsensical values (e.g. negative distances, a steer limit
# above 1.0, a braking gain of zero). Keyed by field name; any field not
# listed is left unclamped.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "center_offset_gain": (0.0, 0.5),
    "heading_error_gain": (0.0, 0.5),
    "lookahead_near_gain": (0.0, 0.3),
    "lookahead_far_gain": (0.0, 0.3),
    # Floor raised to the proven-safe DEFAULT_PARAMS value (0.998) after
    # LAB_NOTEBOOK.md Entry 7's rejected candidate bought distance mainly by
    # shaving this margin to 0.544 — safe on the 5 training seeds' specific
    # corners but caused non-fatal damage on held-out validation corners.
    # The search should earn distance from the new mechanisms, not by
    # eroding an already-tuned safety margin below what's been validated.
    "wall_avoid_margin_m": (0.998, 6.5),
    "wall_avoid_gain": (0.0, 1.0),
    "emergency_front_m": (0.5, 5.0),
    "emergency_steer": (0.0, 3.0),
    "recovery_steer": (0.0, 2.0),
    "recovery_throttle": (-1.0, 0.0),
    "steer_limit": (0.2, 1.0),
    "turn_sharpness_deg": (5.0, 90.0),
    "apex_bias_max_m": (0.0, 2.0),
    "max_speed_mps": (5.0, 30.0),
    "speed_gain": (0.05, 1.5),
    "corner_speed_gain": (0.0, 0.9),
    "corner_signal_deg": (5.0, 120.0),
    "corner_yaw_rate_deg_per_s": (5.0, 300.0),
    "brake_lead_time_s": (0.0, 2.0),
    "brake_min_distance_m": (0.1, 3.0),
    "brake_gain": (0.1, 3.0),
    "brake_quadratic_coeff": (0.0, 0.5),
    # --- reactive2.py's new fields ---
    # B. traction / wall-scrape / ineffective-brake detector.
    "traction_loss_throttle_threshold": (0.1, 0.9),
    "traction_loss_expected_accel_mps2": (0.2, 8.0),  # upper bound ~= max_engine_force/mass_kg (800/92)
    "traction_loss_gain": (0.0, 1.0),
    # C. diagonal early-warning wall avoidance (raw +-45 degree beams).
    # Floor raised to 1.5 for the same Entry 7 reason as wall_avoid_margin_m:
    # this mechanism has no prior "safe" baseline (it was always inert
    # before), so 1.5 is a deliberately conservative minimum rather than the
    # bare 0.5 that let the rejected candidate leave itself almost no
    # diagonal buffer (0.541) at a nonzero gain.
    "diagonal_avoid_margin_m": (1.5, 8.0),  # wider ceiling than wall_avoid_margin_m: +-45 beams see further
    "diagonal_avoid_gain": (0.0, 1.0),
    # D. three-point curvature/anticipation signal.
    "curvature_gain": (0.0, 0.6),  # smaller-magnitude signal than a raw offset; needs more headroom
    # E. damage-based caution scaler.
    "damage_speed_gain": (0.0, 1.0),  # damage is already normalized 0-1
    # F. roll/pitch safety cutoff.
    "attitude_cutoff_deg": (2.0, 45.0),
    "attitude_cutoff_gain": (0.0, 1.0),
    # G. speed-scaled steering safety margins.
    "speed_scaled_margin_lead_s": (0.0, 1.0),  # same order of magnitude as brake_lead_time_s
    # H. instability/skid stability-recovery override. Bounds informed by the
    # observed on-track distribution of these signals during a normal,
    # zero-damage race (see LAB_NOTEBOOK.md Entry 7): yaw rate reaches up to
    # ~270 deg/s and lateral acceleration up to ~43 m/s^2 even while driving
    # safely, since lateral_acceleration_mps2 = speed * yaw_rate is a
    # deterministic kinematic quantity, not an independent slip measurement —
    # both bounds sit high enough that the search can find a threshold that
    # doesn't trivially fire on ordinary hard cornering.
    "instability_yaw_rate_deg_per_s": (60.0, 400.0),
    "instability_lateral_accel_mps2": (5.0, 60.0),
    "instability_gain": (0.0, 1.0),
    # I. predictive multi-beam braking.
    "brake_front_cone_blend": (0.0, 1.0),
}


def _clamp_gene(name: str, value: float) -> float:
    bounds = PARAM_BOUNDS.get(name)
    if bounds is None:
        return value
    low, high = bounds
    return min(max(value, low), high)


def genome_from_params(params: ReactiveParams) -> Genome:
    """Flatten a ReactiveParams into a genome tuple, in dataclass field order."""
    return tuple(getattr(params, name) for name in PARAM_NAMES)


def params_from_genome(genome: Genome) -> ReactiveParams:
    """Rebuild a ReactiveParams from a genome tuple."""
    return replace(DEFAULT_PARAMS, **dict(zip(PARAM_NAMES, genome, strict=True)))


def mutate(genome: Genome, rng: random.Random, sigma_fraction: float) -> Genome:
    """Perturb each gene by Gaussian noise scaled to that gene's own magnitude, then clamp to bounds."""
    mutated: list[float] = []
    for name, value in zip(PARAM_NAMES, genome, strict=True):
        step_sigma = max(abs(value), 0.05) * sigma_fraction
        mutated.append(_clamp_gene(name, value + rng.gauss(0.0, step_sigma)))
    return tuple(mutated)


def _passive_controller(sensors: RobotSensors) -> RobotCommand:
    """Coast in place so the challenger's solo run is barely disturbed."""
    return RobotCommand(throttle=0.0, steer=0.0)


def make_controller(params: ReactiveParams):
    """Build a callable controller bound to one parameter set."""

    def control(sensors: RobotSensors) -> RobotCommand:
        return drive(sensors, params)

    return control


def evaluate_params(params: ReactiveParams, *, seeds: tuple[int, ...], round_seconds: float) -> float:
    """Score a parameter set by a mean/worst-seed blend of per-seed scores (three-tier penalty).

    Returns ``(1 - w) * mean(race_scores) + w * min(race_scores)`` where
    ``w = WORST_SEED_PENALTY_WEIGHT``, rather than a plain mean, so a genome
    that is excellent on most seeds but eliminated on one cannot look good on
    average alone. See LAB_NOTEBOOK.md, Entry 5.
    """
    race_scores: list[float] = []
    for seed in seeds:
        result = run_headless_head_to_head(
            challenger_controller=make_controller(params),
            incumbent_controller=_passive_controller,
            race_count=1,
            round_seconds=round_seconds,
            random_seed=seed,
        )
        stats = result.races[0].challenger
        distance_m = stats.distances_m[0]
        damage = stats.damages[0]
        eliminated = damage >= 1.0
        if eliminated:
            race_scores.append(distance_m - ELIMINATION_PENALTY_M)
        elif distance_m < IDLE_DISTANCE_M:
            race_scores.append(distance_m - IDLE_PENALTY_M)
        else:
            # Penalize non-fatal damage too, not just full elimination — an
            # earlier version only penalized elimination and evolution found
            # a genome that clipped walls hard on every corner, surviving its
            # short training rounds but getting eliminated in most held-out
            # validation races. See LAB_NOTEBOOK.md, Entry 4.
            race_scores.append(distance_m - DAMAGE_PENALTY_SCALE_M * damage)
    mean_score = mean(race_scores)
    worst_score = min(race_scores)
    return mean_score - WORST_SEED_PENALTY_WEIGHT * (mean_score - worst_score)


def _evaluate_genome(genome: Genome, *, seeds: tuple[int, ...], round_seconds: float) -> tuple[float, Genome]:
    """Module-level (picklable) wrapper so ProcessPoolExecutor workers can call it."""
    return evaluate_params(params_from_genome(genome), seeds=seeds, round_seconds=round_seconds), genome


def run_search(
    *,
    population_size: int,
    generations: int,
    elite_count: int,
    mutation_sigma_fraction: float,
    seeds: tuple[int, ...],
    round_seconds: float,
    rng_seed: int,
    seed_genome: Genome | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[ReactiveParams, float]:
    """Run the evolution strategy; return the best params found and their fitness.

    Seeds the initial population from ``seed_genome`` (defaults to the current
    DEFAULT_PARAMS) rather than random init, so every generation's worst
    genome is still a plausible driver. Passing a previous call's best params
    back in as ``seed_genome`` chains a broad-exploration phase into a
    fine-tuning phase.

    Each generation's population members are independent (mu+lambda) fitness
    evaluations, so passing ``executor`` fans them out across processes —
    the sequential version left most of the machine's cores idle every
    generation despite the population having no cross-genome dependencies.
    """
    rng = random.Random(rng_seed)
    base_genome = genome_from_params(DEFAULT_PARAMS) if seed_genome is None else seed_genome
    population = [base_genome] + [
        mutate(base_genome, rng, mutation_sigma_fraction) for _ in range(population_size - 1)
    ]

    best_genome = base_genome
    best_fitness = float("-inf")
    evaluate_one = partial(_evaluate_genome, seeds=seeds, round_seconds=round_seconds)

    for generation in range(generations):
        results = (
            executor.map(evaluate_one, population) if executor is not None else map(evaluate_one, population)
        )
        scored = sorted(results, key=lambda item: item[0], reverse=True)
        generation_best_fitness, generation_best_genome = scored[0]
        if generation_best_fitness > best_fitness:
            best_fitness, best_genome = generation_best_fitness, generation_best_genome

        fitnesses = [fitness for fitness, _ in scored]
        print(
            f"generation {generation}: best={generation_best_fitness:7.1f}m "
            f"mean={mean(fitnesses):7.1f}m worst={min(fitnesses):7.1f}m"
        )

        elites = [genome for _, genome in scored[:elite_count]]
        child_count = population_size - len(elites)
        children = [
            mutate(elites[rng.randrange(len(elites))], rng, mutation_sigma_fraction) for _ in range(child_count)
        ]
        population = [*elites, *children]

    return params_from_genome(best_genome), best_fitness


def print_params_literal(params: ReactiveParams) -> None:
    """Print params as a ReactiveParams(...) literal, ready to paste as DEFAULT_PARAMS."""
    print("ReactiveParams(")
    for name in PARAM_NAMES:
        print(f"    {name}={getattr(params, name)!r},")
    print(")")


# --- Diagnostics -------------------------------------------------------------


def trace_race(
    params: ReactiveParams, *, seed: int, round_seconds: float
) -> tuple[list[dict[str, float]], HeadToHeadTeamRaceStats]:
    """Run one race, recording per-tick sensor/command state, and return (trace, stats)."""
    trace: list[dict[str, float]] = []

    def traced_control(sensors: RobotSensors) -> RobotCommand:
        command = drive(sensors, params)

        camera = sensors.camera
        curvature_signal = 0.0
        offsets = camera.lookahead_offsets_m
        distances = camera.lookahead_distances_m
        if camera.visible and len(offsets) >= 3 and len(distances) >= 3:
            near_run_m = distances[1] - distances[0]
            far_run_m = distances[2] - distances[1]
            if near_run_m > 0.0 and far_run_m > 0.0:
                slope_near = (offsets[1] - offsets[0]) / near_run_m
                slope_far = (offsets[2] - offsets[1]) / far_run_m
                curvature_signal = slope_far - slope_near
        instability_triggered = (
            abs(sensors.imu.yaw_rate_degrees_per_s) > params.instability_yaw_rate_deg_per_s
            and abs(sensors.imu.lateral_acceleration_mps2) > params.instability_lateral_accel_mps2
        )

        trace.append(
            {
                "time_s": sensors.tick / 60.0,
                "speed_mps": sensors.odometry.speed_mps,
                "heading_error_deg": sensors.camera.heading_error_degrees,
                "center_offset_m": sensors.camera.center_offset_m,
                "front_m": sensors.wall_lidar.front_m,
                "left_m": sensors.wall_lidar.left_m,
                "right_m": sensors.wall_lidar.right_m,
                "any_contact_s": sensors.contact.any_contact,
                "damage": sensors.contact.damage,
                "throttle": command.throttle,
                "steer": command.steer,
                "lateral_accel_mps2": sensors.imu.lateral_acceleration_mps2,
                "forward_accel_mps2": sensors.imu.forward_acceleration_mps2,
                "diag_left_m": sensors.wall_lidar.distance_at_angle_degrees(-45.0),
                "diag_right_m": sensors.wall_lidar.distance_at_angle_degrees(45.0),
                "roll_deg": sensors.imu.roll_degrees,
                "pitch_deg": sensors.imu.pitch_degrees,
                "curvature_signal": curvature_signal,
                "instability_triggered": 1.0 if instability_triggered else 0.0,
            }
        )
        return command

    result = run_headless_head_to_head(
        challenger_controller=traced_control,
        incumbent_controller=_passive_controller,
        race_count=1,
        round_seconds=round_seconds,
        random_seed=seed,
    )
    return trace, result.races[0].challenger


def summarize_trace(
    trace: list[dict[str, float]], stats: HeadToHeadTeamRaceStats, *, seed: int, tail: int
) -> None:
    """Print a diagnostic report: what happened, when, and the sensor state at the time."""
    distance_m = stats.distances_m[0]
    laps = stats.lap_counts[0]
    damage = stats.damages[0]
    eliminated = damage >= 1.0
    max_speed = stats.max_speeds_mps[0]
    marshal_count = stats.marshal_counts[0]

    print(f"=== seed {seed}: {'ELIMINATED' if eliminated else 'survived'} ===")
    print(
        f"distance={distance_m:.1f}m laps={laps} damage={damage:.2f} "
        f"max_speed={max_speed:.1f}m/s marshal={marshal_count}x ticks={len(trace)}"
    )

    braking_ticks = sum(1 for row in trace if row["throttle"] < 0.0)
    close_wall_ticks = sum(1 for row in trace if row["front_m"] < 3.0)
    print(
        f"braking: {braking_ticks}/{len(trace)} ticks "
        f"({100.0 * braking_ticks / max(1, len(trace)):.1f}%) | "
        f"front wall < 3m: {close_wall_ticks}/{len(trace)} ticks"
    )

    max_lateral_accel = max((abs(row["lateral_accel_mps2"]) for row in trace), default=0.0)
    max_tilt_deg = max((max(abs(row["roll_deg"]), abs(row["pitch_deg"])) for row in trace), default=0.0)
    instability_ticks = sum(1 for row in trace if row["instability_triggered"] > 0.0)
    curvature_active_ticks = sum(1 for row in trace if abs(row["curvature_signal"]) > 1e-6)
    print(
        f"reactive2 signals: max|lateral_accel|={max_lateral_accel:.1f}m/s^2 "
        f"max|roll or pitch|={max_tilt_deg:.1f}deg "
        f"instability triggered: {instability_ticks}/{len(trace)} ticks "
        f"curvature signal nonzero: {curvature_active_ticks}/{len(trace)} ticks"
    )

    first_contact_index = next((i for i, row in enumerate(trace) if row["any_contact_s"] > 0.0), None)
    if first_contact_index is not None:
        row = trace[first_contact_index]
        print(
            f"first contact at t={row['time_s']:.2f}s: speed={row['speed_mps']:.1f}m/s "
            f"front={row['front_m']:.2f}m left={row['left_m']:.2f}m right={row['right_m']:.2f}m "
            f"heading_error={row['heading_error_deg']:.1f} center_offset={row['center_offset_m']:.2f} "
            f"-> throttle={row['throttle']:.2f} steer={row['steer']:.2f}"
        )

    if eliminated:
        print(f"last {tail} ticks before the trace ended (elimination or race end):")
        for row in trace[-tail:]:
            print(
                f"  t={row['time_s']:6.2f}s speed={row['speed_mps']:5.1f} front={row['front_m']:6.2f} "
                f"left={row['left_m']:6.2f} right={row['right_m']:6.2f} heading_err={row['heading_error_deg']:7.1f} "
                f"center_off={row['center_offset_m']:6.2f} damage={row['damage']:.2f} "
                f"-> throttle={row['throttle']:5.2f} steer={row['steer']:5.2f}"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--mutation-sigma-fraction", type=float, default=0.25)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[13, 55, 110, 271, 997],
        help="Training seeds, kept distinct from the held-out validation suite",
    )
    parser.add_argument("--round-seconds", type=float, default=20.0)
    parser.add_argument("--rng-seed", type=int, default=1, help="Seed for the evolutionary search's own randomness")
    parser.add_argument(
        "--phase2-generations",
        type=int,
        default=0,
        help="If > 0, run a second fine-tuning phase for this many generations, seeded from phase 1's best",
    )
    parser.add_argument("--phase2-mutation-sigma-fraction", type=float, default=0.08)
    parser.add_argument("--diagnose", action="store_true", help="Skip search; trace DEFAULT_PARAMS on --seed instead")
    parser.add_argument("--seed", type=int, default=110, help="Seed used by --diagnose")
    parser.add_argument("--diagnose-round-seconds", type=float, default=30.0)
    parser.add_argument("--tail", type=int, default=15, help="Ticks to print before an eliminated trace ends")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes for fitness evaluation (0 = sequential, -1 = os.cpu_count())",
    )
    args = parser.parse_args()

    if args.diagnose:
        trace, stats = trace_race(DEFAULT_PARAMS, seed=args.seed, round_seconds=args.diagnose_round_seconds)
        summarize_trace(trace, stats, seed=args.seed, tail=args.tail)
        return

    worker_count = (os.cpu_count() or 1) if args.workers < 0 else args.workers
    executor = ProcessPoolExecutor(max_workers=worker_count) if worker_count > 0 else None
    try:
        if executor is not None:
            print(f"(parallel: {worker_count} worker processes)")

        print(f"=== Phase 1: broad exploration (sigma={args.mutation_sigma_fraction}) ===")
        best_params, best_fitness = run_search(
            population_size=args.population,
            generations=args.generations,
            elite_count=args.elite,
            mutation_sigma_fraction=args.mutation_sigma_fraction,
            seeds=tuple(args.seeds),
            round_seconds=args.round_seconds,
            rng_seed=args.rng_seed,
            executor=executor,
        )

        if args.phase2_generations > 0:
            print(f"=== Phase 2: fine-tuning (sigma={args.phase2_mutation_sigma_fraction}) ===")
            best_params, best_fitness = run_search(
                population_size=args.population,
                generations=args.phase2_generations,
                elite_count=args.elite,
                mutation_sigma_fraction=args.phase2_mutation_sigma_fraction,
                seeds=tuple(args.seeds),
                round_seconds=args.round_seconds,
                rng_seed=args.rng_seed + 1,
                seed_genome=genome_from_params(best_params),
                executor=executor,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print("-" * 72)
    print(f"best training fitness: {best_fitness:.1f}m (seeds {args.seeds}, {args.round_seconds:.0f}s rounds)")
    print_params_literal(best_params)

    print("-" * 72)
    print("diagnostic trace of the best params on the training seeds:")
    for seed in args.seeds:
        trace, stats = trace_race(best_params, seed=seed, round_seconds=args.round_seconds)
        summarize_trace(trace, stats, seed=seed, tail=args.tail)


if __name__ == "__main__":
    main()
