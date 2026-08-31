"""Parameter-search and diagnostic tool for controllers.reactive.ReactiveParams.

Optimization: a simple (mu + lambda) evolution strategy, seeded from the
current DEFAULT_PARAMS (a known-safe baseline) rather than random init, since
a from-scratch random search over throttle/steer gains risks wasting most of
its budget on unstable genomes (see the neuroevolution experiment's random-init
struggles in LAB_NOTEBOOK.md, Entry 2). Each parameter is perturbed by noise
scaled to its own magnitude, so one shared step size works across gains,
angles, and distances of very different scale.

Fitness: same three-tier shape used for the neuroevolution trainer (distance,
minus a large penalty on elimination, minus a small penalty for going
nowhere), so a genome cannot out-score a safe driver just by crashing fast, or
out-score a real driver just by idling.

Diagnostics: --diagnose reruns one race with full per-tick tracing (sensors +
command) and prints, in order:
  - a summary (distance, laps, damage, marshal activity, max speed)
  - how often the car braked and how close it got to walls
  - the sensor/command state at the moment of first wall contact, if any
  - the last several ticks before the race ended, if the car was eliminated

This is meant to answer "why did this genome do well or poorly", using
concrete sensor values, the throttle/steer output, and the simulated time at
which the event happened - not just a fitness number.

Usage:
    uv run python scripts/optimize_reactive.py --population 10 --generations 10
    uv run python scripts/optimize_reactive.py --diagnose --seed 110
"""

from __future__ import annotations

import argparse
import random
from dataclasses import fields, replace
from statistics import mean

from controllers.reactive import DEFAULT_PARAMS, ReactiveParams, drive
from racing import HeadToHeadTeamRaceStats, RobotCommand, RobotSensors, run_headless_head_to_head

ELIMINATION_PENALTY_M = 350.0
IDLE_DISTANCE_M = 10.0
IDLE_PENALTY_M = 20.0

PARAM_NAMES: tuple[str, ...] = tuple(field.name for field in fields(ReactiveParams))
Genome = tuple[float, ...]


def genome_from_params(params: ReactiveParams) -> Genome:
    """Flatten a ReactiveParams into a genome tuple, in dataclass field order."""
    return tuple(getattr(params, name) for name in PARAM_NAMES)


def params_from_genome(genome: Genome) -> ReactiveParams:
    """Rebuild a ReactiveParams from a genome tuple."""
    return replace(DEFAULT_PARAMS, **dict(zip(PARAM_NAMES, genome, strict=True)))


def mutate(genome: Genome, rng: random.Random, sigma_fraction: float) -> Genome:
    """Perturb each gene by Gaussian noise scaled to that gene's own magnitude."""
    mutated: list[float] = []
    for value in genome:
        step_sigma = max(abs(value), 0.05) * sigma_fraction
        mutated.append(value + rng.gauss(0.0, step_sigma))
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
    """Score a parameter set by average scored distance across seeds (three-tier)."""
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
        eliminated = stats.damages[0] >= 1.0
        if eliminated:
            race_scores.append(distance_m - ELIMINATION_PENALTY_M)
        elif distance_m < IDLE_DISTANCE_M:
            race_scores.append(distance_m - IDLE_PENALTY_M)
        else:
            race_scores.append(distance_m)
    return mean(race_scores)


def run_search(
    *,
    population_size: int,
    generations: int,
    elite_count: int,
    mutation_sigma_fraction: float,
    seeds: tuple[int, ...],
    round_seconds: float,
    rng_seed: int,
) -> tuple[ReactiveParams, float]:
    """Run the evolution strategy; return the best params found and their fitness."""
    rng = random.Random(rng_seed)
    base_genome = genome_from_params(DEFAULT_PARAMS)
    population = [base_genome] + [
        mutate(base_genome, rng, mutation_sigma_fraction) for _ in range(population_size - 1)
    ]

    best_genome = base_genome
    best_fitness = float("-inf")

    for generation in range(generations):
        scored = sorted(
            (
                (evaluate_params(params_from_genome(genome), seeds=seeds, round_seconds=round_seconds), genome)
                for genome in population
            ),
            key=lambda item: item[0],
            reverse=True,
        )
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
        default=[13, 55],
        help="Training seeds, kept distinct from the held-out evaluation suite",
    )
    parser.add_argument("--round-seconds", type=float, default=20.0)
    parser.add_argument("--rng-seed", type=int, default=1, help="Seed for the evolutionary search's own randomness")
    parser.add_argument("--diagnose", action="store_true", help="Skip search; trace DEFAULT_PARAMS on --seed instead")
    parser.add_argument("--seed", type=int, default=110, help="Seed used by --diagnose")
    parser.add_argument("--diagnose-round-seconds", type=float, default=30.0)
    parser.add_argument("--tail", type=int, default=15, help="Ticks to print before an eliminated trace ends")
    args = parser.parse_args()

    if args.diagnose:
        trace, stats = trace_race(DEFAULT_PARAMS, seed=args.seed, round_seconds=args.diagnose_round_seconds)
        summarize_trace(trace, stats, seed=args.seed, tail=args.tail)
        return

    best_params, best_fitness = run_search(
        population_size=args.population,
        generations=args.generations,
        elite_count=args.elite,
        mutation_sigma_fraction=args.mutation_sigma_fraction,
        seeds=tuple(args.seeds),
        round_seconds=args.round_seconds,
        rng_seed=args.rng_seed,
    )

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
