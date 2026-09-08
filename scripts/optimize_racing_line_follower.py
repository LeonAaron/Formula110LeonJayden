"""Evolution-strategy search over RacingLineFollowerParams (including flat speed).

Seeded from the current DEFAULT_FOLLOWER_PARAMS (already validated safe at
15.8 m/s - see LAB_NOTEBOOK.md Entry 5), not random init, so the search
explores around a known-good driver instead of risking early-generation
crashes (the same lesson as scripts/optimize_reactive.py).

Uses the privileged single-car env directly (ground-truth arc length, no
incumbent car) for a faster, more accurate fitness signal than the public
head-to-head harness.

Usage:
    uv run python scripts/optimize_racing_line_follower.py --population 12 --generations 12
"""

from __future__ import annotations

import argparse
import random
from dataclasses import fields, replace
from statistics import mean

from _privileged_race_env import run_single_car_headless
from racing_line_follower import DEFAULT_FOLLOWER_PARAMS, RacingLineFollowerParams, follow, load_racing_line

ELIMINATION_PENALTY_M = 350.0
IDLE_DISTANCE_M = 10.0
IDLE_PENALTY_M = 20.0
# Non-fatal damage must cost enough to outweigh the distance gained by clipping a
# wall at speed - without this, the search finds genomes that survive training
# seeds with real damage, which then eliminate on held-out seeds after distillation
# (see LAB_NOTEBOOK.md Entry 12: flat_speed_mps=23.09 was exactly this failure).
DAMAGE_PENALTY_M = 600.0

PARAM_NAMES: tuple[str, ...] = tuple(field.name for field in fields(RacingLineFollowerParams))
Genome = tuple[float, ...]


def genome_from_params(params: RacingLineFollowerParams) -> Genome:
    return tuple(getattr(params, name) for name in PARAM_NAMES)


def params_from_genome(genome: Genome) -> RacingLineFollowerParams:
    return replace(DEFAULT_FOLLOWER_PARAMS, **dict(zip(PARAM_NAMES, genome, strict=True)))


def mutate(genome: Genome, rng: random.Random, sigma_fraction: float) -> Genome:
    mutated: list[float] = []
    for value in genome:
        step_sigma = max(abs(value), 0.05) * sigma_fraction
        mutated.append(value + rng.gauss(0.0, step_sigma))
    return tuple(mutated)


def evaluate_params(
    params: RacingLineFollowerParams, *, racing_line, seeds: tuple[int, ...], round_seconds: float
) -> float:
    """Score a parameter set by average scored distance across seeds (three-tier)."""
    race_scores: list[float] = []
    for seed in seeds:
        state = {"true_progress_distance_m": 0.0}

        def controller(sensors, state: dict[str, float] = state) -> object:
            return follow(
                sensors, true_progress_distance_m=state["true_progress_distance_m"], racing_line=racing_line,
                params=params,
            )

        def tick_callback(sensors, projection, state: dict[str, float] = state) -> None:
            state["true_progress_distance_m"] = projection.progress_distance_m

        stats = run_single_car_headless(controller, seed=seed, round_seconds=round_seconds, tick_callback=tick_callback)
        eliminated = stats.damage >= 1.0
        if eliminated:
            race_scores.append(stats.scored_distance_m - ELIMINATION_PENALTY_M)
        elif stats.scored_distance_m < IDLE_DISTANCE_M:
            race_scores.append(stats.scored_distance_m - IDLE_PENALTY_M)
        else:
            race_scores.append(stats.scored_distance_m - DAMAGE_PENALTY_M * stats.damage)
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
) -> tuple[RacingLineFollowerParams, float]:
    racing_line = load_racing_line()
    rng = random.Random(rng_seed)
    base_genome = genome_from_params(DEFAULT_FOLLOWER_PARAMS)
    population = [base_genome] + [
        mutate(base_genome, rng, mutation_sigma_fraction) for _ in range(population_size - 1)
    ]

    best_genome = base_genome
    best_fitness = evaluate_params(DEFAULT_FOLLOWER_PARAMS, racing_line=racing_line, seeds=seeds, round_seconds=round_seconds)
    print(f"init fitness: {best_fitness:7.1f}m")

    for generation in range(generations):
        scored = sorted(
            (
                (
                    evaluate_params(
                        params_from_genome(genome), racing_line=racing_line, seeds=seeds, round_seconds=round_seconds
                    ),
                    genome,
                )
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


def print_params_literal(params: RacingLineFollowerParams) -> None:
    print("RacingLineFollowerParams(")
    for name in PARAM_NAMES:
        print(f"    {name}={getattr(params, name)!r},")
    print(")")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--mutation-sigma-fraction", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 55, 87])
    parser.add_argument("--round-seconds", type=float, default=20.0)
    parser.add_argument("--rng-seed", type=int, default=1)
    args = parser.parse_args()

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
    print(f"best fitness: {best_fitness:.1f}m")
    print_params_literal(best_params)


if __name__ == "__main__":
    main()
