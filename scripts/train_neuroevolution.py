"""Minimum-viable neuroevolution experiment for controllers.neuro's MLP weights.

A simple (mu + lambda) evolution strategy: each generation keeps the top
`--elite` genomes unchanged and refills the rest of the population by
Gaussian-mutating a randomly chosen elite. Fitness is measured with the same
headless harness used to evaluate the reactive controller
(scripts/evaluate_controller.py), so results are directly comparable between
approaches.

Training searches on seeds distinct from the recommended evaluation suite
(42, 110, 271, 997, 2027) so scripts/evaluate_controller.py can validate the
best genome on genuinely held-out starting positions afterward.

Usage:
    uv run python scripts/train_neuroevolution.py
"""

from __future__ import annotations

import argparse
import random
from statistics import mean

from controllers.neuro import GENOME_SIZE, Genome, drive
from racing import RobotCommand, RobotSensors, run_headless_head_to_head

ELIMINATION_PENALTY_M = 350.0
IDLE_DISTANCE_M = 10.0
IDLE_PENALTY_M = 20.0


def _passive_controller(sensors: RobotSensors) -> RobotCommand:
    """Coast in place so the challenger's solo run is barely disturbed."""
    return RobotCommand(throttle=0.0, steer=0.0)


def make_controller(genome: Genome):
    """Build a callable controller bound to one genome."""

    def control(sensors: RobotSensors) -> RobotCommand:
        return drive(sensors, genome)

    return control


def evaluate_genome(genome: Genome, *, seeds: tuple[int, ...], round_seconds: float) -> float:
    """Score a genome by average scored distance across seeds, three-tier:

    - Eliminated: distance minus a large penalty, so covering more ground
      before crashing does not let a reckless genome out-breed a safe one.
      (An earlier version subtracted a smaller, additive-only penalty and
      evolution found a fast, reckless policy that crashed in most held-out
      validation races. See LAB_NOTEBOOK.md, Entry 2.)
    - Idle survivor (distance below `IDLE_DISTANCE_M`): a small penalty, so
      "never move" is not a free zero-fitness local optimum. (A pure
      survival-dominant fitness with no idle penalty converged to genomes
      that stood still or wedged against a wall instead of driving. Also
      documented in Entry 2.)
    - Otherwise: the raw scored distance.
    """
    race_scores: list[float] = []
    for seed in seeds:
        result = run_headless_head_to_head(
            challenger_controller=make_controller(genome),
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


def random_genome(rng: random.Random, sigma: float) -> Genome:
    """Sample a genome with independent Gaussian weights."""
    return tuple(rng.gauss(0.0, sigma) for _ in range(GENOME_SIZE))


def mutate(genome: Genome, rng: random.Random, sigma: float) -> Genome:
    """Return a copy of a genome with independent Gaussian mutation noise added."""
    return tuple(gene + rng.gauss(0.0, sigma) for gene in genome)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--init-sigma", type=float, default=0.6)
    parser.add_argument("--mutation-sigma", type=float, default=0.35)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[13, 55],
        help="Training seeds, kept distinct from the held-out evaluation suite",
    )
    parser.add_argument("--round-seconds", type=float, default=20.0)
    parser.add_argument("--rng-seed", type=int, default=1, help="Seed for the evolutionary search's own randomness")
    args = parser.parse_args()

    rng = random.Random(args.rng_seed)
    population = [random_genome(rng, args.init_sigma) for _ in range(args.population)]

    best_genome: Genome | None = None
    best_fitness = float("-inf")

    for generation in range(args.generations):
        scored = sorted(
            (
                (evaluate_genome(genome, seeds=tuple(args.seeds), round_seconds=args.round_seconds), genome)
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

        elites = [genome for _, genome in scored[: args.elite]]
        child_count = args.population - len(elites)
        children = [mutate(elites[rng.randrange(len(elites))], rng, args.mutation_sigma) for _ in range(child_count)]
        population = [*elites, *children]

    assert best_genome is not None
    print("-" * 72)
    print(f"best training fitness: {best_fitness:.1f}m (seeds {args.seeds}, {args.round_seconds:.0f}s rounds)")
    print("BEST_GENOME: Genome = (")
    for value in best_genome:
        print(f"    {value!r},")
    print(")")


if __name__ == "__main__":
    main()
