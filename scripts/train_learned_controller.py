"""Dense-reward CMA-ES fine-tune for the learned controller's PolicyNet weights.

Starts from the behavior-cloned weights (a known-good driver, not random init
- the lesson carried over from LAB_NOTEBOOK.md Entry 2/3) and searches with
real covariance-adaptation CMA-ES (the `cma` package) instead of the
hand-rolled isotropic-mutation ES used elsewhere in this repo.

Fitness is a *dense*, per-tick shaped reward accumulated over the whole
episode (forward progress each tick, minus a damage-rate penalty, minus a
small idle penalty), not a single end-of-episode number - so the search gets
signal about *when* things went wrong, not just how the episode ended.

Runs in fixed wall-clock chunks (--time-budget-seconds, default 300 = 5
minutes) and checkpoints the best genome after every generation, so a run can
be stopped and evaluated at any point and resumed later with --init.

Usage:
    uv run python scripts/train_learned_controller.py --init artifacts/bc_policy.pt
    uv run python scripts/train_learned_controller.py --init artifacts/learned_policy_best.pt --time-budget-seconds 300
"""

from __future__ import annotations

import argparse
import time
from math import cos, radians
from pathlib import Path

import cma
import torch

from controllers.learned import PolicyNet, build_inputs, get_flat_params, set_flat_params
from racing import RobotCommand, RobotSensors, run_headless_head_to_head

DEFAULT_INIT = Path("artifacts/bc_policy.pt")
DEFAULT_OUTPUT = Path("artifacts/learned_policy_best.pt")
DEFAULT_TRAINING_SEEDS = (21, 34, 63, 88)

IDLE_SPEED_THRESHOLD_MPS = 0.5
IDLE_PENALTY_PER_S = 0.5
DAMAGE_PENALTY_WEIGHT = 200.0
ELIMINATION_PENALTY = 20.0


def _passive_controller(sensors: RobotSensors) -> RobotCommand:
    """Coast in place so the challenger's solo run is barely disturbed."""
    return RobotCommand(throttle=0.0, steer=0.0)


def _dense_episode_reward(model: PolicyNet, *, seed: int, round_seconds: float) -> float:
    """Race one episode, returning the summed per-tick shaped reward."""
    state = {"reward": 0.0, "last_damage": 0.0}

    def control(sensors: RobotSensors) -> RobotCommand:
        inputs = torch.tensor(build_inputs(sensors), dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            throttle, steer = model(inputs).squeeze(0).tolist()

        dt_s = sensors.dt_s
        damage_delta = max(0.0, sensors.contact.damage - state["last_damage"])
        state["last_damage"] = sensors.contact.damage
        # Progress along the track heading, not raw chassis speed - matches the
        # centerline-projection scoring rule instead of rewarding fast-but-misaligned driving.
        aligned_speed_mps = sensors.odometry.speed_mps * cos(radians(sensors.camera.heading_error_degrees))
        state["reward"] += aligned_speed_mps * dt_s
        state["reward"] -= DAMAGE_PENALTY_WEIGHT * damage_delta
        if abs(sensors.odometry.speed_mps) < IDLE_SPEED_THRESHOLD_MPS:
            state["reward"] -= IDLE_PENALTY_PER_S * dt_s
        return RobotCommand(throttle=throttle, steer=steer)

    run_headless_head_to_head(
        challenger_controller=control,
        incumbent_controller=_passive_controller,
        race_count=1,
        round_seconds=round_seconds,
        random_seed=seed,
    )
    if state["last_damage"] >= 1.0:
        state["reward"] -= ELIMINATION_PENALTY
    return state["reward"]


def evaluate_genome(
    model: PolicyNet, flat_params: torch.Tensor, *, seeds: tuple[int, ...], round_seconds: float
) -> float:
    """Return -mean(dense reward) across seeds, since CMA-ES minimizes."""
    set_flat_params(model, flat_params)
    rewards = [_dense_episode_reward(model, seed=seed, round_seconds=round_seconds) for seed in seeds]
    return -(sum(rewards) / len(rewards))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT, help="State dict to seed the CMA-ES mean from")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sigma0", type=float, default=0.4, help="Initial CMA-ES step size")
    parser.add_argument("--popsize", type=int, default=None, help="Override CMA-ES population size")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_TRAINING_SEEDS))
    parser.add_argument("--round-seconds", type=float, default=15.0)
    parser.add_argument("--time-budget-seconds", type=float, default=300.0)
    parser.add_argument("--max-generations", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=1)
    args = parser.parse_args()

    model = PolicyNet()
    init_state = torch.load(args.init, weights_only=True)
    model.load_state_dict(init_state)
    x0 = get_flat_params(model).tolist()

    cma_options: dict[str, object] = {"seed": args.rng_seed}
    if args.popsize is not None:
        cma_options["popsize"] = args.popsize
    es = cma.CMAEvolutionStrategy(x0, args.sigma0, cma_options)

    # Seed with the init checkpoint's own fitness so a resumed run can never save a regression.
    best_params = torch.tensor(x0, dtype=torch.float32)
    best_fitness = evaluate_genome(model, best_params, seeds=tuple(args.seeds), round_seconds=args.round_seconds)
    print(f"init checkpoint reward: {-best_fitness:7.1f} (seeds {args.seeds}, {args.round_seconds:.0f}s rounds)")
    start_time = time.monotonic()
    generation = 0

    while (
        time.monotonic() - start_time < args.time_budget_seconds and generation < args.max_generations and not es.stop()
    ):
        solutions = es.ask()
        fitnesses = [
            evaluate_genome(
                model,
                torch.tensor(solution, dtype=torch.float32),
                seeds=tuple(args.seeds),
                round_seconds=args.round_seconds,
            )
            for solution in solutions
        ]
        es.tell(solutions, fitnesses)

        generation_best_index = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        generation_best_fitness = fitnesses[generation_best_index]
        if generation_best_fitness < best_fitness:
            best_fitness = generation_best_fitness
            best_params = torch.tensor(solutions[generation_best_index], dtype=torch.float32)
            set_flat_params(model, best_params)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)

        elapsed_s = time.monotonic() - start_time
        print(
            f"generation {generation}: best_reward={-generation_best_fitness:7.1f} "
            f"mean_reward={-(sum(fitnesses) / len(fitnesses)):7.1f} "
            f"worst_reward={-max(fitnesses):7.1f} elapsed={elapsed_s:5.1f}s"
        )
        generation += 1

    set_flat_params(model, best_params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print("-" * 72)
    print(f"best training reward: {-best_fitness:.1f} over {generation} generations -> saved to {args.output}")


if __name__ == "__main__":
    main()
