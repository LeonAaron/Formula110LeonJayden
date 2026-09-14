"""Log the reactive controller's (sensors -> command) pairs as behavior-cloning data.

Runs `controllers.reactive.drive()` headless across many seeds (disjoint from
the held-out evaluation suite and the reactive/neuro training seeds) and
records one (features, throttle, steer) row per tick, using the learned
controller's own feature encoder so the dataset matches what it sees at
inference. Saved as a torch tensor pair for `scripts/train_behavior_cloning.py`.

Usage:
    uv run python scripts/generate_expert_demonstrations.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from controllers.learned import build_inputs
from controllers.reactive import DEFAULT_PARAMS, drive
from racing import RobotCommand, RobotSensors, run_headless_head_to_head

DEFAULT_SEEDS = tuple(range(3000, 3040))
DEFAULT_OUTPUT = Path("artifacts/expert_demonstrations.pt")


def _passive_controller(sensors: RobotSensors) -> RobotCommand:
    """Coast in place so the challenger's solo run is barely disturbed."""
    return RobotCommand(throttle=0.0, steer=0.0)


def collect_demonstrations(*, seeds: tuple[int, ...], round_seconds: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Race the reactive controller across `seeds`, recording every tick's (features, action)."""
    feature_rows: list[tuple[float, ...]] = []
    action_rows: list[tuple[float, float]] = []

    def traced_control(sensors: RobotSensors) -> RobotCommand:
        command = drive(sensors, DEFAULT_PARAMS)
        feature_rows.append(build_inputs(sensors))
        action_rows.append((command.throttle, command.steer))
        return command

    for seed in seeds:
        run_headless_head_to_head(
            challenger_controller=traced_control,
            incumbent_controller=_passive_controller,
            race_count=1,
            round_seconds=round_seconds,
            random_seed=seed,
        )

    features = torch.tensor(feature_rows, dtype=torch.float32)
    actions = torch.tensor(action_rows, dtype=torch.float32)
    return features, actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--round-seconds", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    features, actions = collect_demonstrations(seeds=tuple(args.seeds), round_seconds=args.round_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "actions": actions}, args.output)
    print(f"saved {features.shape[0]} ticks from {len(args.seeds)} seeds to {args.output}")


if __name__ == "__main__":
    main()
