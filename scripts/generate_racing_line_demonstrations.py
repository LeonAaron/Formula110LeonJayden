"""Log the racing-line-follower expert's (sensors -> command) pairs as BC data.

Unlike scripts/generate_expert_demonstrations.py (which clones the reactive
controller), this expert tracks the offline-computed minimum-curvature racing
line (scripts/compute_racing_line.py) using the privileged single-car env
(scripts/_privileged_race_env.py) for ground-truth arc-length lookup. The
network is still only ever trained on public sensor features - the racing
line itself is not an input - so the resulting policy should generalize to
tracks it never saw the optimized line for.

Usage:
    uv run python scripts/generate_racing_line_demonstrations.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from _privileged_race_env import run_single_car_headless
from racing_line_follower import DEFAULT_FOLLOWER_PARAMS, RacingLine, follow, load_racing_line

from controllers.learned import build_inputs
from racing.race.progress import TrackProjection
from racing.student.api import RobotCommand, RobotSensors

DEFAULT_SEEDS = tuple(range(4000, 4040))
DEFAULT_OUTPUT = Path("artifacts/racing_line_demonstrations.pt")


def collect_demonstrations(
    *, racing_line: RacingLine, seeds: tuple[int, ...], round_seconds: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Race the racing-line follower across `seeds`, recording every tick's (features, action)."""
    feature_rows: list[tuple[float, ...]] = []
    action_rows: list[tuple[float, float]] = []

    for seed in seeds:
        state = {"true_progress_distance_m": 0.0}

        def controller(sensors: RobotSensors, state: dict[str, float] = state) -> RobotCommand:
            command = follow(
                sensors,
                true_progress_distance_m=state["true_progress_distance_m"],
                racing_line=racing_line,
                params=DEFAULT_FOLLOWER_PARAMS,
            )
            feature_rows.append(build_inputs(sensors))
            action_rows.append((command.throttle, command.steer))
            return command

        def tick_callback(sensors: RobotSensors, projection: TrackProjection, state: dict[str, float] = state) -> None:
            state["true_progress_distance_m"] = projection.progress_distance_m

        run_single_car_headless(controller, seed=seed, round_seconds=round_seconds, tick_callback=tick_callback)

    features = torch.tensor(feature_rows, dtype=torch.float32)
    actions = torch.tensor(action_rows, dtype=torch.float32)
    return features, actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--round-seconds", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    racing_line = load_racing_line()
    features, actions = collect_demonstrations(
        racing_line=racing_line, seeds=tuple(args.seeds), round_seconds=args.round_seconds
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "actions": actions}, args.output)
    print(f"saved {features.shape[0]} ticks from {len(args.seeds)} seeds to {args.output}")


if __name__ == "__main__":
    main()
