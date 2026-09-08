"""Try ReactiveParams overrides against the held-out seed suite, without editing reactive_v3.py.

v3 counterpart of scripts/try_params.py, pointed at controllers.reactive_v3
(off-axis wall detection) instead of controllers.reactive.

Every field of ReactiveParams is exposed as a CLI flag (underscores become
dashes). Unspecified fields keep their current DEFAULT_PARAMS value. This is
for fast, one-variable-at-a-time manual tuning: test a change, see all 5 seeds
immediately, and only hand-edit reactive_v3.py once a value is confirmed safe.

Usage:
    uv run python scripts/try_params_v3.py --wall-scan-cone-deg 30.0
    uv run python scripts/try_params_v3.py  # no overrides: re-checks DEFAULT_PARAMS as-is
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from statistics import mean

from controllers.reactive_v3 import DEFAULT_PARAMS, ReactiveParams, drive
from racing import RobotCommand, RobotSensors, run_headless_head_to_head

HELD_OUT_SEEDS = (42, 110, 271, 997, 2027)


def _passive_controller(sensors: RobotSensors) -> RobotCommand:
    """Coast in place so the challenger's solo run is barely disturbed."""
    return RobotCommand(throttle=0.0, steer=0.0)


def make_controller(params: ReactiveParams):
    """Build a callable controller bound to one parameter set."""

    def control(sensors: RobotSensors) -> RobotCommand:
        return drive(sensors, params)

    return control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in fields(ReactiveParams):
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=float, default=None)
    parser.add_argument("--races", type=int, default=5, help="Races per seed")
    parser.add_argument("--round-seconds", type=float, default=30.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(HELD_OUT_SEEDS))
    args = parser.parse_args()

    overrides = {
        field.name: getattr(args, field.name)
        for field in fields(ReactiveParams)
        if getattr(args, field.name) is not None
    }
    params = replace(DEFAULT_PARAMS, **overrides)
    print("Overrides:", overrides if overrides else "(none — testing DEFAULT_PARAMS as-is)")

    overall_distances: list[float] = []
    overall_survived = 0
    total_races = 0
    for seed in args.seeds:
        result = run_headless_head_to_head(
            challenger_controller=make_controller(params),
            incumbent_controller=_passive_controller,
            race_count=args.races,
            round_seconds=args.round_seconds,
            random_seed=seed,
        )
        stats = [race.challenger for race in result.races]
        distances = [s.distances_m[0] for s in stats]
        damages = [s.damages[0] for s in stats]
        survived = sum(1 for damage in damages if damage < 1.0)
        overall_survived += survived
        total_races += len(stats)
        overall_distances.extend(distances)
        print(
            f"seed {seed}: survived {survived}/{len(stats)}, "
            f"avg_dist={mean(distances):.1f}m, damages={[round(damage, 2) for damage in damages]}"
        )

    print("-" * 60)
    print(
        f"TOTAL: survived {overall_survived}/{total_races}, "
        f"avg_dist={mean(overall_distances):.1f}m, max_dist={max(overall_distances):.1f}m"
    )


if __name__ == "__main__":
    main()
