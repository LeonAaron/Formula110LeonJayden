"""Evaluate a student controller's solo performance across seeded starts.

Runs the controller headless against a passive (do-nothing) baseline so the
opponent barely interferes, then reports scored/raw distance, laps, damage,
and marshal activity for each seeded starting position.

Usage:
    uv run python scripts/evaluate_controller.py --module controllers.reactive
"""

from __future__ import annotations

import argparse

from racing import RobotCommand, RobotSensors, load_student_submission, run_headless_head_to_head


def _passive_controller(sensors: RobotSensors) -> RobotCommand:
    """Coast in place so the challenger's solo run is barely disturbed."""
    return RobotCommand(throttle=0.0, steer=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="controllers.reactive", help="Dotted module path or file path")
    parser.add_argument("--seed", type=int, default=110, help="Base random seed for deterministic starts")
    parser.add_argument("--races", type=int, default=5, help="Number of seeded starting positions to test")
    parser.add_argument("--round-seconds", type=float, default=30.0, help="Race duration in seconds")
    args = parser.parse_args()

    submission = load_student_submission(args.module)
    result = run_headless_head_to_head(
        challenger_controller=submission.controller,
        incumbent_controller=_passive_controller,
        challenger_name=submission.display_name or args.module,
        incumbent_name="passive-baseline",
        race_count=args.races,
        round_seconds=args.round_seconds,
        random_seed=args.seed,
    )

    eliminated_count = 0
    lap_completed_count = 0
    scored_distances_m: list[float] = []

    for race in result.races:
        stats = race.challenger
        scored_m = stats.distances_m[0]
        raw_m = stats.raw_distances_m[0]
        laps = stats.lap_counts[0]
        damage = stats.damages[0]
        eliminated = damage >= 1.0
        max_speed = stats.max_speeds_mps[0]
        marshal_count = stats.marshal_counts[0]
        marshal_penalty = stats.marshal_penalties_m[0]

        scored_distances_m.append(scored_m)
        eliminated_count += int(eliminated)
        lap_completed_count += int(laps >= 1)

        status = "ELIMINATED" if eliminated else "survived"
        print(
            f"race {race.race_index}: {status} | scored={scored_m:6.1f}m raw={raw_m:6.1f}m "
            f"laps={laps} damage={damage:.2f} max_speed={max_speed:5.1f}m/s "
            f"marshal={marshal_count}x({marshal_penalty:.1f}m)"
        )

    race_count = len(result.races)
    avg_scored_m = sum(scored_distances_m) / race_count
    print("-" * 72)
    print(
        f"summary: {race_count - eliminated_count}/{race_count} survived, "
        f"{lap_completed_count}/{race_count} completed >=1 lap, "
        f"avg scored distance {avg_scored_m:.1f}m"
    )


if __name__ == "__main__":
    main()
