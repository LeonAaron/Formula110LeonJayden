"""Parallel multi-seed evaluation of a controller, mirroring scripts/evaluate_controller.py.

Each seed runs in its own worker process using the same public head-to-head
harness with a passive incumbent, so the per-race numbers are identical to
``evaluate_controller.py`` — this just runs the seed suite concurrently.

Usage:
    uv run python scripts/eval_parallel.py --module controllers.apex
    uv run python scripts/eval_parallel.py --module controllers.apex --seeds 42 110 271 997 2027 --races 5
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

DEFAULT_SEEDS = (42, 110, 271, 997, 2027, 2026)


@dataclass(frozen=True, slots=True)
class RaceRow:
    seed: int
    race_index: int
    scored_m: float
    raw_m: float
    laps: int
    partial_laps: float
    damage: float
    max_speed_mps: float
    marshal_count: int
    wall_contact_s: float
    eliminated: bool


def _passive_controller(sensors: object) -> object:
    from racing import RobotCommand

    return RobotCommand(throttle=0.0, steer=0.0)


def run_seed(args: tuple[str, int, int, float]) -> list[dict[str, object]]:
    module, seed, races, round_seconds = args
    from racing import load_student_submission, run_headless_head_to_head
    from racing.race.progress import default_track_progress_model

    lap_length = default_track_progress_model().total_length_m
    submission = load_student_submission(module)
    result = run_headless_head_to_head(
        challenger_controller=submission.controller,
        incumbent_controller=_passive_controller,
        challenger_name=submission.display_name or module,
        incumbent_name="passive-baseline",
        race_count=races,
        round_seconds=round_seconds,
        random_seed=seed,
    )
    rows: list[dict[str, object]] = []
    for race in result.races:
        stats = race.challenger
        rows.append(
            asdict(
                RaceRow(
                    seed=seed,
                    race_index=race.race_index,
                    scored_m=stats.distances_m[0],
                    raw_m=stats.raw_distances_m[0],
                    laps=stats.lap_counts[0],
                    partial_laps=stats.raw_distances_m[0] / lap_length,
                    damage=stats.damages[0],
                    max_speed_mps=stats.max_speeds_mps[0],
                    marshal_count=stats.marshal_counts[0],
                    wall_contact_s=stats.wall_contact_seconds[0],
                    eliminated=stats.damages[0] >= 1.0,
                )
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="controllers.apex")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--races", type=int, default=5)
    parser.add_argument("--round-seconds", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print a JSON summary line at the end")
    args = parser.parse_args()

    jobs = [(args.module, seed, args.races, args.round_seconds) for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        per_seed = list(pool.map(run_seed, jobs))

    rows = [row for seed_rows in per_seed for row in seed_rows]
    for row in rows:
        status = "ELIMINATED" if row["eliminated"] else "survived"
        print(
            f"seed {row['seed']:5d} race {row['race_index']}: {status:10s} scored={row['scored_m']:6.1f}m "
            f"laps={row['partial_laps']:.2f} damage={row['damage']:.2f} max_speed={row['max_speed_mps']:5.1f} "
            f"marshal={row['marshal_count']} wall_contact={row['wall_contact_s']:.2f}s"
        )
    n = len(rows)
    avg = sum(float(r["scored_m"]) for r in rows) / n
    worst = min(float(r["scored_m"]) for r in rows)
    survived = sum(1 for r in rows if not r["eliminated"])
    avg_damage = sum(float(r["damage"]) for r in rows) / n
    lap_length = float(rows[0]["raw_m"]) / float(rows[0]["partial_laps"]) if float(rows[0]["partial_laps"]) > 0 else 1.0
    print("-" * 78)
    print(
        f"summary: {survived}/{n} survived | avg scored {avg:.1f}m ({avg / lap_length:.2f} laps) | "
        f"worst {worst:.1f}m ({worst / lap_length:.2f} laps) | avg damage {avg_damage:.3f}"
    )
    if args.json:
        print(json.dumps({"avg_scored_m": avg, "worst_scored_m": worst, "survived": survived, "n": n}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
