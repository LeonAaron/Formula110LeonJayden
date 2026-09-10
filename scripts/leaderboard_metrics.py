"""Reproduce the Gradescope leaderboard trial locally for one controller module.

Mirrors ``autograder/gradescope/race_worker.py::run_trial``: a single car, seeds
110 and 2026, 30 simulated seconds at 60 Hz, marshal recovery disabled, and
reports partial laps, top speed, first-lap time and best-lap time per seed plus
their averages (the leaderboard values).

Usage:
    uv run python scripts/leaderboard_metrics.py --module controllers.apex
"""

from __future__ import annotations

import argparse
from importlib import import_module
from typing import Any, cast

from racing import load_student_submission
from racing.graphics.panda_config import configure_headless_panda
from racing.graphics.track_rendering import add_racing_scene_collisions
from racing.physics import (
    FORMULA_VEHICLE_PHYSICS_CONFIG,
    PhysicsScene,
    apply_robot_vehicle_command,
    apply_wall_impact_damage,
    create_physics_world,
    create_robot_vehicle,
)
from racing.race.progress import default_track_progress_model, project_track_position
from racing.race.runtime import (
    RaceCarRuntime,
    lap_progress_tracker_for_spawn_pose,
    race_contact_states,
    race_spawn_poses,
    robot_is_eliminated,
    robot_score_damage,
    robot_track_point,
    update_race_runtime_after_step,
)
from racing.race.sensors import build_robot_sensors
from racing.student.api import RobotController

LEADERBOARD_SEEDS = (110, 2026)


def run_trial(controller: RobotController, *, seed: int, seconds: float = 30.0) -> dict[str, Any]:
    fixed_delta_seconds = 1.0 / 60.0
    configure_headless_panda()
    showbase = cast(Any, import_module("direct.showbase.ShowBase"))
    base = showbase.ShowBase(windowType="none")
    root: Any | None = None
    try:
        model = default_track_progress_model()
        physics_world = create_physics_world()
        physics_scene = PhysicsScene(world=physics_world, vehicles=[])
        root = base.render.attachNewNode(f"leaderboard-{seed}")
        add_racing_scene_collisions(physics_world=physics_world, render=root)
        spawn_pose = race_spawn_poses(
            1, model=model, config=FORMULA_VEHICLE_PHYSICS_CONFIG, random_seed=seed, race_index=1
        )[0]
        robot = create_robot_vehicle(
            world=physics_world,
            render=root,
            name=f"leaderboard-{seed}-car",
            position=spawn_pose.position,
            heading_degrees=spawn_pose.heading_degrees,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        )
        physics_scene.vehicles.append(robot)
        runtime = RaceCarRuntime(
            robot=robot, tracker=lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=spawn_pose)
        )
        elapsed_seconds = 0.0
        lap_crossing_times: list[float] = []
        previous_lap_count = 0
        while elapsed_seconds < seconds:
            if not robot_is_eliminated(runtime.robot):
                sensors, runtime.sensor_state = build_robot_sensors(
                    physics_world=physics_world,
                    robot=runtime.robot,
                    track_model=model,
                    time_s=elapsed_seconds,
                    dt_s=fixed_delta_seconds,
                    previous_state=runtime.sensor_state,
                )
                apply_robot_vehicle_command(robot=runtime.robot, command=controller(sensors))
            physics_scene.step(fixed_delta_seconds)
            next_elapsed_seconds = min(seconds, elapsed_seconds + fixed_delta_seconds)
            contact_state = race_contact_states(physics_world=physics_world, runtimes=(runtime,))[0]
            apply_wall_impact_damage(
                physics_world=physics_world, robots=(runtime.robot,), fixed_time_step=physics_scene.fixed_time_step
            )
            projection = project_track_position(model, robot_track_point(runtime.robot))
            update_race_runtime_after_step(
                runtime=runtime,
                projection=projection,
                contact_state=contact_state,
                elapsed_seconds=next_elapsed_seconds,
                delta_seconds=fixed_delta_seconds,
            )
            while previous_lap_count < runtime.tracker.lap_count:
                lap_crossing_times.append(next_elapsed_seconds)
                previous_lap_count += 1
            elapsed_seconds = next_elapsed_seconds
        lap_durations = [
            crossing - (lap_crossing_times[index - 1] if index else 0.0)
            for index, crossing in enumerate(lap_crossing_times)
        ]
        damage = robot_score_damage(runtime.robot)
        return {
            "seed": seed,
            "raw_distance_m": runtime.tracker.best_distance_m,
            "partial_laps": runtime.tracker.best_distance_m / model.total_length_m,
            "lap_count": runtime.tracker.lap_count,
            "damage": damage,
            "survived": not robot_is_eliminated(runtime.robot) and damage < 1.0,
            "wall_contact_seconds": runtime.tracker.wall_contact_seconds,
            "max_speed_mps": runtime.max_speed_mps,
            "first_lap_time_seconds": lap_crossing_times[0] if lap_crossing_times else None,
            "best_lap_time_seconds": min(lap_durations) if lap_durations else None,
        }
    finally:
        if root is not None:
            root.removeNode()
        base.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="controllers.apex")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(LEADERBOARD_SEEDS))
    args = parser.parse_args()
    submission = load_student_submission(args.module)
    trials = [run_trial(submission.controller, seed=seed) for seed in args.seeds]
    for t in trials:
        first = f"{t['first_lap_time_seconds']:.3f}" if t["first_lap_time_seconds"] is not None else "No lap"
        best = f"{t['best_lap_time_seconds']:.3f}" if t["best_lap_time_seconds"] is not None else "No lap"
        print(
            f"seed {t['seed']}: {t['partial_laps']:.3f} laps ({t['raw_distance_m']:.1f} m), top speed {t['max_speed_mps']:.2f} m/s, "
            f"first lap {first} s, best lap {best} s, damage {t['damage'] * 100:.1f}%, wall contact {t['wall_contact_seconds']:.3f} s, "
            f"{'survived' if t['survived'] else 'ELIMINATED'}"
        )
    if all(t["survived"] for t in trials):
        n = len(trials)
        firsts = [t["first_lap_time_seconds"] for t in trials if t["first_lap_time_seconds"] is not None]
        bests = [t["best_lap_time_seconds"] for t in trials if t["best_lap_time_seconds"] is not None]
        print("Leaderboard (average of seeds):")
        print(f"  Laps (partial): {sum(t['partial_laps'] for t in trials) / n:.4f}")
        print(f"  Top speed (m/s): {sum(t['max_speed_mps'] for t in trials) / n:.3f}")
        print(f"  First lap (s): {sum(firsts) / len(firsts):.3f}" if len(firsts) == n else "  First lap (s): No lap")
        print(f"  Best lap (s): {sum(bests) / len(bests):.3f}" if len(bests) == n else "  Best lap (s): No lap")
    else:
        print("Leaderboard: DQ — the controller did not finish both 30-second seeded runs.")


if __name__ == "__main__":
    main()
