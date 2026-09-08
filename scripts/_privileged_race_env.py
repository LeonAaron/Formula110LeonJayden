"""Single-car headless stepping loop with ground-truth track position each tick.

Training tooling only - not part of the public student controller API. Reuses
the same internal simulator building blocks as ``racing.race.head_to_head``,
but for exactly one car (no incumbent) and with a per-tick callback that
exposes the true arc-length position (``TrackProjection.progress_distance_m``)
so an offline "racing line follower" expert can look up its target offset and
speed without needing to estimate its own position from noisy sensors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

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
from racing.race.progress import TrackProjection, default_track_progress_model, project_track_position
from racing.race.runtime import (
    RaceCarRuntime,
    lap_progress_tracker_for_spawn_pose,
    race_contact_states,
    race_scored_distance_m,
    race_spawn_poses,
    robot_score_damage,
    robot_track_point,
    update_race_runtime_after_step,
)
from racing.race.sensors import build_robot_sensors
from racing.student.api import RobotController, RobotSensors

TickCallback = Callable[[RobotSensors, TrackProjection], None]


@dataclass(frozen=True, slots=True)
class SingleCarRaceStats:
    """Summary stats for one privileged single-car headless race."""

    scored_distance_m: float
    lap_count: int
    damage: float
    max_speed_mps: float


def run_single_car_headless(
    controller: RobotController,
    *,
    seed: int,
    round_seconds: float,
    fixed_delta_seconds: float = 1 / 60,
    tick_callback: TickCallback | None = None,
) -> SingleCarRaceStats:
    """Race one car alone (no incumbent) headless, exposing ground-truth position per tick."""
    model = default_track_progress_model()
    configure_headless_panda()
    showbase = cast(Any, import_module("direct.showbase.ShowBase"))
    base = showbase.ShowBase(windowType="none")
    try:
        physics_world = create_physics_world()
        physics_scene = PhysicsScene(world=physics_world, vehicles=[])
        root = base.render.attachNewNode("privileged-single-car")
        add_racing_scene_collisions(physics_world=physics_world, render=root)
        try:
            spawn_pose = race_spawn_poses(
                1, model=model, config=FORMULA_VEHICLE_PHYSICS_CONFIG, random_seed=seed, race_index=1
            )[0]
            robot = create_robot_vehicle(
                world=physics_world,
                render=root,
                name="privileged-single-car",
                position=spawn_pose.position,
                heading_degrees=spawn_pose.heading_degrees,
                config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            )
            physics_scene.vehicles.append(robot)
            runtime = RaceCarRuntime(
                robot=robot, tracker=lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=spawn_pose)
            )

            elapsed_seconds = 0.0
            while elapsed_seconds < round_seconds and not bool(getattr(robot, "eliminated", False)):
                sensors, runtime.sensor_state = build_robot_sensors(
                    physics_world=physics_world,
                    robot=robot,
                    track_model=model,
                    time_s=elapsed_seconds,
                    dt_s=fixed_delta_seconds,
                    previous_state=runtime.sensor_state,
                )
                projection = project_track_position(model, robot_track_point(robot))
                if tick_callback is not None:
                    tick_callback(sensors, projection)
                apply_robot_vehicle_command(robot=robot, command=controller(sensors))

                physics_scene.step(fixed_delta_seconds)
                elapsed_seconds += fixed_delta_seconds
                contact_state = race_contact_states(physics_world=physics_world, runtimes=(runtime,))[0]
                apply_wall_impact_damage(
                    physics_world=physics_world, robots=(robot,), fixed_time_step=physics_scene.fixed_time_step
                )
                projection = project_track_position(model, robot_track_point(robot))
                update_race_runtime_after_step(
                    runtime=runtime,
                    projection=projection,
                    contact_state=contact_state,
                    elapsed_seconds=elapsed_seconds,
                    delta_seconds=fixed_delta_seconds,
                )

            return SingleCarRaceStats(
                scored_distance_m=race_scored_distance_m(runtime),
                lap_count=runtime.tracker.lap_count,
                damage=robot_score_damage(robot),
                max_speed_mps=runtime.max_speed_mps,
            )
        finally:
            root.removeNode()
    finally:
        base.destroy()
