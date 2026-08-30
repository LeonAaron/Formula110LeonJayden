
"""Reactive rule-based controller: sensor-driven throttle and steering.

All tunable gains and thresholds live in ``ReactiveParams`` so a future
parameter-search step (random search, Optuna, CMA-ES, ...) can explore the
knob space by calling :func:`drive` with different parameter sets, without
touching the control logic itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from racing import LidarSensors, RobotCommand, RobotSensors

RACING_NAME = "Reactive Pilot"
RACING_COLOR = "#39D98A"


@dataclass(frozen=True, slots=True)
class ReactiveParams:
    """Tunable gains and thresholds for the reactive controller."""

    # Steering: follow the track centerline and upcoming curvature.
    center_offset_gain: float = 0.10
    heading_error_gain: float = 0.012
    lookahead_near_gain: float = 0.05
    lookahead_far_gain: float = 0.02

    # Steering: nudge away from a close side wall before it becomes urgent.
    wall_avoid_margin_m: float = 3.0
    wall_avoid_gain: float = 0.30

    # Steering: override toward open space when a wall is immediately ahead.
    emergency_front_m: float = 1.6
    emergency_steer: float = 0.9

    # Steering: back away from active contact toward whichever side is open.
    recovery_steer: float = 0.7
    recovery_throttle: float = -0.45

    steer_limit: float = 0.95

    # Throttle: slow down for sharper upcoming turns.
    max_speed_mps: float = 14.0
    min_corner_speed_mps: float = 4.5
    corner_heading_error_deg: float = 50.0
    speed_gain: float = 0.6

    # Throttle: brake in time for the wall directly ahead.
    brake_lead_time_s: float = 1.1
    brake_min_distance_m: float = 2.2
    brake_gain: float = 1.0


DEFAULT_PARAMS = ReactiveParams()


def control(sensors: RobotSensors) -> RobotCommand:
    """Map one sensor snapshot to a throttle/steer command."""
    return drive(sensors, DEFAULT_PARAMS)


def drive(sensors: RobotSensors, params: ReactiveParams) -> RobotCommand:
    """Pure sensor-to-command mapping, parameterized for tuning."""
    if sensors.contact.any_contact > 0.0:
        return _recovery_command(sensors, params)

    steer = _steering_command(sensors, params)
    throttle = _throttle_command(sensors, params)
    return RobotCommand(throttle=_clamp(throttle, -1.0, 1.0), steer=_clamp(steer, -1.0, 1.0))


def _recovery_command(sensors: RobotSensors, params: ReactiveParams) -> RobotCommand:
    wall = sensors.wall_lidar
    open_side = -1.0 if wall.left_m > wall.right_m else 1.0
    return RobotCommand(throttle=params.recovery_throttle, steer=open_side * params.recovery_steer)


def _steering_command(sensors: RobotSensors, params: ReactiveParams) -> float:
    camera = sensors.camera
    wall = sensors.wall_lidar

    steer = 0.0
    if camera.visible:
        steer += params.center_offset_gain * camera.center_offset_m
        steer += params.heading_error_gain * camera.heading_error_degrees
        offsets = camera.lookahead_offsets_m
        if len(offsets) >= 1:
            steer += params.lookahead_near_gain * offsets[0]
        if len(offsets) >= 2:
            steer += params.lookahead_far_gain * offsets[1]

    steer += _wall_avoidance_term(wall, params)

    if wall.front_m < params.emergency_front_m:
        emergency_side = -1.0 if wall.front_left_m > wall.front_right_m else 1.0
        steer += emergency_side * params.emergency_steer

    return _clamp(steer, -params.steer_limit, params.steer_limit)


def _wall_avoidance_term(wall: LidarSensors, params: ReactiveParams) -> float:
    term = 0.0
    if wall.left_m < params.wall_avoid_margin_m:
        term += params.wall_avoid_gain * (params.wall_avoid_margin_m - wall.left_m)
    if wall.right_m < params.wall_avoid_margin_m:
        term -= params.wall_avoid_gain * (params.wall_avoid_margin_m - wall.right_m)
    return term


def _throttle_command(sensors: RobotSensors, params: ReactiveParams) -> float:
    speed_mps = sensors.odometry.speed_mps
    wall = sensors.wall_lidar
    camera = sensors.camera

    brake_distance_m = max(params.brake_min_distance_m, speed_mps * params.brake_lead_time_s)
    if wall.front_m < brake_distance_m:
        deficit = _clamp((brake_distance_m - wall.front_m) / brake_distance_m, 0.0, 1.0)
        return -params.brake_gain * deficit

    curve_severity = 0.0
    if camera.visible:
        curve_severity = _clamp(abs(camera.heading_error_degrees) / params.corner_heading_error_deg, 0.0, 1.0)
    target_speed_mps = params.max_speed_mps - curve_severity * (params.max_speed_mps - params.min_corner_speed_mps)

    speed_error_mps = target_speed_mps - speed_mps
    return _clamp(speed_error_mps * params.speed_gain, -1.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
