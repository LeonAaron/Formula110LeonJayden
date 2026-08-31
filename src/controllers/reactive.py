"""Reactive rule-based controller: sensor-driven throttle and steering.

All tunable gains and thresholds live in ``ReactiveParams`` so a future
parameter-search step (random search, Optuna, CMA-ES, ...) can explore the
knob space by calling :func:`drive` with different parameter sets, without
touching the control logic itself.

Design notes:

- Steering targets an "apex" offset biased toward the inside of the upcoming
  turn (right for a right turn, left for a left turn), inferred from the sign
  and magnitude of ``camera.heading_error_degrees``. Because the simulator
  scores forward progress as centerline projection rather than physical path
  length, cutting the inside of a turn advances scored distance per meter
  actually driven.
- Throttle targets ``max_speed_mps`` by default, with two optional,
  independently-tunable adjustments (each off by default via a 0 gain):
  proactive corner slow-down (``corner_speed_gain``, driven by anticipated
  heading error or actual yaw rate, whichever is sharper) and a
  speed-squared term in the braking-distance formula (``brake_quadratic_coeff``,
  matching true constant-deceleration stopping distance). The reactive,
  wall-proximity braking check remains the hard safety floor against
  destruction regardless of these settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from racing import LidarSensors, RobotCommand, RobotSensors

RACING_NAME = "Reactive Pilot"
RACING_COLOR = "#39D98A"


@dataclass(frozen=True, slots=True)
class ReactiveParams:
    """Tunable gains and thresholds for the reactive controller.

    Defaults are the result of scripts/optimize_reactive.py (bounded two-phase
    evolution strategy, trained on 5 seeds). See LAB_NOTEBOOK.md, Entry 6.
    Validated: 50/50 survived, 0.00 damage in every race across 5 training
    seeds (13, 55, 110, 271, 997) and 5 disjoint validation seeds (42, 2027,
    8675, 31415, 777001), avg. ~442-445 m/30s (up from a prior ~432-435 m).
    """

    # Steering: follow the track centerline and upcoming curvature.
    center_offset_gain: float = 0.09437765812661737
    heading_error_gain: float = 0.05761283755797004
    lookahead_near_gain: float = 0.09033539947299812
    lookahead_far_gain: float = 0.05150093277359806

    # Steering: nudge away from a close side wall before it becomes urgent.
    wall_avoid_margin_m: float = 0.9984770482614183
    wall_avoid_gain: float = 0.33676478535712756

    # Steering: override toward open space when a wall is immediately ahead.
    emergency_front_m: float = 2.500129543202483
    emergency_steer: float = 0.8733926601175921

    # Steering: back away from active contact toward whichever side is open.
    recovery_steer: float = 1.183216644687921
    recovery_throttle: float = -0.21430855823458606

    steer_limit: float = 1.0

    # Steering: infer turn sharpness from heading error and hug the inside
    # of the turn (right side on a right turn, left side on a left turn).
    turn_sharpness_deg: float = 7.9347978275593345
    apex_bias_max_m: float = 0.31137445196443336

    # Throttle: target speed, optionally reduced proactively for an anticipated
    # turn (camera heading error) or an actual one already underway (yaw
    # rate). corner_speed_gain settled near 0 in search — negligible effect.
    max_speed_mps: float = 18.194525474628353
    speed_gain: float = 0.33561427559043133
    corner_speed_gain: float = 0.021591851522117442
    corner_signal_deg: float = 41.85343568146501
    corner_yaw_rate_deg_per_s: float = 49.77802577209076

    # Throttle: brake in time for the wall directly ahead. Distance is
    # linear-in-speed plus an optional speed-squared term (true stopping
    # distance under constant deceleration is quadratic in speed);
    # brake_quadratic_coeff settled at 0 in search — not useful here.
    brake_lead_time_s: float = 0.23534499772289047
    brake_min_distance_m: float = 0.2620843869496009
    brake_gain: float = 1.5356470476775665
    brake_quadratic_coeff: float = 0.0
    brake_quadratic_coeff: float = 0.0


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
        heading_error = camera.heading_error_degrees
        apex_target_offset_m = _apex_target_offset_m(heading_error, params)
        center_error_m = camera.center_offset_m - apex_target_offset_m

        steer += params.center_offset_gain * center_error_m
        steer += params.heading_error_gain * heading_error
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


def _apex_target_offset_m(heading_error_degrees: float, params: ReactiveParams) -> float:
    """Return the lateral centerline offset to target: biased toward the inside of the turn.

    Positive ``heading_error_degrees`` means the track turns right, so the
    target is biased so the car sits to the right of center (negative
    ``center_offset_m``), and symmetrically for a left turn.
    """
    turn_severity = _clamp(abs(heading_error_degrees) / params.turn_sharpness_deg, 0.0, 1.0)
    turn_sign = 1.0 if heading_error_degrees > 0.0 else (-1.0 if heading_error_degrees < 0.0 else 0.0)
    return -turn_sign * turn_severity * params.apex_bias_max_m


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

    brake_distance_m = max(
        params.brake_min_distance_m,
        speed_mps * params.brake_lead_time_s + params.brake_quadratic_coeff * speed_mps * speed_mps,
    )
    if wall.front_m < brake_distance_m:
        deficit = _clamp((brake_distance_m - wall.front_m) / brake_distance_m, 0.0, 1.0)
        return -params.brake_gain * deficit

    corner_signal = _corner_speed_signal(sensors, params)
    target_speed_mps = params.max_speed_mps * (1.0 - params.corner_speed_gain * corner_signal)
    speed_error_mps = target_speed_mps - speed_mps
    return _clamp(speed_error_mps * params.speed_gain, -1.0, 1.0)


def _corner_speed_signal(sensors: RobotSensors, params: ReactiveParams) -> float:
    """Return an anticipated-or-actual turn severity signal in [0, 1].

    Combines the anticipated turn (camera heading error, known before the car
    turns) with the actual turn already underway (IMU yaw rate), taking
    whichever indicates the sharper need to slow down.
    """
    heading_signal = 0.0
    if sensors.camera.visible and params.corner_signal_deg > 0.0:
        heading_signal = _clamp(abs(sensors.camera.heading_error_degrees) / params.corner_signal_deg, 0.0, 1.0)
    yaw_signal = 0.0
    if params.corner_yaw_rate_deg_per_s > 0.0:
        yaw_signal = _clamp(abs(sensors.imu.yaw_rate_degrees_per_s) / params.corner_yaw_rate_deg_per_s, 0.0, 1.0)
    return max(heading_signal, yaw_signal)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
