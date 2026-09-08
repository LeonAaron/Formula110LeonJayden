"""Reactive rule-based controller v3: sensor-driven throttle and steering.

Forked from ``controllers.reactive`` (do not edit that file from here) to try
one isolated change: off-axis wall detection. The rest of the control law is
unchanged from the v2 (Entry 6) tuned defaults.

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
- v3 change: the emergency steer override no longer looks only at the beam
  nearest 0 degrees (``front_m``) plus the fixed +-20 degree pair. ``wall_lidar``
  actually reports 7 beams (-90, -45, -20, 0, 20, 45, 90 degrees; see
  ``DEFAULT_LIDAR_ANGLES_DEGREES``) but the original controller's convenience
  properties only ever surface 0/-20/20/-90/90 — the +-45 degree beams are
  read by no code path at all. ``_forward_cone_scan`` projects every beam
  within a cone of straight-ahead onto the forward axis
  (``distance * cos(angle)``) and takes the minimum, so a wall closing in at
  an angle within that cone (not just dead ahead) can trigger the emergency
  override, and the escape direction is chosen from the same scan rather than
  just the +-20 degree pair.
  This is deliberately *not* applied to the proactive braking check
  (still plain ``front_m``, unchanged from v2): braking runs every tick, and
  a wide-angle beam grazing the inside wall the controller intentionally
  hugs via apex bias returns a short raw range that is not a real head-on
  hazard, so cone-projecting it there caused constant false-positive braking
  and a large distance regression in testing (~442m -> ~300m at a 45 degree
  cone). The emergency override only fires inside ``emergency_front_m``
  (~2.5m) as a last resort, where that grazing false-positive risk is much
  smaller and extra caution is cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from racing import LidarSensors, RobotCommand, RobotSensors

RACING_NAME = "Reactive Pilot v3"
RACING_COLOR = "#2FB5D9"


@dataclass(frozen=True, slots=True)
class ReactiveParams:
    """Tunable gains and thresholds for the reactive controller.

    Defaults are the result of scripts/optimize_reactive_v3.py (bounded
    two-phase evolution strategy, 20 genomes x 25 broad + 15 fine-tuning
    generations, trained on 7 seeds at 20s rounds: 13, 55, 110, 271, 997,
    3001, 4002), seeded from the previous defaults and jointly re-tuning
    around the v3 off-axis emergency steer scan (see module docstring).

    Validated at the standard 30s protocol (5 races/seed) on 42, 110, 271,
    997, 2027: 25/25 survived, 0.00 damage in every race, avg. 468.7 m — up
    from the previous defaults' 463.8 m on the identical protocol. The gain is
    concentrated on the weakest seed (2027: 444.6 -> 466.3 m), tightening the
    per-seed spread from 31.5 m to 23.0 m; seed 42 gave back 4.1 m. Note that
    110/271/997 appear in both the training and validation lists, so 42 and
    2027 are the only genuinely unseen seeds here — those two alone went
    458.7 -> 467.5 m, so the improvement is not an artifact of the overlap.

    Broad exploration (sigma=0.25) found nothing better than the seeded
    baseline across all 25 generations; the entire gain came from the
    sigma=0.08 fine-tuning phase, which suggests these values sit in a fairly
    tight local optimum.
    """

    # Steering: follow the track centerline and upcoming curvature.
    center_offset_gain: float = 0.12229326954711707
    heading_error_gain: float = 0.056192419287577795
    lookahead_near_gain: float = 0.0036761772898821067
    lookahead_far_gain: float = 0.04033397631842171

    # Steering: nudge away from a close side wall before it becomes urgent.
    wall_avoid_margin_m: float = 0.7677215051928223
    wall_avoid_gain: float = 0.35174638765757454

    # Steering: override toward open space when a wall is immediately ahead.
    emergency_front_m: float = 3.846023723575205
    emergency_steer: float = 0.3275309120292628

    # Steering (v3): how wide a cone around straight-ahead the emergency
    # override scans for the forward-projected wall distance (see module
    # docstring). Settled much narrower than the 45-degree starting point —
    # covers roughly the -20/0/20 beams. Not used for braking.
    wall_scan_cone_deg: float = 25.795402779647443

    # Steering: back away from active contact toward whichever side is open.
    recovery_steer: float = 1.2506016466133008
    recovery_throttle: float = -0.15530413084727615

    steer_limit: float = 0.7273114677075063

    # Steering: infer turn sharpness from heading error and hug the inside
    # of the turn (right side on a right turn, left side on a left turn).
    turn_sharpness_deg: float = 12.153498231519766
    apex_bias_max_m: float = 0.21915967092080046

    # Throttle: target speed, optionally reduced proactively for an anticipated
    # turn (camera heading error) or an actual one already underway (yaw
    # rate). corner_speed_gain settled near 0 in search — negligible effect.
    max_speed_mps: float = 18.474453632987856
    speed_gain: float = 0.13456128958890018
    corner_speed_gain: float = 0.009731541992039133
    corner_signal_deg: float = 26.727155579997856
    corner_yaw_rate_deg_per_s: float = 65.31069316941816

    # Throttle: brake in time for the wall directly ahead. Distance is
    # linear-in-speed plus an optional speed-squared term (true stopping
    # distance under constant deceleration is quadratic in speed);
    # brake_quadratic_coeff settled at 0 in search — not useful here.
    brake_lead_time_s: float = 0.12566842391480273
    brake_min_distance_m: float = 0.20860217040280343
    brake_gain: float = 0.7842894334027986
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

    front_forward_m, left_forward_m, right_forward_m = _forward_cone_scan(wall, params.wall_scan_cone_deg)
    if front_forward_m < params.emergency_front_m:
        emergency_side = -1.0 if left_forward_m > right_forward_m else 1.0
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


def _forward_cone_scan(wall: LidarSensors, cone_half_angle_deg: float) -> tuple[float, float, float]:
    """Scan every lidar beam within ``cone_half_angle_deg`` of straight ahead.

    Each beam's raw distance is projected onto the forward axis
    (``distance * cos(angle)``), so a wall angled into the car's path -
    not just one dead ahead - counts toward how soon it will actually be
    reached; a beam at 0 degrees is unaffected (cos(0) == 1).

    Returns ``(min_forward_m, left_min_forward_m, right_min_forward_m)``:
    the closest projected distance overall, and the closest among
    left-of-center / right-of-center beams only (0 degrees excluded from
    both), for picking which way has more room.
    """
    max_m = wall.max_distance_m
    min_forward_m = max_m
    left_min_forward_m = max_m
    right_min_forward_m = max_m
    for angle_deg, distance_m in zip(wall.angles_degrees, wall.distances_m, strict=True):
        if abs(angle_deg) > cone_half_angle_deg:
            continue
        forward_m = distance_m * math.cos(math.radians(angle_deg))
        min_forward_m = min(min_forward_m, forward_m)
        if angle_deg < 0.0:
            left_min_forward_m = min(left_min_forward_m, forward_m)
        elif angle_deg > 0.0:
            right_min_forward_m = min(right_min_forward_m, forward_m)
    return min_forward_m, left_min_forward_m, right_min_forward_m


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
