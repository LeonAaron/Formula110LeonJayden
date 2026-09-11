"""Reactive rule-based controller: sensor-driven throttle and steering.

All tunable gains and thresholds live in ``ReactiveParams`` so a future
parameter-search step (random search, Optuna, CMA-ES, ...) can explore the
knob space by calling :func:`drive` with different parameter sets, without
touching the control logic itself.

Two design choices worth noting:

- Steering targets an "apex" offset biased toward the inside of the upcoming
  turn (right for a right turn, left for a left turn), inferred from the sign
  and magnitude of ``camera.heading_error_degrees``. Because the simulator
  scores forward progress as centerline projection rather than physical path
  length, cutting the inside of a turn advances scored distance per meter
  actually driven.
- Throttle always targets ``max_speed_mps``; there is no proactive
  slow-down for corners. The only source of negative throttle is the
  reactive, wall-proximity braking check — a hard safety floor against
  destruction, not a cornering strategy. Turn geometry is handled entirely by
  steering (apex bias + wall avoidance).

:func:`drive` stays a pure sensor-to-command mapping. :class:`Controller`
wraps it with the one piece of state the simulator's throttle model needs:
a negative throttle while rolling forward requests a direction change, and
the simulator keeps braking on every following tick until the car has nearly
stopped, unless a tick with throttle exactly ``0.0`` clears the request. So
the controller coasts for exactly one tick when switching from braking back
to driving; without that, every brake application turns into a full stop.
"""

from __future__ import annotations

from dataclasses import dataclass

from racing import LidarSensors, RobotCommand, RobotSensors

RACING_NAME = "Reactive Pilot"
RACING_COLOR = "#39D98A"


@dataclass(frozen=True, slots=True)
class ReactiveParams:
    """Tunable gains and thresholds for the reactive controller.

    Defaults are the result of scripts/optimize_reactive.py (evolution
    strategy, seeded from a hand-tuned baseline). See LAB_NOTEBOOK.md, Entry 3.
    """

    # Steering: follow the track centerline and upcoming curvature.
    center_offset_gain: float = 0.08509623824373289
    heading_error_gain: float = 0.05791054478631408
    lookahead_near_gain: float = 0.036385639649514995
    lookahead_far_gain: float = 0.03652536774873335

    # Steering: nudge away from a close side wall before it becomes urgent.
    wall_avoid_margin_m: float = 4.980647121782573
    wall_avoid_gain: float = 0.232047729634254

    # Steering: override toward open space when a wall is immediately ahead.
    emergency_front_m: float = 2.0344921028737457
    emergency_steer: float = 2.1794681274516834

    # Steering: back away from active contact toward whichever side is open.
    recovery_steer: float = 1.03633882463599
    recovery_throttle: float = -0.3378246479078141

    steer_limit: float = 0.6270346971392542

    # Steering: infer turn sharpness from heading error and hug the inside
    # of the turn (right side on a right turn, left side on a left turn).
    turn_sharpness_deg: float = 13.06080719185704
    apex_bias_max_m: float = 0.4604484764745242

    # Throttle: always chase max speed; cornering is handled by steering, not
    # by slowing down.
    max_speed_mps: float = 15.815503530875
    speed_gain: float = 0.5063421687653541

    # Throttle: brake in time for the wall directly ahead.
    brake_lead_time_s: float = 0.24393703057360375
    brake_min_distance_m: float = 0.2412588895512276
    brake_gain: float = 1.4993178625857668


DEFAULT_PARAMS = ReactiveParams()


class Controller:
    """Stateful wrapper around :func:`drive` that applies the direction-change rule."""

    def __init__(self, params: ReactiveParams = DEFAULT_PARAMS) -> None:
        self.params = params
        self.prev_throttle = 0.0

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        command = drive(sensors, self.params)
        return self._command(command.throttle, command.steer)

    def _command(self, throttle: float, steer: float) -> RobotCommand:
        """Coast for one tick between braking and driving so the brake request clears."""
        throttle = _clamp(throttle, -1.0, 1.0)
        if self.prev_throttle < 0.0 and throttle > 0.0:
            throttle = 0.0
        self.prev_throttle = throttle
        return RobotCommand(throttle=throttle, steer=_clamp(steer, -1.0, 1.0))


def create_controller() -> Controller:
    return Controller()


_SHARED_CONTROLLER = Controller()


def control(sensors: RobotSensors) -> RobotCommand:
    """Map one sensor snapshot to a throttle/steer command (module-level, shared state)."""
    return _SHARED_CONTROLLER(sensors)


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

    brake_distance_m = max(params.brake_min_distance_m, speed_mps * params.brake_lead_time_s)
    if wall.front_m < brake_distance_m:
        deficit = _clamp((brake_distance_m - wall.front_m) / brake_distance_m, 0.0, 1.0)
        return -params.brake_gain * deficit

    speed_error_mps = params.max_speed_mps - speed_mps
    return _clamp(speed_error_mps * params.speed_gain, -1.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
