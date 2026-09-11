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

import math
from dataclasses import dataclass

from racing import CameraSensors, LidarSensors, RobotCommand, RobotSensors

RACING_NAME = "Reactive Pilot"
RACING_COLOR = "#39D98A"

WHEELBASE_M = 1.4
MAX_STEER_DEGREES = 25.0


@dataclass(frozen=True, slots=True)
class ReactiveParams:
    """Tunable gains and thresholds for the reactive controller.

    Defaults are the result of scripts/optimize_reactive.py (evolution
    strategy, seeded from a hand-tuned baseline). See LAB_NOTEBOOK.md, Entry 3.
    """

    # Steering: pure pursuit toward a point on the centerline ahead, at a
    # speed-scaled lookahead distance (capped by the camera's farthest point).
    lookahead_time_s: float = 0.3
    lookahead_min_m: float = 3.5

    # Steering: nudge away from a close side wall before it becomes urgent.
    wall_avoid_margin_m: float = 4.980647121782573
    wall_avoid_gain: float = 0.232047729634254

    # Steering: override toward open space when a wall is immediately ahead.
    emergency_front_m: float = 2.0344921028737457
    emergency_steer: float = 2.1794681274516834

    # Steering: back away from active contact toward whichever side is open.
    recovery_steer: float = 1.03633882463599
    recovery_throttle: float = -0.3378246479078141

    steer_limit: float = 1.0

    # Steering: the pursuit target is shifted toward the inside of the turn it
    # sits in (right side on a right turn, left side on a left turn), by up to
    # ``apex_bias_max_m``, but never so far that the target's radius drops below
    # ``apex_min_radius_m`` (full steering lock is about 3 m).
    apex_bias_max_m: float = 1.5
    apex_min_radius_m: float = 4.5

    # Steering: keep clear of an opponent ahead by shifting the pursuit target
    # away from it, toward whichever side of the track it leaves open.
    opponent_range_m: float = 20.0
    opponent_clearance_m: float = 2.5

    # Turned around (heading error beyond ``uturn_enter_deg``): commit to a
    # full-lock turn at low speed until the heading error is small again.
    uturn_enter_deg: float = 100.0
    uturn_exit_deg: float = 40.0
    uturn_speed_mps: float = 5.0

    # Throttle: chase max speed (the car's measured limits, not the old 15.8 m/s
    # ceiling that was tuned around a broken brake); cornering is handled by
    # steering, not by slowing down.
    max_speed_mps: float = 30.0
    speed_gain: float = 0.5063421687653541

    # Throttle: brake in time for the wall directly ahead. The stopping distance
    # comes from the car's measured braking limit (about 18 m/s^2 is the most
    # it can shed while staying stable), plus a fixed margin.
    brake_decel_mps2: float = 18.0
    brake_margin_m: float = 0.5
    brake_gain: float = 1.4993178625857668

    # Throttle: corner anticipation. The curvature of the track ahead is recovered
    # from the camera lookahead offsets (constant-curvature arc fit per point), and
    # the speed target is what a fraction of the measured lateral grip allows there.
    lateral_grip_mps2: float = 46.0
    corner_grip_fraction: float = 0.9
    corner_brake_decel_mps2: float = 12.0
    corner_brake_max: float = 0.6

    # Throttle: friction circle. Braking for a corner while already using most of
    # the lateral grip unsettles the car, so that braking is scaled down by the
    # fraction of grip in use (from the IMU), down to a floor. The emergency wall
    # brake is never scaled.
    brake_floor: float = 0.3

    # Stalled against a wall (nearly stopped, wall right ahead): reverse with the
    # wheels turned toward the wall so the nose swings toward open track.
    stall_speed_mps: float = 1.5


DEFAULT_PARAMS = ReactiveParams()


class Controller:
    """Stateful wrapper around :func:`drive` that applies the direction-change rule."""

    def __init__(self, params: ReactiveParams = DEFAULT_PARAMS) -> None:
        self.params = params
        self.prev_throttle = 0.0
        self.uturn_side = 0.0  # +1 turning right, -1 left, 0 not turning around

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        params = self.params
        heading_error = sensors.camera.heading_error_degrees if sensors.camera.visible else 0.0
        if self.uturn_side == 0.0:
            if abs(heading_error) > params.uturn_enter_deg:
                self.uturn_side = math.copysign(1.0, heading_error)
        elif abs(heading_error) < params.uturn_exit_deg:
            self.uturn_side = 0.0
        command = _uturn_command(sensors, params, self.uturn_side) if self.uturn_side != 0.0 else drive(sensors, params)
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

    stalled = sensors.odometry.speed_mps < params.stall_speed_mps and sensors.lidar.front_m < params.emergency_front_m
    steer = _steering_command(sensors, params, stalled)
    throttle = params.recovery_throttle if stalled else _throttle_command(sensors, params)
    return RobotCommand(throttle=_clamp(throttle, -1.0, 1.0), steer=_clamp(steer, -1.0, 1.0))


def _uturn_command(sensors: RobotSensors, params: ReactiveParams, side: float) -> RobotCommand:
    """Turn around toward ``side`` (+1 right): full lock at low speed, backing up when blocked."""
    if sensors.contact.any_contact > 0.0:
        return _recovery_command(sensors, params)
    speed_mps = sensors.odometry.speed_mps
    if sensors.lidar.front_m < params.emergency_front_m and speed_mps < params.stall_speed_mps:
        # Reversing with the wheels the other way keeps the nose swinging toward ``side``.
        return RobotCommand(throttle=params.recovery_throttle, steer=-side)
    throttle = _clamp((params.uturn_speed_mps - speed_mps) * params.speed_gain, -0.6, 0.5)
    return RobotCommand(throttle=throttle, steer=side)


def _recovery_command(sensors: RobotSensors, params: ReactiveParams) -> RobotCommand:
    wall = sensors.wall_lidar
    open_side = -1.0 if wall.left_m > wall.right_m else 1.0
    return RobotCommand(throttle=params.recovery_throttle, steer=open_side * params.recovery_steer)


def _steering_command(sensors: RobotSensors, params: ReactiveParams, stalled: bool = False) -> float:
    camera = sensors.camera
    wall = sensors.wall_lidar
    objects = sensors.lidar  # barriers and other cars

    steer = _pure_pursuit_steer(sensors, params) if camera.visible else 0.0
    steer += _wall_avoidance_term(wall, params)

    if objects.front_m < params.emergency_front_m:
        emergency_side = -1.0 if objects.front_left_m > objects.front_right_m else 1.0
        if stalled:
            # Reversing: wheels toward the wall swing the nose toward the open side.
            return _clamp(-emergency_side * params.emergency_steer, -params.steer_limit, params.steer_limit)
        steer += emergency_side * params.emergency_steer

    return _clamp(steer, -params.steer_limit, params.steer_limit)


def _pure_pursuit_steer(sensors: RobotSensors, params: ReactiveParams) -> float:
    """Steer along the circle through the car and a target on the reconstructed centerline ahead.

    The target sits a speed-scaled distance along the track and is shifted toward the inside of
    the turn it lies in, so the driven path cuts corners the way a racing line does.
    """
    camera = sensors.camera
    segments = _track_ahead(camera)
    if not segments:
        return 0.0
    speed_mps = max(sensors.odometry.speed_mps, 0.0)
    lookahead_m = _clamp(params.lookahead_time_s * speed_mps, params.lookahead_min_m, segments[-1].end_m)

    previous_right, previous_forward = _projection_point(camera)
    right, forward = previous_right, previous_forward
    for segment in segments:
        if lookahead_m <= segment.end_m or segment is segments[-1]:
            fraction = _clamp((lookahead_m - segment.start_m) / (segment.end_m - segment.start_m), 0.0, 1.0)
            right = previous_right + fraction * (segment.end_right_m - previous_right)
            forward = previous_forward + fraction * (segment.end_forward_m - previous_forward)
            shift_m = _apex_shift_m(segment.curvature, params) + _opponent_shift_m(sensors, params, right, forward)
            # Track-right at the target, in the car frame, from the track heading there.
            right += shift_m * math.cos(segment.end_heading_rad)
            forward -= shift_m * math.sin(segment.end_heading_rad)
            break
        previous_right, previous_forward = segment.end_right_m, segment.end_forward_m

    distance_sq = max(right * right + forward * forward, 1e-6)
    curvature_command = 2.0 * right / distance_sq
    return math.degrees(math.atan(WHEELBASE_M * curvature_command)) / MAX_STEER_DEGREES


def _apex_shift_m(curvature: float, params: ReactiveParams) -> float:
    """Signed track-right shift of the pursuit target toward the inside of a turn of ``curvature``."""
    if abs(curvature) < 1e-4:
        return 0.0
    radius_m = 1.0 / abs(curvature)
    shift_m = _clamp(radius_m - params.apex_min_radius_m, 0.0, params.apex_bias_max_m)
    return math.copysign(shift_m, curvature)


def _opponent_shift_m(
    sensors: RobotSensors, params: ReactiveParams, target_right_m: float, target_forward_m: float
) -> float:
    """Track-right shift that keeps the pursuit target ``opponent_clearance_m`` clear of the nearest car ahead.

    The camera gives each opponent's range and bearing; the one nearest ahead that sits within the
    clearance of the target is passed on the side of the target it leaves more room on.
    """
    for competitor in sensors.camera.competitors:
        if competitor.distance_m > params.opponent_range_m or abs(competitor.angle_degrees) > 60.0:
            continue
        angle = math.radians(competitor.angle_degrees)
        right_m = competitor.distance_m * math.sin(angle)
        forward_m = competitor.distance_m * math.cos(angle)
        if forward_m > target_forward_m + params.opponent_clearance_m:
            continue
        gap_m = target_right_m - right_m
        if abs(gap_m) >= params.opponent_clearance_m:
            return 0.0
        side = 1.0 if gap_m > 0.0 else -1.0
        return side * params.opponent_clearance_m - gap_m
    return 0.0


def _wall_avoidance_term(wall: LidarSensors, params: ReactiveParams) -> float:
    term = 0.0
    if wall.left_m < params.wall_avoid_margin_m:
        term += params.wall_avoid_gain * (params.wall_avoid_margin_m - wall.left_m)
    if wall.right_m < params.wall_avoid_margin_m:
        term -= params.wall_avoid_gain * (params.wall_avoid_margin_m - wall.right_m)
    return term


def _throttle_command(sensors: RobotSensors, params: ReactiveParams) -> float:
    speed_mps = sensors.odometry.speed_mps

    target_mps = min(params.max_speed_mps, _corner_speed_mps(sensors, params))
    throttle = _clamp((target_mps - speed_mps) * params.speed_gain, -params.corner_brake_max, 1.0)
    if throttle < 0.0:
        throttle *= _brake_scale(sensors, params)

    front_m = sensors.lidar.front_m  # barriers and other cars
    brake_distance_m = speed_mps * speed_mps / (2.0 * params.brake_decel_mps2) + params.brake_margin_m
    if speed_mps > 0.0 and front_m < brake_distance_m:
        deficit = _clamp((brake_distance_m - front_m) / brake_distance_m, 0.0, 1.0)
        throttle = min(throttle, -params.brake_gain * deficit)
    return throttle


def _corner_speed_mps(sensors: RobotSensors, params: ReactiveParams) -> float:
    """Highest speed from which every corner seen by the camera can still be taken.

    A segment of mean curvature ``k`` allows ``sqrt(a_lat / k)`` at its start, and the car may
    still shed ``2 * a_brake * distance`` of speed squared before getting there.
    """
    camera = sensors.camera
    if not camera.visible:
        return params.max_speed_mps
    allowed = params.lateral_grip_mps2 * params.corner_grip_fraction
    limit = float("inf")
    for segment in _track_ahead(camera):
        curvature = abs(segment.curvature)
        if curvature > 1e-4:
            corner_speed_sq = allowed / curvature
            limit = min(limit, math.sqrt(corner_speed_sq + 2.0 * params.corner_brake_decel_mps2 * segment.start_m))
    return limit


@dataclass(frozen=True, slots=True)
class _Segment:
    """One stretch of centerline between consecutive camera lookahead points, in the car frame."""

    start_m: float
    end_m: float
    curvature: float  # mean signed curvature, positive = the track turns right
    end_right_m: float
    end_forward_m: float
    end_heading_rad: float  # track heading at the end point relative to the car heading, positive = right


def _projection_point(camera: CameraSensors) -> tuple[float, float]:
    """Car-frame (right, forward) of the car's own projection onto the centerline."""
    heading = math.radians(camera.heading_error_degrees)
    cos_h = math.cos(heading)
    if abs(cos_h) < 0.2:
        cos_h = math.copysign(0.2, cos_h)
    return camera.center_offset_m, -camera.center_offset_m * math.sin(heading) / cos_h


def _track_ahead(camera: CameraSensors) -> list[_Segment]:
    """Reconstruct the centerline ahead of the car from the camera's lookahead offsets.

    Only lateral offsets are observed. Fitting a constant-curvature arc from the car's centerline
    projection to each point (:func:`_arc_curvature`) supplies the forward coordinate and the
    track heading change to that point; consecutive heading changes give each segment's mean
    curvature. Exact for constant-curvature track, a fair approximation otherwise.
    """
    heading = math.radians(camera.heading_error_degrees)
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    center = camera.center_offset_m
    _, start_forward = _projection_point(camera)
    segments: list[_Segment] = []
    previous_distance = 0.0
    previous_theta = 0.0
    for offset_m, distance_m in zip(camera.lookahead_offsets_m, camera.lookahead_distances_m, strict=False):
        if distance_m <= previous_distance:
            continue
        curvature = _arc_curvature(offset_m, distance_m, camera.heading_error_degrees, center)
        theta = curvature * distance_m
        if abs(curvature) < 1e-6:
            u, w = 0.0, distance_m
        else:
            u, w = (1.0 - math.cos(theta)) / curvature, math.sin(theta) / curvature
        segments.append(
            _Segment(
                start_m=previous_distance,
                end_m=distance_m,
                curvature=(theta - previous_theta) / (distance_m - previous_distance),
                end_right_m=u * cos_h + w * sin_h + center,
                end_forward_m=w * cos_h - u * sin_h + start_forward,
                end_heading_rad=heading + theta,
            )
        )
        previous_distance, previous_theta = distance_m, theta
    return segments


def _arc_curvature(offset_m: float, distance_m: float, heading_error_degrees: float, center_offset_m: float) -> float:
    """Curvature of the constant-curvature arc, starting at the car's centerline projection along the
    track heading, that passes ``distance_m`` of arc later through the observed lookahead point.

    The camera reports lateral offsets in the car frame; the arc is expressed in the track-heading
    frame and rotated by the heading error. The predicted offset grows monotonically with the
    curvature while the swept angle stays below about 2.3 rad, so a bisection recovers it.
    """
    if distance_m <= 0.0:
        return 0.0
    heading = math.radians(heading_error_degrees)
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    target = offset_m - center_offset_m

    def predicted(curvature: float) -> float:
        if abs(curvature) < 1e-6:
            return distance_m * sin_h
        theta = curvature * distance_m
        return ((1.0 - math.cos(theta)) * cos_h + math.sin(theta) * sin_h) / curvature

    low, high = -2.3 / distance_m, 2.3 / distance_m
    if target <= predicted(low):
        return low
    if target >= predicted(high):
        return high
    for _ in range(20):
        mid = 0.5 * (low + high)
        if predicted(mid) < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _brake_scale(sensors: RobotSensors, params: ReactiveParams) -> float:
    """Friction circle: how much braking is left once the lateral grip in use is accounted for."""
    lateral_use = _clamp(abs(sensors.imu.lateral_acceleration_mps2) / params.lateral_grip_mps2, 0.0, 1.0)
    return max(params.brake_floor, 1.0 - lateral_use * lateral_use)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
