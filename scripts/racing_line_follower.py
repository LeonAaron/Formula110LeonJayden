"""Racing-line-following expert controller (privileged: needs ground-truth arc length).

Reuses ``controllers.reactive``'s steering/safety structure (lookahead terms,
wall avoidance, emergency steer, contact recovery) but replaces its heuristic
apex-bias offset and flat max-speed throttle target with the actual
minimum-curvature offset and curvature-limited speed profile computed by
``scripts/compute_racing_line.py``. Only usable where the true track
arc-length position is known (offline data generation via
``scripts/_privileged_race_env.py``) - the deployed learned controller only
ever sees public sensors, never this module.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

import torch

from racing import LidarSensors, RobotCommand, RobotSensors

DEFAULT_RACING_LINE_PATH = Path("artifacts/racing_line.pt")


@dataclass(frozen=True, slots=True)
class RacingLine:
    """Precomputed minimum-curvature offset and speed profile, by arc length."""

    s_m: tuple[float, ...]
    offset_m: tuple[float, ...]
    speed_mps: tuple[float, ...]
    total_length_m: float

    def target_offset_m(self, s: float) -> float:
        return _interpolate_circular(self.s_m, self.offset_m, s % self.total_length_m, self.total_length_m)

    def target_speed_mps(self, s: float) -> float:
        return _interpolate_circular(self.s_m, self.speed_mps, s % self.total_length_m, self.total_length_m)


def load_racing_line(path: Path = DEFAULT_RACING_LINE_PATH) -> RacingLine:
    data = torch.load(path, weights_only=True)
    return RacingLine(
        s_m=tuple(data["s_m"].tolist()),
        offset_m=tuple(data["offset_m"].tolist()),
        speed_mps=tuple(data["speed_mps"].tolist()),
        total_length_m=float(data["total_length_m"]),
    )


def _interpolate_circular(xs: tuple[float, ...], ys: tuple[float, ...], x: float, period: float) -> float:
    """Linearly interpolate a value sampled on a closed loop, wrapping at `period`."""
    index = bisect.bisect_right(xs, x)
    if index == 0:
        x0, y0 = xs[-1] - period, ys[-1]
        x1, y1 = xs[0], ys[0]
    elif index == len(xs):
        x0, y0 = xs[-1], ys[-1]
        x1, y1 = xs[0] + period, ys[0]
    else:
        x0, y0 = xs[index - 1], ys[index - 1]
        x1, y1 = xs[index], ys[index]
    if x1 == x0:
        return y0
    fraction = (x - x0) / (x1 - x0)
    return y0 + fraction * (y1 - y0)


@dataclass(frozen=True, slots=True)
class RacingLineFollowerParams:
    """Tunable gains, mirroring controllers.reactive.ReactiveParams' steering/safety structure.

    Defaults are the result of scripts/optimize_racing_line_follower.py (evolution
    strategy, seeded from a hand-tuned baseline). See LAB_NOTEBOOK.md, Entry 6.
    """

    center_offset_gain: float = 0.04646983645575174
    heading_error_gain: float = 0.011969058650635005
    lookahead_near_gain: float = -0.05641812583155188
    lookahead_far_gain: float = 0.11930633186426046

    wall_avoid_margin_m: float = 1.9478186185133812
    wall_avoid_gain: float = 0.11298311580834951

    emergency_front_m: float = 2.775567055718721
    emergency_steer: float = 0.7742171195339129

    recovery_steer: float = 0.17509229333000273
    recovery_throttle: float = -0.4307810576870986

    steer_limit: float = 1.3143862095626797

    speed_gain: float = 0.2279368992608865
    brake_lead_time_s: float = 0.14049539154367582
    brake_min_distance_m: float = 0.14169744492358835
    brake_gain: float = 2.113623025002169

    # Flat throttle target overriding the racing line's own speed profile - this
    # simulator's grip makes flat-out driving faster than curvature-based
    # slowdown (see LAB_NOTEBOOK.md Entry 5), so speed is tuned here instead.
    # 17.0 m/s is the measured zero-damage ceiling on this line (20/20 seeds,
    # see LAB_NOTEBOOK.md Entry 6): a hillclimb search pushed this to ~23 m/s,
    # which drives with real wall damage every race (survives narrowly under
    # privileged ground-truth tracking) and does not survive distillation to a
    # sensor-only policy (7/25 held-out). Value chosen with the same
    # zero-damage safety margin as the shipped reactive/BC controllers.
    flat_speed_mps: float = 17.0


DEFAULT_FOLLOWER_PARAMS = RacingLineFollowerParams()


def follow(
    sensors: RobotSensors,
    *,
    true_progress_distance_m: float,
    racing_line: RacingLine,
    params: RacingLineFollowerParams = DEFAULT_FOLLOWER_PARAMS,
) -> RobotCommand:
    """Map sensors + ground-truth arc length to a command that tracks the racing line."""
    if sensors.contact.any_contact > 0.0:
        wall = sensors.wall_lidar
        open_side = -1.0 if wall.left_m > wall.right_m else 1.0
        return RobotCommand(throttle=params.recovery_throttle, steer=open_side * params.recovery_steer)

    target_offset_m = racing_line.target_offset_m(true_progress_distance_m)
    target_speed_mps = params.flat_speed_mps

    steer = _steering_command(sensors, target_offset_m=target_offset_m, params=params)
    throttle = _throttle_command(sensors, target_speed_mps=target_speed_mps, params=params)
    return RobotCommand(throttle=_clamp(throttle, -1.0, 1.0), steer=_clamp(steer, -1.0, 1.0))


def _steering_command(sensors: RobotSensors, *, target_offset_m: float, params: RacingLineFollowerParams) -> float:
    camera = sensors.camera
    wall = sensors.wall_lidar

    steer = 0.0
    if camera.visible:
        center_error_m = camera.center_offset_m - target_offset_m
        steer += params.center_offset_gain * center_error_m
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


def _wall_avoidance_term(wall: LidarSensors, params: RacingLineFollowerParams) -> float:
    term = 0.0
    if wall.left_m < params.wall_avoid_margin_m:
        term += params.wall_avoid_gain * (params.wall_avoid_margin_m - wall.left_m)
    if wall.right_m < params.wall_avoid_margin_m:
        term -= params.wall_avoid_gain * (params.wall_avoid_margin_m - wall.right_m)
    return term


def _throttle_command(sensors: RobotSensors, *, target_speed_mps: float, params: RacingLineFollowerParams) -> float:
    speed_mps = sensors.odometry.speed_mps
    wall = sensors.wall_lidar

    brake_distance_m = max(params.brake_min_distance_m, speed_mps * params.brake_lead_time_s)
    if wall.front_m < brake_distance_m:
        deficit = _clamp((brake_distance_m - wall.front_m) / brake_distance_m, 0.0, 1.0)
        return -params.brake_gain * deficit

    speed_error_mps = target_speed_mps - speed_mps
    return _clamp(speed_error_mps * params.speed_gain, -1.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
