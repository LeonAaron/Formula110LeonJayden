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

    # Entry 13: jointly re-tuned by a damage-aware search (optimize_racing_line_follower.py,
    # DAMAGE_PENALTY_M penalizing partial damage, not just elimination), then a second
    # refinement pass specifically retrained against the first pass's own failing seeds.
    # Validated 30/30 fresh seeds, zero damage, avg 564.0m at 30s rounds (vs. 516.9m before
    # this entry). See LAB_NOTEBOOK.md.
    # Entry 14: further damage-aware search from the Entry 13 baseline. Validated
    # 12/12 fresh seeds, zero damage, avg 577.8m raw at 30s rounds. See LAB_NOTEBOOK.md.
    # These match the currently-shipped src/controllers/learned_policy.pt exactly
    # (Entry 14). Entry 16 explored a much faster anticipatory-braking configuration
    # (flat_speed_mps ~38, lookahead_speed_gain ~3.0) that validated well on the raw
    # privileged follower (591.8m, 0/20 eliminated) but did NOT survive distillation -
    # the resulting expert behavior was too high-variance for the small BC network to
    # imitate (training MSE 0.036 vs. ~0.0002 normally), and the distilled policy
    # regressed to 484.5m with a real held-out elimination. Not shipped; see
    # LAB_NOTEBOOK.md Entry 16 for the full negative result and the exact params tried.
    # These match the currently-shipped src/controllers/learned_policy.pt (Entry 14).
    # Entry 16/17 tried the much faster anticipatory-braking candidate below (as a
    # comment, since it wasn't adopted) - see LAB_NOTEBOOK.md for both attempts and why
    # neither beat this baseline end-to-end (after distillation) despite being much
    # faster at the raw-follower level:
    #   flat_speed_mps=37.944328546608624, lookahead_speed_gain=2.9706281473562366,
    #   min_corner_speed_mps=21.256961027335564, corner_speed_gain=-0.001213177099887005,
    #   center_offset_gain=0.00223195922022746, heading_error_gain=-0.018531572430895284,
    #   lookahead_near_gain=0.015775280600251164, lookahead_far_gain=0.7070583791986762,
    #   wall_avoid_margin_m=1.3074001206750197, wall_avoid_gain=0.0789513198549351,
    #   emergency_front_m=4.099916042555758, emergency_steer=0.3715877139817356,
    #   recovery_steer=0.10399086191606098, recovery_throttle=-0.8273207340060894,
    #   steer_limit=1.0935232732492093, speed_gain=0.12989387169239225,
    #   brake_lead_time_s=0.07244818546617149, brake_min_distance_m=0.19882628915905254,
    #   brake_gain=1.8715397173284583
    center_offset_gain: float = 0.012899463681782353
    heading_error_gain: float = -0.01391992857960881
    lookahead_near_gain: float = 0.024801403037894198
    lookahead_far_gain: float = 0.6057741143566229

    wall_avoid_margin_m: float = 1.4023947457897594
    wall_avoid_gain: float = 0.08549973960175224

    emergency_front_m: float = 3.6462873832878526
    emergency_steer: float = 0.5970607825765307

    recovery_steer: float = 0.13617973165530606
    recovery_throttle: float = -0.7259265652910875

    steer_limit: float = 1.9779874908017732

    speed_gain: float = 0.145477008220457
    brake_lead_time_s: float = 0.11106257659743615
    brake_min_distance_m: float = 0.2000036852800541
    brake_gain: float = 1.1476924166101243

    # Flat throttle target overriding the racing line's own speed profile - this
    # simulator's grip makes flat-out driving faster than curvature-based
    # slowdown (see LAB_NOTEBOOK.md Entry 5), so speed is tuned here instead. This is now
    # the straight-line ceiling, not a flat everywhere-target - see lookahead_speed_gain.
    flat_speed_mps: float = 18.687471445458616

    # Proactive cornering speed control (Entry 15): target speed drops linearly with
    # upcoming heading error, so flat_speed_mps can be raised well above the old flat
    # ceiling for straights while corners still get a lower, safe target - instead of
    # relying only on the reactive wall-proximity brake. Defaults to 0 (no effect,
    # exactly the old flat-speed behavior). Tested at gain>0 twice this session (once on
    # each racing line): both searches converged back to an effectively-flat solution
    # (min_corner_speed_mps >= flat_speed_mps, neutralizing the term) - this track's
    # corners are too close together for a straight/corner speed split to pay off. Left
    # in place (0 = inert) since the mechanism itself is correct, just unused here.
    corner_speed_gain: float = 0.0
    min_corner_speed_mps: float = 8.0

    # Entry 16/17: anticipatory version of the above - signal comes from the FAR camera
    # lookahead point (16m ahead, camera.lookahead_offsets_m[-1]), which reports upcoming
    # curvature ~1s before the car arrives there, instead of heading_error_degrees which
    # only reflects the car's *current* alignment. Left inert (0) - see the comment above
    # flat_speed_mps for the working (but not adopted) params.
    lookahead_speed_gain: float = 0.0


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
    far_lookahead_m = sensors.camera.lookahead_offsets_m[-1] if sensors.camera.lookahead_offsets_m else 0.0
    target_speed_mps = max(
        params.min_corner_speed_mps,
        params.flat_speed_mps
        - params.corner_speed_gain * abs(sensors.camera.heading_error_degrees)
        - params.lookahead_speed_gain * abs(far_lookahead_m),
    )

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
