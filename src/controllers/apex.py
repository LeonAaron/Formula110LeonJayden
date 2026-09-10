"""Apex: map-localized racing-line controller.

Pipeline, every tick, from public sensors only:

1. **Localize** (``apex_map.TrackMap``): recover the car's centerline arc
   length ``s`` and lateral position from ``imu.heading_degrees``,
   ``camera.heading_error_degrees``, ``camera.center_offset_m`` and the three
   ``camera.lookahead_offsets_m`` values, matched against the public track
   layout. No race-progress field is read; the localization is verified to
   within one grid cell (0.2 m) everywhere on the track.
2. **Steer** with pure pursuit toward a point on a precomputed racing line
   (``apex_plan``, produced offline by ``scripts/plan_apex_line.py``) a
   speed-scaled distance ahead.
3. **Throttle** toward the line's precomputed speed profile (grip-limited
   corner speeds with acceleration/braking-limited transitions), with a
   proportional controller.
4. **Safety layer**: wall-lidar emergency braking and steering, contact
   recovery, and opponent avoidance from ``camera.competitors``. If the
   localization residual is ever implausible, the tick falls back to the
   sensor-only reactive controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from controllers import apex_plan
from controllers.apex_map import TrackMap
from controllers.reactive import DEFAULT_PARAMS as REACTIVE_PARAMS
from controllers.reactive import drive as reactive_drive
from racing import RobotCommand, RobotSensors

RACING_NAME = "Apex Pilot"
RACING_COLOR = "#FF3B6B"

WHEELBASE_M = 1.4
MAX_STEER_DEGREES = 25.0


@dataclass(frozen=True, slots=True)
class ApexParams:
    # Pure pursuit lookahead: L = clamp(lookahead_time_s * v, min, max)
    lookahead_time_s: float = 0.32
    lookahead_min_m: float = 3.5
    lookahead_max_m: float = 12.0
    # Feed a fraction of the line's curvature at the car as steering feedforward.
    curvature_feedforward: float = 0.0
    # Speed profile scaling and tracking.
    speed_scale: float = 1.0
    speed_lead_m: float = 1.0
    throttle_gain: float = 0.6
    brake_max: float = 0.7
    # Safety: wall ahead.
    wall_stop_decel_mps2: float = 22.0
    wall_stop_margin_m: float = 1.2
    emergency_front_m: float = 2.5
    emergency_steer: float = 0.8
    # Contact recovery.
    recovery_throttle: float = -0.4
    recovery_steer: float = 0.5
    # Opponent avoidance.
    opponent_range_m: float = 22.0
    opponent_clearance_m: float = 2.3
    # Localization sanity.
    residual_limit: float = 1.5


DEFAULT_PARAMS = ApexParams()


class Controller:
    def __init__(self, params: ApexParams = DEFAULT_PARAMS) -> None:
        self.params = params
        self.map = TrackMap()
        self.plan_step = apex_plan.GRID_STEP_M
        self.plan_length = apex_plan.TRACK_LENGTH_M
        self.offset = np.asarray(apex_plan.OFFSET_M)
        self.speed = np.asarray(apex_plan.SPEED_MPS)
        self.curvature = np.asarray(apex_plan.CURVATURE)
        self.fallback_ticks = 0
        self.last: dict[str, float] = {}
        self.prev_throttle = 0.0

    # -- plan lookups -----------------------------------------------------
    def _interp(self, table: np.ndarray, s: float) -> float:
        position = (s % self.plan_length) / self.plan_step
        index = int(position)
        fraction = position - index
        n = len(table)
        return float(table[index % n] * (1.0 - fraction) + table[(index + 1) % n] * fraction)

    def line_point(self, s: float) -> tuple[float, float]:
        cx, cz = self.map.center_at(s)
        lx, lz = self.map.left_at(s)
        d = self._interp(self.offset, s)
        return cx + d * lx, cz + d * lz

    # -- control ----------------------------------------------------------
    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        p = self.params
        wall = sensors.wall_lidar
        if sensors.contact.any_contact > 0.0:
            open_side = -1.0 if wall.left_m > wall.right_m else 1.0
            return self._command(p.recovery_throttle, open_side * p.recovery_steer)

        loc = self.map.localize(sensors, None)
        if loc.residual > p.residual_limit:
            self.fallback_ticks += 1
            command = reactive_drive(sensors, REACTIVE_PARAMS)
            return self._command(command.throttle, command.steer)

        s = loc.s_m
        v = sensors.odometry.speed_mps
        psi = math.radians(sensors.imu.heading_degrees)
        fx, fz = math.sin(psi), math.cos(psi)
        rx, rz = fz, -fx

        # Car world position from the localization hypothesis.
        cx, cz = self.map.center_at(s)
        lx, lz = self.map.left_at(s)
        px, pz = cx + loc.lateral_m * lx, cz + loc.lateral_m * lz

        # Pure pursuit target on the racing line.
        lookahead = min(p.lookahead_max_m, max(p.lookahead_min_m, p.lookahead_time_s * max(v, 0.0)))
        s_target = s + lookahead
        tx, tz = self.line_point(s_target)
        tx, tz = self._avoid_opponents(sensors, s_target, tx, tz, px, pz, fx, fz, rx, rz)
        dx, dz = tx - px, tz - pz
        ahead = dx * fx + dz * fz
        right = dx * rx + dz * rz
        distance_sq = max(ahead * ahead + right * right, 1e-6)
        curvature_cmd = 2.0 * right / distance_sq
        if p.curvature_feedforward:
            curvature_cmd += -p.curvature_feedforward * self._interp(self.curvature, s)
        steer = math.degrees(math.atan(WHEELBASE_M * curvature_cmd)) / MAX_STEER_DEGREES

        # Speed target from the profile a little ahead of the car.
        v_target = p.speed_scale * min(self._interp(self.speed, s + p.speed_lead_m), self._interp(self.speed, s))
        throttle = (v_target - v) * p.throttle_gain
        throttle = max(-p.brake_max, min(1.0, throttle))

        # Safety: wall directly ahead.
        stop_distance = v * v / (2.0 * p.wall_stop_decel_mps2) + p.wall_stop_margin_m
        if wall.front_m < stop_distance:
            throttle = min(throttle, -p.brake_max)
        if wall.front_m < p.emergency_front_m:
            emergency_side = -1.0 if wall.front_left_m > wall.front_right_m else 1.0
            steer += emergency_side * p.emergency_steer

        self.last = {
            "s": s, "lat": loc.lateral_m, "res": loc.residual, "v": v, "v_target": v_target,
            "throttle": throttle, "steer": steer, "front": wall.front_m, "stop": stop_distance,
            "wall_brake": float(wall.front_m < stop_distance), "lookahead": lookahead,
            "line_lat": self._interp(self.offset, s), "curv": self._interp(self.curvature, s),
        }
        return self._command(throttle, steer)

    def _command(self, throttle: float, steer: float) -> RobotCommand:
        """Clamp and apply the simulator's direction-change rule.

        A negative throttle while moving forward requests a direction change,
        and the simulator then keeps braking on every following tick until the
        car nearly stops, unless a tick with throttle exactly 0.0 clears the
        pending request. So when switching from braking back to driving, coast
        for exactly one tick first.
        """
        throttle = _clamp(throttle, -1.0, 1.0)
        if self.prev_throttle < 0.0 and throttle > 0.0:
            throttle = 0.0
        self.prev_throttle = throttle
        return RobotCommand(throttle=throttle, steer=_clamp(steer, -1.0, 1.0))

    def _avoid_opponents(
        self,
        sensors: RobotSensors,
        s_target: float,
        tx: float,
        tz: float,
        px: float,
        pz: float,
        fx: float,
        fz: float,
        rx: float,
        rz: float,
    ) -> tuple[float, float]:
        p = self.params
        for competitor in sensors.camera.competitors:
            if competitor.distance_m > p.opponent_range_m or abs(competitor.angle_degrees) > 60.0:
                continue
            angle = math.radians(competitor.angle_degrees)
            ox = px + competitor.distance_m * (math.sin(angle) * rx + math.cos(angle) * fx)
            oz = pz + competitor.distance_m * (math.sin(angle) * rz + math.cos(angle) * fz)
            # Lateral gap between the target point and the opponent, measured along the track normal.
            lx, lz = self.map.left_at(s_target)
            cx, cz = self.map.center_at(s_target)
            target_lat = (tx - cx) * lx + (tz - cz) * lz
            opp_lat = (ox - cx) * lx + (oz - cz) * lz
            gap = target_lat - opp_lat
            if abs(gap) >= p.opponent_clearance_m:
                continue
            # Move the target to the side of the opponent with more room.
            limit = apex_plan.MAX_OFFSET_M
            left_option = opp_lat + p.opponent_clearance_m
            right_option = opp_lat - p.opponent_clearance_m
            candidates = [c for c in (left_option, right_option) if abs(c) <= limit]
            if not candidates:
                candidates = [max(-limit, min(limit, left_option if opp_lat < 0 else right_option))]
            new_lat = min(candidates, key=lambda c: abs(c - target_lat))
            tx, tz = cx + new_lat * lx, cz + new_lat * lz
        return tx, tz


def create_controller() -> Controller:
    return Controller()


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
