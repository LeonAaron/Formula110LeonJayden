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
   sensor-only reactive controller (``controllers.reactive.Controller``,
   which needs no map and no external dependencies).
"""

from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from types import ModuleType

from controllers import apex_plan as _default_plan
from controllers.apex_map import TrackMap
from controllers.reactive import Controller as ReactiveController
from racing import RobotCommand, RobotSensors

RACING_NAME = "Apex Pilot"
RACING_COLOR = "#FF3B6B"

WHEELBASE_M = 1.4
MAX_STEER_DEGREES = 25.0


@dataclass(frozen=True, slots=True)
class ApexParams:
    # Pure pursuit lookahead: L = clamp(lookahead_time_s * v, min, max)
    lookahead_time_s: float = 0.5
    lookahead_min_m: float = 3.5
    lookahead_max_m: float = 16.0
    # Feed a fraction of the line's curvature at the car as steering feedforward.
    curvature_feedforward: float = 0.0
    lateral_gain: float = 0.0
    # Speed profile scaling and tracking.
    speed_scale: float = 1.0
    speed_lead_m: float = 1.0
    throttle_gain: float = 1.0
    brake_max: float = 0.6
    # Safety: wall ahead.
    wall_stop_decel_mps2: float = 35.0
    wall_stop_margin_m: float = 1.0
    wall_brake: float = 0.5
    emergency_front_m: float = 2.5
    emergency_steer: float = 0.8
    # Contact recovery.
    recovery_throttle: float = -0.4
    recovery_steer: float = 0.5
    recovery_speed_mps: float = 4.0
    contact_steer: float = 0.4
    contact_throttle: float = 0.3
    # Opponent avoidance: the racing line is bent around an opponent projected onto the track.
    opponent_range_m: float = 30.0
    opponent_clearance_m: float = 2.7
    pass_grip_fraction: float = 0.7
    opponent_window_before_m: float = 40.0
    opponent_window_after_m: float = 14.0
    opponent_blend_m: float = 20.0
    opponent_blend_out_m: float = 14.0
    opponent_commit_m: float = 16.0
    opponent_late_speed_mps: float = 8.0
    opponent_lookahead_max_m: float = 9.0
    opponent_edge_margin_m: float = 0.6
    opponent_follow_gap_m: float = 10.0
    opponent_follow_speed_mps: float = 6.0
    follow_brake: float = 0.35
    pass_brake_mps2: float = 9.0
    pass_brake_max: float = 0.4
    # Recovery state machine (wall contact at low speed): reverse, then drive off.
    recovery_reverse_s: float = 0.6
    recovery_forward_s: float = 0.4
    # Localization: local search window around the dead-reckoned estimate, global re-search
    # whenever the local best is implausible, reactive fallback when even that is bad.
    localize_window_m: float = 5.0
    relocalize_residual: float = 0.5
    residual_limit: float = 1.5


DEFAULT_PARAMS = ApexParams()


def _load_plan() -> ModuleType:
    """Load the shipped plan, or an alternative file named by APEX_PLAN_PATH (experiments only)."""
    path = os.environ.get("APEX_PLAN_PATH")
    if not path:
        return _default_plan
    spec = importlib.util.spec_from_file_location("apex_plan_override", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load plan from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Controller:
    def __init__(self, params: ApexParams = DEFAULT_PARAMS) -> None:
        self.params = params
        self.map = TrackMap()
        self.fallback = ReactiveController()  # sensor-only, stateful (coast tick, U-turn), no map
        apex_plan = _load_plan()
        self.max_offset_m = apex_plan.MAX_OFFSET_M
        self.a_lat = apex_plan.A_LAT_MPS2
        self.a_brake = apex_plan.A_BRAKE_MPS2
        self.pass_profile: tuple[float, list[float]] | None = None  # (start s, speeds on the plan grid)
        self.pass_basis: tuple[float, float, float] | None = None
        self.plan_step: float = apex_plan.GRID_STEP_M
        self.plan_length: float = apex_plan.TRACK_LENGTH_M
        self.offset: tuple[float, ...] = tuple(apex_plan.OFFSET_M)
        self.speed: tuple[float, ...] = tuple(apex_plan.SPEED_MPS)
        self.curvature: tuple[float, ...] = tuple(apex_plan.CURVATURE)
        self.left_limit: tuple[float, ...] = tuple(apex_plan.LEFT_LIMIT_M)
        self.right_limit: tuple[float, ...] = tuple(apex_plan.RIGHT_LIMIT_M)
        # Heavily smoothed plan lateral (about 4.5 m): the bend deviation is defined against this so
        # that its second derivative (used for the pass speed) stays free of the plan's fine detail.
        n = len(self.offset)
        self.offset_smooth: tuple[float, ...] = tuple(
            sum(self.offset[(i + k) % n] for k in range(-4, 5)) / 9.0 for i in range(n)
        )
        self.prior_s: float | None = None
        self.fallback_ticks = 0
        self.last: dict[str, float] = {}
        self.prev_throttle = 0.0
        self.avoid: dict[str, float] = {}
        self.pass_side = 0.0  # +1 pass on the left of the opponent, -1 on the right, 0 none
        self.pass_late = False  # opponent first seen too close to plan a bend: plan line + follow rule only
        self.opponent: tuple[float, float] | None = None  # (s, lateral) of the opponent being avoided
        self.recovery_phase = 0  # 0 none, 1 reversing, 2 driving forward
        self.recovery_ticks = 0
        self.recovery_side = 1.0

    # -- plan lookups -----------------------------------------------------
    def _interp(self, table: tuple[float, ...], s: float) -> float:
        position = (s % self.plan_length) / self.plan_step
        index = int(position)
        fraction = position - index
        n = len(table)
        return table[index % n] * (1.0 - fraction) + table[(index + 1) % n] * fraction

    def line_lateral(self, s: float) -> float:
        """Racing-line lateral offset at s, bent around a registered opponent."""
        d = self._interp(self.offset, s)
        if self.opponent is None or self.pass_side == 0.0 or self.pass_late:
            return d
        p = self.params
        opp_s, opp_lat = self.opponent
        rel = self.map.signed_delta(s, opp_s)  # positive: s is past the opponent
        if rel < -p.opponent_window_before_m or rel > p.opponent_window_after_m:
            return d
        required = self._clip_lateral(
            opp_lat + self.pass_side * p.opponent_clearance_m,
            s,
            margin=p.opponent_edge_margin_m,
        )
        smooth = self._interp(self.offset_smooth, s)
        # Deviation needed on the chosen side, measured against the smoothed plan lateral, through a
        # soft rectifier (width ~1 m) so the deviation stays C2 where the plan line crosses the
        # required lateral (a hard max() would put a curvature spike there).
        gap = self.pass_side * (required - smooth)
        needed = _softplus(gap, 1.0)
        if needed <= 0.02:
            return d
        # Smooth (C1) blend in before the opponent and out after it.
        if rel < -p.opponent_window_before_m + p.opponent_blend_m:
            weight = (rel + p.opponent_window_before_m) / p.opponent_blend_m
        elif rel > p.opponent_window_after_m - p.opponent_blend_out_m:
            weight = (p.opponent_window_after_m - rel) / p.opponent_blend_out_m
        else:
            weight = 1.0
        weight = _clamp(weight, 0.0, 1.0)
        weight = weight * weight * (3.0 - 2.0 * weight)
        return d + weight * self.pass_side * needed

    def _clip_lateral(self, lateral: float, s: float, margin: float = 0.2) -> float:
        """Clip a lateral offset to the per-point limits (corridor and bend-inside fold limit)."""
        return max(
            -self._interp(self.right_limit, s) + margin,
            min(self._interp(self.left_limit, s) - margin, lateral),
        )

    def line_point(self, s: float) -> tuple[float, float]:
        cx, cz = self.map.center_at(s)
        lx, lz = self.map.left_at(s)
        d = self.line_lateral(s)
        return cx + d * lx, cz + d * lz

    def project(self, x: float, z: float, near_s: float, window_m: float = 40.0) -> tuple[float, float]:
        """Project a world point onto the centerline near ``near_s``: returns (s, lateral)."""
        m = self.map
        best_index = 0
        best_dist = float("inf")
        for raw_index in m.window_indices(near_s, window_m):
            index = raw_index % m.count
            dx = x - m.cx[index]
            dz = z - m.cz[index]
            # distance along the local tangent only counts if the point is beside the sample
            along = dx * m.fwd_x[index] + dz * m.fwd_z[index]
            dist = math.hypot(dx, dz) + abs(along)
            if dist < best_dist:
                best_dist = dist
                best_index = index
        dx = x - m.cx[best_index]
        dz = z - m.cz[best_index]
        return m.s[best_index], dx * m.left_x[best_index] + dz * m.left_z[best_index]

    # -- control ----------------------------------------------------------
    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        p = self.params
        wall = sensors.wall_lidar
        recovery = self._recovery(sensors)
        if recovery is not None:
            return recovery

        loc = self.map.localize(sensors, self.prior_s, prior_window_m=p.localize_window_m)
        if self.prior_s is not None and loc.residual > p.relocalize_residual:
            loc = self.map.localize(sensors, None)  # lost (marshal reset, spin): search the whole track
        if loc.residual > p.residual_limit:
            self.prior_s = None
            self.fallback_ticks += 1
            command = self.fallback(sensors)
            return self._command(command.throttle, command.steer)

        s = loc.s_m
        v = sensors.odometry.speed_mps
        # Dead-reckoned prior for the next tick's local search.
        self.prior_s = self.map.wrap(
            s + v * sensors.dt_s * math.cos(math.radians(sensors.camera.heading_error_degrees))
        )
        psi = math.radians(sensors.imu.heading_degrees)
        fx, fz = math.sin(psi), math.cos(psi)
        rx, rz = fz, -fx

        # Car world position from the localization hypothesis.
        cx, cz = self.map.center_at(s)
        lx, lz = self.map.left_at(s)
        px, pz = cx + loc.lateral_m * lx, cz + loc.lateral_m * lz

        # Opponent bookkeeping first: it can shorten the lookahead and bend the line.
        self.avoid = {}
        self._register_opponent(sensors, s, px, pz, fx, fz, rx, rz, lateral=loc.lateral_m)

        # Pure pursuit target on the racing line.
        lookahead = min(p.lookahead_max_m, max(p.lookahead_min_m, p.lookahead_time_s * max(v, 0.0)))
        if self.opponent is not None and -25.0 <= self.map.signed_delta(self.opponent[0], s) <= 6.0:
            lookahead = min(lookahead, p.opponent_lookahead_max_m)
        s_target = s + lookahead
        tx, tz = self.line_point(s_target)
        dx, dz = tx - px, tz - pz
        ahead = dx * fx + dz * fz
        right = dx * rx + dz * rz
        distance_sq = max(ahead * ahead + right * right, 1e-6)
        curvature_cmd = 2.0 * right / distance_sq
        if p.curvature_feedforward:
            curvature_cmd += -p.curvature_feedforward * self._interp(self.curvature, s)
        steer = math.degrees(math.atan(WHEELBASE_M * curvature_cmd)) / MAX_STEER_DEGREES
        if p.lateral_gain:
            steer -= p.lateral_gain * (self.line_lateral(s) - loc.lateral_m)

        # Speed target from the profile a little ahead of the car.
        v_target = p.speed_scale * min(self._interp(self.speed, s + p.speed_lead_m), self._interp(self.speed, s))
        plan_target = v_target
        follow_active = 0.0
        if self.opponent is not None:
            self._ensure_pass_profile()
            if self.pass_profile is not None:
                v_target = min(v_target, self._pass_speed(s + p.speed_lead_m), self._pass_speed(s))
            opp_s, opp_lat = self.opponent
            gap_ahead = self.map.signed_delta(opp_s, s)
            if (
                0.0 < gap_ahead < p.opponent_follow_gap_m
                and abs(loc.lateral_m - opp_lat) < p.opponent_clearance_m - 0.8
            ):
                v_target = min(v_target, p.opponent_follow_speed_mps)
                follow_active = 1.0
        # Throttle: the plan's own braking is never limited; extra braking asked for by the pass
        # zone or the follow rule is capped (gently, and by the friction circle at the current
        # position, since an opponent hidden behind a barrier can be discovered mid-corner).
        plan_throttle = max(-p.brake_max, min(1.0, (plan_target - v) * p.throttle_gain))
        pass_throttle = max(-p.brake_max, min(1.0, (v_target - v) * p.throttle_gain))
        if follow_active:
            extra_cap = p.follow_brake
        else:
            lateral_use = max(
                v * v * abs(self._interp(self.curvature, s)) / self.a_lat,
                abs(sensors.imu.lateral_acceleration_mps2) / self.a_lat,
            )
            lateral_use = _clamp(lateral_use, 0.0, 1.0)
            extra_cap = p.pass_brake_max * (1.0 - lateral_use * lateral_use)
        throttle = min(plan_throttle, max(pass_throttle, -extra_cap))

        # Sliding along a wall at speed: steer off it, keep rolling (no brake -> no forced stop).
        if sensors.contact.wall > 0.0:
            open_side = -1.0 if wall.left_m > wall.right_m else 1.0
            steer = open_side * p.contact_steer
            throttle = min(throttle, p.contact_throttle) if throttle > 0.0 else p.contact_throttle

        # Safety: wall directly ahead.
        stop_distance = v * v / (2.0 * p.wall_stop_decel_mps2) + p.wall_stop_margin_m
        if wall.front_m < stop_distance:
            # Braking hard while steering hard costs front grip; keep it light when already turning.
            throttle = min(throttle, -p.wall_brake if abs(steer) < 0.5 else -0.25)
        if wall.front_m < p.emergency_front_m:
            emergency_side = -1.0 if wall.front_left_m > wall.front_right_m else 1.0
            steer += emergency_side * p.emergency_steer

        self.last = {
            "s": s,
            "lat": loc.lateral_m,
            "res": loc.residual,
            "v": v,
            "v_target": v_target,
            "throttle": throttle,
            "steer": steer,
            "front": wall.front_m,
            "stop": stop_distance,
            "wall_brake": float(wall.front_m < stop_distance),
            "lookahead": lookahead,
            "line_lat": self.line_lateral(s),
            "curv": self._interp(self.curvature, s),
            "plan_target": plan_target,
            "follow": follow_active,
            **{f"avoid_{k}": val for k, val in self.avoid.items()},
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

    def _register_opponent(
        self,
        sensors: RobotSensors,
        s: float,
        px: float,
        pz: float,
        fx: float,
        fz: float,
        rx: float,
        rz: float,
        lateral: float = 0.0,
        speed: float = 0.0,
    ) -> None:
        """Track the nearest opponent ahead, projected onto the track, and pick a passing side once.

        The opponent stays registered (it is static between marshal resets) until the car is past it,
        even when the camera loses it beside the car, so the bent line never snaps back mid-pass.
        """
        p = self.params
        if self.opponent is not None:
            opp_s, _ = self.opponent
            if self.map.signed_delta(s, opp_s) > p.opponent_window_after_m:
                self.opponent = None
                self.pass_side = 0.0
                self.pass_late = False
                self.pass_profile = None
                self.pass_basis = None
        ahead = [
            c for c in sensors.camera.competitors if c.distance_m <= p.opponent_range_m and abs(c.angle_degrees) <= 90.0
        ]
        if not ahead:
            return
        competitor = ahead[0]
        angle = math.radians(competitor.angle_degrees)
        ox = px + competitor.distance_m * (math.sin(angle) * rx + math.cos(angle) * fx)
        oz = pz + competitor.distance_m * (math.sin(angle) * rz + math.cos(angle) * fz)
        opp_s, opp_lat = self.project(ox, oz, s)
        rel = self.map.signed_delta(opp_s, s)
        if rel < -p.opponent_window_after_m:
            return
        if self.pass_side == 0.0 or self.opponent is None or abs(self.map.signed_delta(opp_s, self.opponent[0])) > 5.0:
            if rel < p.opponent_commit_m and speed > p.opponent_late_speed_mps:
                # Discovered too close (barriers hide parked cars in hairpins): bending the line or
                # re-profiling speed now would only unsettle the car. Keep the plan line and rely on
                # the follow rule; pass on the side we are already on.
                self.pass_late = True
                side = 1.0 if lateral > opp_lat else -1.0
            elif rel < p.opponent_commit_m and abs(lateral - opp_lat) > 0.4:
                # Too close to cross over: pass on the side we are already on.
                side = 1.0 if lateral > opp_lat else -1.0
                self.pass_late = False
            else:
                options: list[tuple[float, float]] = []
                for candidate in (1.0, -1.0):
                    required = self._clip_lateral(
                        opp_lat + candidate * p.opponent_clearance_m,
                        opp_s,
                        margin=p.opponent_edge_margin_m,
                    )
                    if abs(required - opp_lat) < p.opponent_clearance_m - 0.4:
                        continue  # not enough room on that side
                    # Simulate the bent line's speed profile on this side and score its time loss.
                    self.opponent = (opp_s, opp_lat)
                    self.pass_side = candidate
                    self.pass_basis = None
                    self._ensure_pass_profile()
                    options.append(
                        (
                            self._pass_time_loss() + 0.3 * max(0.0, abs(required) - 2.5),
                            candidate,
                        )
                    )
                side = (1.0 if opp_lat <= 0.0 else -1.0) if not options else min(options)[1]
                self.pass_basis = None
                self.pass_late = False
            self.pass_side = side
        self.opponent = (opp_s, opp_lat)
        self.avoid = {
            "opp_s": opp_s,
            "opp_lat": opp_lat,
            "rel": rel,
            "side": self.pass_side,
            "dist": competitor.distance_m,
            "angle": competitor.angle_degrees,
        }

    def _ensure_pass_profile(self) -> None:
        """Build (once per encounter) a grip- and brake-limited speed profile for the bent line."""
        if self.opponent is None or self.pass_late:
            self.pass_profile = None
            self.pass_basis = None
            return
        opp_s, opp_lat = self.opponent
        basis = (opp_s, opp_lat, self.pass_side)
        if (
            self.pass_basis is not None
            and abs(self.map.signed_delta(opp_s, self.pass_basis[0])) < 0.3
            and abs(opp_lat - self.pass_basis[1]) < 0.3
            and self.pass_side == self.pass_basis[2]
        ):
            return
        p = self.params
        # Sample on the planner's own grid so the unbent sections reproduce the plan's curvature exactly
        # (the track normals are piecewise constant, so off-grid resampling would invent kinks).
        step = self.plan_step
        first = math.floor((opp_s - p.opponent_window_before_m - 30.0) / step)
        last = math.ceil((opp_s + p.opponent_window_after_m + 6.0) / step)
        count = last - first + 1
        ss = [(first + i) * step for i in range(count)]
        # Curvature of the bent line from the plan line's curvature and the smooth lateral deviation
        # delta(s) (positive = left): kappa = (kappa_plan + delta'') / (1 - kappa_plan * delta).
        # delta is a smooth blend, so its finite-difference second derivative is well behaved,
        # unlike curvature sampled from the kinked polyline centerline.
        delta = [self.line_lateral(value) - self._interp(self.offset, value) for value in ss]
        plan_kappa = [self._interp(self.curvature, value) for value in ss]
        plan = [self._interp(self.speed, value) for value in ss]
        v: list[float] = []
        for i in range(count):
            second = 0.0
            if 0 < i < count - 1:
                second = (delta[i + 1] - 2.0 * delta[i] + delta[i - 1]) / (step * step)
            denominator = max(1.0 - plan_kappa[i] * delta[i], 0.3)
            kappa = max(abs((plan_kappa[i] + second) / denominator), 1e-6)
            speed = min(math.sqrt(self.a_lat / kappa), plan[i])
            # Near the opponent, leave grip in reserve so the car actually tracks the bent line
            # (tracking error grows quickly at the cornering limit).
            rel = self.map.signed_delta(ss[i], opp_s)
            if -25.0 <= rel <= 6.0:
                speed = min(speed, math.sqrt(p.pass_grip_fraction * self.a_lat / kappa))
            v.append(speed)
        # Backward braking pass with a friction circle: where the plan already uses most of the
        # lateral grip, little braking is allowed, so slowdowns move back onto straighter track.
        for i in range(count - 2, -1, -1):
            lateral_use = _clamp(plan[i] * plan[i] * abs(plan_kappa[i]) / self.a_lat, 0.0, 1.0)
            allowed = p.pass_brake_mps2 * _clamp(1.0 - lateral_use * lateral_use, 0.1, 1.0)
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * allowed * step))
        self.pass_profile = (ss[0], v)
        self.pass_basis = basis

    def _pass_time_loss(self) -> float:
        """Seconds lost to the current pass profile relative to the plan speed."""
        if self.pass_profile is None:
            return 0.0
        start, v = self.pass_profile
        loss = 0.0
        for i, speed in enumerate(v):
            plan = self._interp(self.speed, start + i * self.plan_step)
            loss += self.plan_step * (1.0 / max(speed, 1.0) - 1.0 / plan)
        return loss

    def _pass_speed(self, s: float) -> float:
        if self.pass_profile is None:
            return float("inf")
        start, v = self.pass_profile
        rel = self.map.signed_delta(s, start)
        if rel < 0.0 or rel > (len(v) - 1) * self.plan_step:
            return float("inf")
        position = rel / self.plan_step
        index = min(int(position), len(v) - 2)
        fraction = position - index
        return v[index] * (1.0 - fraction) + v[index + 1] * fraction

    def _recovery(self, sensors: RobotSensors) -> RobotCommand | None:
        """Wall contact at low speed: reverse with the nose swinging toward open track, then drive off."""
        p = self.params
        wall = sensors.wall_lidar
        speed = sensors.odometry.speed_mps
        if self.recovery_phase == 0:
            if sensors.contact.wall > 0.0 and abs(speed) < p.recovery_speed_mps:
                self.recovery_phase = 1
                self.recovery_ticks = 0
                self.recovery_side = -1.0 if wall.left_m > wall.right_m else 1.0  # side that is open
            else:
                return None
        self.recovery_ticks += 1
        if self.recovery_phase == 1:
            if self.recovery_ticks > p.recovery_reverse_s * 60:
                self.recovery_phase = 2
                self.recovery_ticks = 0
                return self._command(0.0, 0.0)
            # Reversing with the wheels turned toward the wall swings the nose toward open track.
            return self._command(p.recovery_throttle, -self.recovery_side * p.recovery_steer * 2.0)
        if self.recovery_ticks > p.recovery_forward_s * 60:
            self.recovery_phase = 0
            self.recovery_ticks = 0
            return None
        return self._command(0.6, self.recovery_side * p.recovery_steer * 2.0)


def create_controller() -> Controller:
    override = os.environ.get("APEX_PARAMS_JSON")  # experiments only
    if override:
        import json
        from dataclasses import replace

        return Controller(replace(DEFAULT_PARAMS, **json.loads(override)))
    return Controller()


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _softplus(value: float, width: float) -> float:
    scaled = value / width
    if scaled > 30.0:
        return value
    return width * math.log1p(math.exp(scaled))
