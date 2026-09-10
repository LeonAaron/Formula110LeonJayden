"""Track map and sensor-only localization for the Apex controller.

The simulator never hands a controller its official race progress. This module
recovers the car's arc-length position ``s`` along the centerline purely from
public sensor fields plus the public track layout exported by ``racing``:

- ``imu.heading_degrees`` (world yaw) and ``camera.heading_error_degrees``
  together give the local track heading, which the map turns into a set of
  candidate positions.
- ``camera.center_offset_m`` fixes the car's lateral position for each
  candidate, and the three ``camera.lookahead_offsets_m`` values (track-center
  points 4, 9 and 16 m ahead, expressed in the car frame) are predicted for
  every candidate and compared with the observed values.
- The candidate with the smallest residual wins. Once locked in, only a window
  around the dead-reckoned estimate is searched each tick; a global search is
  repeated whenever the best local residual is implausible.

Everything is precomputed on a fine arc-length grid with the exact same
centerline helpers the simulator uses for its camera sensor, so a correct
hypothesis reproduces the observation almost exactly. Pure Python on purpose:
the grading sandbox ships only the simulator's own dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from racing import RobotSensors
from racing.race.progress import (
    TrackProgressModel,
    default_track_progress_model,
    track_heading_at_distance,
    track_pose_at_distance,
)

GRID_STEP_M = 0.2
LOOKAHEAD_DISTANCES_M = (4.0, 9.0, 16.0)


def _forward_vector(heading_degrees: float) -> tuple[float, float]:
    radians = math.radians(heading_degrees)
    return math.sin(radians), math.cos(radians)


def _wrap_signed(value: float, period: float) -> float:
    return (value + period / 2) % period - period / 2


@dataclass(frozen=True, slots=True)
class Localization:
    s_m: float
    # Signed distance from the centerline, positive = left of the track direction.
    lateral_m: float
    residual: float
    heading_track_degrees: float


class TrackMap:
    """Fine-grained lookup tables for the default track."""

    def __init__(
        self, model: TrackProgressModel | None = None, grid_step_m: float = GRID_STEP_M
    ) -> None:
        self.model = default_track_progress_model() if model is None else model
        self.length_m = self.model.total_length_m
        self.count = round(self.length_m / grid_step_m)
        self.step_m = self.length_m / self.count
        self.s = [index * self.step_m for index in range(self.count)]

        self.cx: list[float] = []
        self.cz: list[float] = []
        self.heading: list[float] = []
        self.left_x: list[float] = []
        self.left_z: list[float] = []
        self.fwd_x: list[float] = []
        self.fwd_z: list[float] = []
        # Lookahead point coordinates, one list per lookahead distance.
        self.lax: list[list[float]] = [[] for _ in LOOKAHEAD_DISTANCES_M]
        self.laz: list[list[float]] = [[] for _ in LOOKAHEAD_DISTANCES_M]
        for s in self.s:
            pose = track_pose_at_distance(self.model, s)
            heading = track_heading_at_distance(self.model, s)
            self.cx.append(pose.position.x)
            self.cz.append(pose.position.z)
            self.heading.append(heading)
            radians = math.radians(heading)
            # track forward = (sin h, cos h); left = (-cos h, sin h)
            self.fwd_x.append(math.sin(radians))
            self.fwd_z.append(math.cos(radians))
            self.left_x.append(-math.cos(radians))
            self.left_z.append(math.sin(radians))
            for k, distance in enumerate(LOOKAHEAD_DISTANCES_M):
                ahead = track_pose_at_distance(self.model, s + distance)
                self.lax[k].append(ahead.position.x)
                self.laz[k].append(ahead.position.z)

    def wrap(self, s: float) -> float:
        return s % self.length_m

    def signed_delta(self, a: float, b: float) -> float:
        """Shortest signed arc distance from b to a."""
        return _wrap_signed(a - b, self.length_m)

    def center_at(self, s: float) -> tuple[float, float]:
        pose = track_pose_at_distance(self.model, s)
        return pose.position.x, pose.position.z

    def heading_at(self, s: float) -> float:
        return track_heading_at_distance(self.model, s)

    def left_at(self, s: float) -> tuple[float, float]:
        h = math.radians(self.heading_at(s))
        return -math.cos(h), math.sin(h)

    def window_indices(self, center_s: float, window_m: float) -> range:
        """Grid indices within ``window_m`` of ``center_s`` (wrap-around handled by the caller via %)."""
        center = round(center_s / self.step_m)
        half = int(math.ceil(window_m / self.step_m))
        return range(center - half, center + half + 1)

    def localize(
        self,
        sensors: RobotSensors,
        prior_s: float | None,
        *,
        prior_window_m: float = 5.0,
    ) -> Localization:
        """Find the arc-length position best explaining the current camera observation.

        With ``prior_s`` only the grid cells within ``prior_window_m`` are searched; without it
        the whole track is searched.
        """
        psi = sensors.imu.heading_degrees
        err = sensors.camera.heading_error_degrees
        center = sensors.camera.center_offset_m
        offsets = sensors.camera.lookahead_offsets_m
        beams = min(len(offsets), len(LOOKAHEAD_DISTANCES_M))

        fx, fz = _forward_vector(psi)
        rx, rz = fz, -fx  # car right vector
        track_heading_obs = psi + err

        indices = (
            range(self.count)
            if prior_s is None
            else self.window_indices(prior_s, prior_window_m)
        )
        count = self.count
        cx, cz, lx, lz, headings = (
            self.cx,
            self.cz,
            self.left_x,
            self.left_z,
            self.heading,
        )
        lax, laz = self.lax, self.laz

        best_index = 0
        best_residual = float("inf")
        best_d = 0.0
        for raw_index in indices:
            index = raw_index % count
            # center = (C - P) . r with P = C + d * left  =>  center = -d (left . r)
            denom = lx[index] * rx + lz[index] * rz
            perpendicular = abs(denom) < 0.2
            if perpendicular:
                denom = 0.2 if denom >= 0.0 else -0.2
            d = -center / denom
            px = cx[index] + d * lx[index]
            pz = cz[index] + d * lz[index]
            residual = 0.0
            for k in range(beams):
                pred = (lax[k][index] - px) * rx + (laz[k][index] - pz) * rz
                diff = pred - offsets[k]
                residual += diff * diff
            heading_res = (
                _wrap_signed(headings[index] - track_heading_obs, 360.0) / 10.0
            )
            residual += (
                heading_res * heading_res
            )  # 10 degrees of heading error ~ 1 m of lookahead error
            if perpendicular:
                residual += 25.0
            if residual < best_residual:
                best_residual = residual
                best_index = index
                best_d = d
        return Localization(
            s_m=self.s[best_index],
            lateral_m=best_d,
            residual=best_residual,
            heading_track_degrees=headings[best_index],
        )
