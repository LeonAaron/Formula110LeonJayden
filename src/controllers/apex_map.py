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
- The candidate with the smallest residual wins; a weak prior around the
  dead-reckoned estimate keeps the answer continuous once locked in.

Everything is precomputed on a fine arc-length grid with the exact same
centerline helpers the simulator uses for its camera sensor, so a correct
hypothesis reproduces the observation almost exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from racing import RobotSensors
from racing.race.progress import (
    TrackProgressModel,
    default_track_progress_model,
    track_heading_at_distance,
    track_pose_at_distance,
)

GRID_STEP_M = 0.2


def _forward_vector(heading_degrees: float) -> tuple[float, float]:
    radians = math.radians(heading_degrees)
    return math.sin(radians), math.cos(radians)


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
        count = round(self.length_m / grid_step_m)
        self.s = np.linspace(0.0, self.length_m, count, endpoint=False)
        lookahead = (4.0, 9.0, 16.0)
        self.lookahead_distances = lookahead

        cx = np.empty(count)
        cz = np.empty(count)
        heading = np.empty(count)
        lax = np.empty((len(lookahead), count))
        laz = np.empty((len(lookahead), count))
        for index, s in enumerate(self.s):
            pose = track_pose_at_distance(self.model, float(s))
            cx[index] = pose.position.x
            cz[index] = pose.position.z
            heading[index] = track_heading_at_distance(self.model, float(s))
            for k, distance in enumerate(lookahead):
                ahead = track_pose_at_distance(self.model, float(s) + distance)
                lax[k, index] = ahead.position.x
                laz[k, index] = ahead.position.z
        self.cx, self.cz, self.heading = cx, cz, heading
        self.lax, self.laz = lax, laz
        rad = np.radians(heading)
        # track forward = (sin h, cos h); left = (-cos h, sin h)
        self.left_x = -np.cos(rad)
        self.left_z = np.sin(rad)
        self.fwd_x = np.sin(rad)
        self.fwd_z = np.cos(rad)

    def wrap(self, s: float) -> float:
        return s % self.length_m

    def signed_delta(self, a: float, b: float) -> float:
        """Shortest signed arc distance from b to a."""
        d = (a - b) % self.length_m
        if d > self.length_m / 2:
            d -= self.length_m
        return d

    def center_at(self, s: float) -> tuple[float, float]:
        pose = track_pose_at_distance(self.model, s)
        return pose.position.x, pose.position.z

    def heading_at(self, s: float) -> float:
        return track_heading_at_distance(self.model, s)

    def left_at(self, s: float) -> tuple[float, float]:
        h = math.radians(self.heading_at(s))
        return -math.cos(h), math.sin(h)

    def localize(
        self,
        sensors: RobotSensors,
        prior_s: float | None,
        *,
        prior_weight: float = 0.02,
        prior_window_m: float | None = None,
    ) -> Localization:
        """Find the arc-length position best explaining the current camera observation."""
        psi = sensors.imu.heading_degrees
        err = sensors.camera.heading_error_degrees
        center = sensors.camera.center_offset_m
        offsets = sensors.camera.lookahead_offsets_m

        fx, fz = _forward_vector(psi)
        rx, rz = fz, -fx  # car right vector

        if prior_window_m is not None and prior_s is not None:
            delta = np.abs(_wrap_signed(self.s - prior_s, self.length_m))
            mask = delta <= prior_window_m
        else:
            mask = np.ones(len(self.s), dtype=bool)

        s = self.s[mask]
        cx, cz = self.cx[mask], self.cz[mask]
        lx, lz = self.left_x[mask], self.left_z[mask]
        heading = self.heading[mask]

        # center = (C - P) . r with P = C + d * left  =>  center = -d (left . r)
        denom = lx * rx + lz * rz
        safe = np.where(
            np.abs(denom) < 0.2, np.sign(denom) * 0.2 + (denom == 0) * 0.2, denom
        )
        d = -center / safe
        px = cx + d * lx
        pz = cz + d * lz

        residual = np.zeros(len(s))
        for k in range(len(self.lookahead_distances)):
            if k >= len(offsets):
                break
            pred = (self.lax[k][mask] - px) * rx + (self.laz[k][mask] - pz) * rz
            residual += (pred - offsets[k]) ** 2
        heading_res = _wrap_signed(heading - psi - err, 360.0)
        residual += (
            heading_res / 10.0
        ) ** 2  # 10 degrees of heading error ~ 1 m of lookahead error
        residual += np.where(np.abs(denom) < 0.2, 25.0, 0.0)

        if prior_s is not None and prior_weight > 0.0:
            delta = _wrap_signed(s - prior_s, self.length_m)
            residual = residual + prior_weight * delta**2

        best = int(np.argmin(residual))
        return Localization(
            s_m=float(s[best]),
            lateral_m=float(d[best]),
            residual=float(residual[best]),
            heading_track_degrees=float(heading[best]),
        )


def _wrap_signed(values: np.ndarray, period: float) -> np.ndarray:
    return (values + period / 2) % period - period / 2
