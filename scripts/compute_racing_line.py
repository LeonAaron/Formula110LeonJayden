"""Compute an optimized racing line for the default track (offline, one-time).

Produces, for evenly-spaced arc-length samples around the closed track loop:
  - a lateral offset from centerline (same sign convention as
    ``camera.center_offset_m``: negative is right of centerline, positive is
    left) that approximates a minimum-curvature path within the track corridor
  - a target speed at each sample, derived from the path's curvature (lateral
    grip limit) and then smoothed forward/backward for achievable
    acceleration/braking

Algorithm (a standard, simple "iterative corner-cutting" smoother used by
several open racing-line-optimization tools): repeatedly pull each path point
toward the midpoint of its immediate neighbors, projected back onto the local
track-normal direction and clamped to the track corridor. This directly
reduces path curvature (straightens corners, i.e. cuts toward the apex) while
keeping every point within the track width.

This produces a track-specific reference line. It is intentionally NOT fed to
the deployed learned controller as an input - only used offline to generate
supervised training labels (see scripts/generate_racing_line_demonstrations.py)
so the trained network only ever sees sensors, and can generalize to new
tracks it was not optimized for.

Usage:
    uv run python scripts/compute_racing_line.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from racing.physics import FORMULA_VEHICLE_PHYSICS_CONFIG, vehicle_collision_bounds
from racing.race.progress import default_track_progress_model
from racing.track.world import TRACK_WIDTH

DEFAULT_OUTPUT = Path("artifacts/racing_line.pt")
WALL_SAFETY_MARGIN_M = 0.3
SMOOTHING_ITERATIONS = 800
SMOOTHING_ALPHA = 0.3
LATERAL_ACCEL_LIMIT_MPS2 = 26.0
LONGITUDINAL_ACCEL_LIMIT_MPS2 = 18.0
BRAKING_ACCEL_LIMIT_MPS2 = 30.0
MAX_SPEED_CAP_MPS = 35.0


def _tangents_and_left_normals(xs: list[float], zs: list[float]) -> tuple[list[float], list[float]]:
    """Return the left-normal (x, z) unit vector at each point via central differences."""
    n = len(xs)
    normals_x: list[float] = []
    normals_z: list[float] = []
    for i in range(n):
        prev_i, next_i = (i - 1) % n, (i + 1) % n
        dx, dz = xs[next_i] - xs[prev_i], zs[next_i] - zs[prev_i]
        length = math.hypot(dx, dz)
        tangent_x, tangent_z = dx / length, dz / length
        normals_x.append(-tangent_z)
        normals_z.append(tangent_x)
    return normals_x, normals_z


def _menger_curvature(p_prev: tuple[float, float], p: tuple[float, float], p_next: tuple[float, float]) -> float:
    """Discrete curvature (1/radius) of the circle through three points."""
    a = math.hypot(p[0] - p_prev[0], p[1] - p_prev[1])
    b = math.hypot(p_next[0] - p[0], p_next[1] - p[1])
    c = math.hypot(p_next[0] - p_prev[0], p_next[1] - p_prev[1])
    area = abs((p[0] - p_prev[0]) * (p_next[1] - p_prev[1]) - (p[1] - p_prev[1]) * (p_next[0] - p_prev[0])) / 2.0
    denominator = a * b * c
    if denominator < 1e-9:
        return 0.0
    return 4.0 * area / denominator


def compute_minimum_curvature_offsets(
    *, xs: list[float], zs: list[float], max_offset_m: float, iterations: int, alpha: float
) -> list[float]:
    """Iteratively smooth a closed path to reduce curvature, within the track corridor."""
    n = len(xs)
    normals_x, normals_z = _tangents_and_left_normals(xs, zs)
    path_x, path_z = list(xs), list(zs)
    offsets = [0.0] * n

    for _ in range(iterations):
        new_path_x, new_path_z = list(path_x), list(path_z)
        for i in range(n):
            prev_i, next_i = (i - 1) % n, (i + 1) % n
            midpoint_x = (path_x[prev_i] + path_x[next_i]) / 2.0
            midpoint_z = (path_z[prev_i] + path_z[next_i]) / 2.0
            displacement_x, displacement_z = midpoint_x - xs[i], midpoint_z - zs[i]
            target_offset = displacement_x * normals_x[i] + displacement_z * normals_z[i]
            target_offset = max(-max_offset_m, min(max_offset_m, target_offset))
            offsets[i] += alpha * (target_offset - offsets[i])
            new_path_x[i] = xs[i] + offsets[i] * normals_x[i]
            new_path_z[i] = zs[i] + offsets[i] * normals_z[i]
        path_x, path_z = new_path_x, new_path_z

    return offsets


def compute_speed_profile(
    *,
    path_x: list[float],
    path_z: list[float],
    segment_lengths_m: list[float],
) -> list[float]:
    """Curvature-limited speed capped by achievable forward/backward acceleration."""
    n = len(path_x)
    curvatures = [
        _menger_curvature(
            (path_x[(i - 1) % n], path_z[(i - 1) % n]),
            (path_x[i], path_z[i]),
            (path_x[(i + 1) % n], path_z[(i + 1) % n]),
        )
        for i in range(n)
    ]
    curvature_speed = [
        min(MAX_SPEED_CAP_MPS, math.sqrt(LATERAL_ACCEL_LIMIT_MPS2 / kappa)) if kappa > 1e-6 else MAX_SPEED_CAP_MPS
        for kappa in curvatures
    ]

    # Forward pass: cap by achievable acceleration since the previous sample.
    forward = list(curvature_speed)
    for i in range(1, n + n):  # two laps around so the wrap-around constraint converges
        index, prev_index = i % n, (i - 1) % n
        ds = segment_lengths_m[prev_index]
        achievable = math.sqrt(max(0.0, forward[prev_index] ** 2 + 2.0 * LONGITUDINAL_ACCEL_LIMIT_MPS2 * ds))
        forward[index] = min(forward[index], achievable)

    # Backward pass: cap by achievable braking before the next sample.
    backward = list(forward)
    for i in range(n + n, 0, -1):
        index, next_index = i % n, (i + 1) % n
        ds = segment_lengths_m[index]
        achievable = math.sqrt(max(0.0, backward[next_index] ** 2 + 2.0 * BRAKING_ACCEL_LIMIT_MPS2 * ds))
        backward[index] = min(backward[index], achievable)

    return backward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=SMOOTHING_ITERATIONS)
    parser.add_argument("--alpha", type=float, default=SMOOTHING_ALPHA)
    parser.add_argument(
        "--flat-speed-mps",
        type=float,
        default=None,
        help="Override the curvature-limited profile with a constant target speed (this simulator's grip makes "
        "flat-out driving faster than slowing for corners - see LAB_NOTEBOOK.md Entry 3 and 5)",
    )
    args = parser.parse_args()

    model = default_track_progress_model()
    half_width = vehicle_collision_bounds(FORMULA_VEHICLE_PHYSICS_CONFIG).half_width
    max_offset_m = TRACK_WIDTH / 2.0 - half_width - WALL_SAFETY_MARGIN_M

    xs = [point.x for point in model.points]
    zs = [point.z for point in model.points]
    s_m = list(model.cumulative_lengths[: len(model.points)])
    segment_lengths_m = list(model.segment_lengths)

    offsets_m = compute_minimum_curvature_offsets(
        xs=xs, zs=zs, max_offset_m=max_offset_m, iterations=args.iterations, alpha=args.alpha
    )
    normals_x, normals_z = _tangents_and_left_normals(xs, zs)
    path_x = [x + offset * normal_x for x, offset, normal_x in zip(xs, offsets_m, normals_x, strict=True)]
    path_z = [z + offset * normal_z for z, offset, normal_z in zip(zs, offsets_m, normals_z, strict=True)]
    speed_mps = compute_speed_profile(path_x=path_x, path_z=path_z, segment_lengths_m=segment_lengths_m)
    if args.flat_speed_mps is not None:
        speed_mps = [args.flat_speed_mps] * len(speed_mps)

    torch.save(
        {
            "s_m": torch.tensor(s_m, dtype=torch.float32),
            "offset_m": torch.tensor(offsets_m, dtype=torch.float32),
            "speed_mps": torch.tensor(speed_mps, dtype=torch.float32),
            "total_length_m": model.total_length_m,
        },
        args.output,
    )
    print(
        f"saved {len(s_m)}-point racing line to {args.output} | "
        f"max_offset={max(abs(o) for o in offsets_m):.2f}m (limit {max_offset_m:.2f}m) | "
        f"speed range {min(speed_mps):.1f}-{max(speed_mps):.1f} m/s"
    )


if __name__ == "__main__":
    main()
