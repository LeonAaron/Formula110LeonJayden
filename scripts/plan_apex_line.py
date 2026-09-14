"""Plan the Apex controller's racing line and speed profile (offline, one-time).

Writes ``src/controllers/apex_plan.py``: a small data module with, on a fine
arc-length grid around the closed centerline,

- ``OFFSET_M``: lateral offset of the racing line from the centerline
  (positive = left of the track direction, same as ``TrackMap`` normals)
- ``SPEED_MPS``: target speed on the line at that centerline position
- ``CURVATURE``: signed path curvature (positive = turning left)

Line: minimises squared second differences of the path (a curvature proxy)
plus a tiny pull toward the centerline, solved exactly (active set) inside the
corridor the car centre can actually use. The physical barriers sit
``TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER`` (4.7 m) from the centerline; the
painted kerb strip between 3.3 m and 4.7 m is flat and drivable, so the
corridor is much wider than the 3.3 m ribbon the earlier racing-line script
assumed.

Speed: lateral-grip cap ``sqrt(a_lat / |kappa|)`` capped at ``v_max``, then a
forward pass with the measured speed-dependent acceleration and a backward
pass with a chosen braking deceleration (both measured in
scratch dynamics tests: ~13 m/s^2 accel falling to ~10 at 50 m/s; braking
above ~15 m/s^2 gets directionally unstable).

Usage:
    uv run python scripts/plan_apex_line.py --corridor-margin 0.7 --a-lat 30
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from racing.graphics.track_rendering import TRACK_EDGE_BUFFER
from racing.physics import FORMULA_VEHICLE_PHYSICS_CONFIG, vehicle_collision_bounds
from racing.race.progress import default_track_progress_model, track_heading_at_distance, track_pose_at_distance
from racing.track.world import TRACK_WIDTH

OUTPUT = Path("src/controllers/apex_plan.py")
GRID_STEP_M = 0.5


def build_centerline(step_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = default_track_progress_model()
    count = round(model.total_length_m / step_m)
    s = np.linspace(0.0, model.total_length_m, count, endpoint=False)
    cx = np.empty(count)
    cz = np.empty(count)
    lx = np.empty(count)
    lz = np.empty(count)
    for i, value in enumerate(s):
        pose = track_pose_at_distance(model, float(value))
        cx[i], cz[i] = pose.position.x, pose.position.z
        h = math.radians(track_heading_at_distance(model, float(value)))
        lx[i], lz[i] = -math.cos(h), math.sin(h)
    return s, cx, cz, lx, lz


def _path_energy(
    d: torch.Tensor,
    cx: torch.Tensor,
    cz: torch.Tensor,
    lx: torch.Tensor,
    lz: torch.Tensor,
    *,
    mode: str,
    speed_args: dict[str, float],
) -> torch.Tensor:
    px, pz = cx + d * lx, cz + d * lz
    tx, tz = torch.roll(px, -1) - px, torch.roll(pz, -1) - pz
    seg = torch.sqrt(tx * tx + tz * tz + 1e-12)
    heading = torch.atan2(tx, tz)
    dh = torch.roll(heading, -1) - heading
    dh = torch.remainder(dh + math.pi, 2 * math.pi) - math.pi
    ds = 0.5 * (seg + torch.roll(seg, -1))
    kappa = dh / ds
    if mode == "curvature":
        return torch.sum(kappa.abs() ** speed_args.get("power", 2.0) * ds)
    # lap time under the speed model (differentiable version of speed_profile)
    v = torch.clamp(torch.sqrt(speed_args["a_lat"] / torch.clamp(kappa.abs(), min=1e-6)), max=speed_args["v_max"])
    n = len(v)
    for _ in range(2):
        vals = list(v.unbind(0))
        for i in range(n):
            j = (i + 1) % n
            a = torch.clamp(speed_args["a_acc_base"] - speed_args["a_acc_slope"] * vals[i], min=2.0)
            vals[j] = torch.minimum(vals[j], torch.sqrt(vals[i] ** 2 + 2 * a * ds[i]))
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            vals[i] = torch.minimum(vals[i], torch.sqrt(vals[j] ** 2 + 2 * speed_args["a_brake"] * ds[i]))
        v = torch.stack(vals)
    vm = 0.5 * (v + torch.roll(v, -1))
    return torch.sum(ds / vm)


def inside_limits(
    cx: np.ndarray, cz: np.ndarray, *, max_offset_m: float, inside_margin_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-point (left_limit, right_limit) for the offset: the corridor, tightened on the inside of
    a bend so the offset never reaches the centerline's own centre of curvature (which would fold the
    path into a cusp). Curvature is smoothed over a few samples first."""
    kappa, _ = path_curvature(cx, cz)  # positive = turning left
    # Conservative: use the sharpest curvature within +/-3 samples (the sampled centerline is a
    # polyline whose bends concentrate at vertices, so an average would hide them).
    padded = np.concatenate((kappa[-3:], kappa, kappa[:3]))
    windows = np.stack([padded[i : i + len(kappa)] for i in range(7)])
    magnitude = np.max(np.abs(windows), axis=0)
    sign = np.sign(np.sum(windows, axis=0))
    kappa = sign * magnitude
    radius = 1.0 / np.maximum(magnitude, 1e-6)
    inside = np.maximum(0.0, np.minimum(max_offset_m, radius - inside_margin_m))
    left_limit = np.where(kappa > 0, inside, max_offset_m)  # left turn: left side is the inside
    right_limit = np.where(kappa < 0, inside, max_offset_m)
    return left_limit, right_limit


def optimize_offsets(
    cx: np.ndarray,
    cz: np.ndarray,
    lx: np.ndarray,
    lz: np.ndarray,
    *,
    max_offset_m: float,
    inside_margin_m: float = 1.5,
    center_weight: float,
    iterations: int,
    mode: str = "curvature",
    speed_args: dict[str, float] | None = None,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Nonlinear minimisation of the true discrete curvature energy (or modeled lap time).

    Offsets are parametrised as ``max_offset * tanh(u)`` so the corridor is a
    hard constraint, and optimised with L-BFGS through torch autograd.
    """
    t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    cxt, czt, lxt, lzt = t(cx), t(cz), t(lx), t(lz)
    left_limit, right_limit = inside_limits(cx, cz, max_offset_m=max_offset_m, inside_margin_m=inside_margin_m)
    # d = centre + half_span * tanh(u) maps u onto [-right_limit, +left_limit]
    centre = t(0.5 * (left_limit - right_limit))
    half_span = t(0.5 * (left_limit + right_limit))
    if init is None:
        u = torch.zeros(len(cx), dtype=torch.float64, requires_grad=True)
    else:
        ratio = np.clip((init - centre.numpy()) / np.maximum(half_span.numpy(), 1e-6), -0.999, 0.999)
        u = torch.tensor(np.arctanh(ratio), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([u], lr=0.5, max_iter=iterations, history_size=30, line_search_fn="strong_wolfe")
    args = speed_args or {}

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        d = centre + half_span * torch.tanh(u)
        loss = _path_energy(d, cxt, czt, lxt, lzt, mode=mode, speed_args=args) + center_weight * torch.sum(d * d)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return (centre + half_span * torch.tanh(u)).numpy()


def path_curvature(px: np.ndarray, pz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed curvature (positive = turning left) and segment lengths of a closed path."""
    tx = np.roll(px, -1) - px
    tz = np.roll(pz, -1) - pz
    seg = np.hypot(tx, tz)
    heading = np.arctan2(tx, tz)  # simulator convention: 0 along +Z, positive toward +X (right turn)
    dh = np.roll(heading, -1) - heading
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    ds = 0.5 * (seg + np.roll(seg, -1))
    # positive dh means heading rotating toward +X = right turn; left-positive curvature is -dh/ds
    kappa = -dh / ds
    # average onto points
    kappa = 0.5 * (kappa + np.roll(kappa, 1))
    return kappa, seg


def accel_limit(v: float, base: float, slope: float) -> float:
    return max(2.0, base - slope * v)


def speed_profile(
    kappa: np.ndarray,
    seg: np.ndarray,
    *,
    a_lat: float,
    v_max: float,
    a_acc_base: float,
    a_acc_slope: float,
    a_brake: float,
) -> np.ndarray:
    n = len(kappa)
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-6)))
    for _ in range(3):
        for i in range(n):
            j = (i + 1) % n
            a = accel_limit(v[i], a_acc_base, a_acc_slope)
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2 * a * seg[i]))
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * a_brake * seg[i]))
    return v


def lap_time(v: np.ndarray, seg: np.ndarray) -> float:
    vm = 0.5 * (v + np.roll(v, -1))
    return float(np.sum(seg / vm))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corridor-margin", type=float, default=0.5, help="Clearance kept from the barrier face")
    parser.add_argument(
        "--inside-margin", type=float, default=1.5, help="Keep the line this far from a bend's centre of curvature"
    )
    parser.add_argument("--center-weight", type=float, default=1e-6)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--mode", choices=("curvature", "laptime", "both"), default="both")
    parser.add_argument("--a-lat", type=float, default=30.0)
    parser.add_argument("--v-max", type=float, default=34.0)
    parser.add_argument("--a-acc-base", type=float, default=13.0)
    parser.add_argument("--a-acc-slope", type=float, default=0.07)
    parser.add_argument("--a-brake", type=float, default=12.0)
    parser.add_argument("--power", type=float, default=2.0, help="Exponent on |curvature| in the line objective")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    half_width = vehicle_collision_bounds(FORMULA_VEHICLE_PHYSICS_CONFIG).half_width
    barrier_m = TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER
    max_offset_m = barrier_m - half_width - args.corridor_margin

    s, cx, cz, lx, lz = build_centerline(GRID_STEP_M)
    track_length_m = default_track_progress_model().total_length_m
    grid_step_m = track_length_m / len(s)
    speed_args = {
        "a_lat": args.a_lat,
        "v_max": args.v_max,
        "a_acc_base": args.a_acc_base,
        "a_acc_slope": args.a_acc_slope,
        "a_brake": args.a_brake,
        "power": args.power,
    }
    d = optimize_offsets(
        cx,
        cz,
        lx,
        lz,
        max_offset_m=max_offset_m,
        inside_margin_m=args.inside_margin,
        center_weight=args.center_weight,
        iterations=args.iterations,
        mode="curvature",
        speed_args=speed_args,
    )
    if args.mode in ("laptime", "both"):
        d_curv = d
        px, pz = cx + d * lx, cz + d * lz
        kappa, seg = path_curvature(px, pz)
        v = speed_profile(kappa, seg, **speed_args)
        print(f"after curvature stage: min radius {1 / np.max(np.abs(kappa)):.2f} m, lap time {lap_time(v, seg):.2f} s")
        d = optimize_offsets(
            cx,
            cz,
            lx,
            lz,
            max_offset_m=max_offset_m,
            inside_margin_m=args.inside_margin,
            center_weight=args.center_weight,
            iterations=args.iterations // 4,
            mode="laptime",
            speed_args=speed_args,
            init=None if args.mode == "laptime" else d_curv,
        )
    left_limit, right_limit = inside_limits(cx, cz, max_offset_m=max_offset_m, inside_margin_m=args.inside_margin)
    px, pz = cx + d * lx, cz + d * lz
    kappa, seg = path_curvature(px, pz)
    v = speed_profile(
        kappa,
        seg,
        a_lat=args.a_lat,
        v_max=args.v_max,
        a_acc_base=args.a_acc_base,
        a_acc_slope=args.a_acc_slope,
        a_brake=args.a_brake,
    )
    t_lap = lap_time(v, seg)
    total = float(np.sum(seg))
    print(
        f"max|d|={np.max(np.abs(d)):.2f} frac_at_wall={np.mean(np.abs(d) > max_offset_m - 1e-6):.2f} | "
        f"corridor +/-{max_offset_m:.2f} m | path length {total:.1f} m (centerline {track_length_m:.1f}) | "
        f"min seg {seg.min():.3f} m | min radius {1 / np.max(np.abs(kappa)):.2f} m | speed {v.min():.1f}-{v.max():.1f} m/s | "
        f"lap time {t_lap:.2f} s -> {30 / t_lap:.2f} laps in 30 s (from a rolling start)"
    )
    if args.dry_run:
        return

    def fmt(values: np.ndarray) -> str:
        return (
            "(\n"
            + "\n".join(
                "    " + ", ".join(f"{x:.4f}" for x in values[i : i + 8]) + "," for i in range(0, len(values), 8)
            )
            + "\n)"
        )

    body = f'''"""Generated by scripts/plan_apex_line.py — do not edit by hand.

Racing line and speed profile for the default track, sampled every
{grid_step_m:.4f} m of centerline arc length. See the planner script for the method
and the parameters used:
corridor_margin={args.corridor_margin} inside_margin={args.inside_margin} center_weight={args.center_weight} a_lat={args.a_lat}
power={args.power} v_max={args.v_max} a_acc_base={args.a_acc_base} a_acc_slope={args.a_acc_slope} a_brake={args.a_brake}
"""

GRID_STEP_M = {grid_step_m:.8f}
TRACK_LENGTH_M = {track_length_m:.8f}
MAX_OFFSET_M = {max_offset_m:.4f}
A_LAT_MPS2 = {args.a_lat}
A_BRAKE_MPS2 = {args.a_brake}
V_MAX_MPS = {args.v_max}

OFFSET_M: tuple[float, ...] = {fmt(d)}

SPEED_MPS: tuple[float, ...] = {fmt(v)}

CURVATURE: tuple[float, ...] = {fmt(kappa)}

# Per-point lateral limits (metres): how far left / right of the centerline the car centre may go
# without folding into the bend's centre of curvature or leaving the corridor.
LEFT_LIMIT_M: tuple[float, ...] = {fmt(left_limit)}

RIGHT_LIMIT_M: tuple[float, ...] = {fmt(right_limit)}
'''
    args.output.write_text(body)
    subprocess.run([sys.executable, "-m", "ruff", "format", "--isolated", str(args.output)], check=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
