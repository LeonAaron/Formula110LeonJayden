"""Reactive rule-based controller: sensor-driven throttle and steering.

All tunable gains and thresholds live in ``ReactiveParams`` so a future
parameter-search step (random search, Optuna, CMA-ES, ...) can explore the
knob space by calling :func:`drive` with different parameter sets, without
touching the control logic itself.

Design notes:

- Steering targets an "apex" offset biased toward the inside of the upcoming
  turn (right for a right turn, left for a left turn), inferred from the sign
  and magnitude of ``camera.heading_error_degrees``. Because the simulator
  scores forward progress as centerline projection rather than physical path
  length, cutting the inside of a turn advances scored distance per meter
  actually driven.
- Throttle targets ``max_speed_mps`` by default, with two optional,
  independently-tunable adjustments (each off by default via a 0 gain):
  proactive corner slow-down (``corner_speed_gain``, driven by anticipated
  heading error or actual yaw rate, whichever is sharper) and a
  speed-squared term in the braking-distance formula (``brake_quadratic_coeff``,
  matching true constant-deceleration stopping distance). The reactive,
  wall-proximity braking check remains the hard safety floor against
  destruction regardless of these settings.

New in reactive2.py — sensor-expansion pass (LAB_NOTEBOOK.md Entries 3-6 document
a plateau at ~442-445m/30s for reactive.py; this file adds sensors reactive.py
never reads, each gated by its own off-by-default gain so that all-gains-zero
reproduces reactive.py's behavior bit-for-bit):

- Traction / wall-scrape / ineffective-braking detector (B): reads
  ``imu.forward_acceleration_mps2`` (a real measurement of what actually
  happened last tick, independent of what throttle commanded) to detect
  wheelspin, wall-scrape, or a brake that isn't actually slowing the car.
- Diagonal early-warning wall avoidance (C): reads the raw +-45 degree
  wall-lidar beams (already-existing beams, previously unused) as a softer,
  earlier layer before the existing +-90-degree-based avoidance and the
  close-range emergency override.
- Three-point curvature/anticipation signal (D): uses all three
  ``lookahead_offsets_m``/``lookahead_distances_m`` points (only the first two
  are used elsewhere in this file) to estimate how fast the track's bend is
  tightening, not just its current bend. Deliberately independent of
  ``max_speed_mps``/``apex_bias_max_m`` to avoid Entry 4's confound (an
  earlier attempt combined an anticipatory apex bias with a speed increase in
  one search and could not attribute credit to either change).
- Damage-based caution scaler (E): reads ``contact.damage`` as a throttle
  input for the first time (previously ``drive()`` only reads
  ``contact.any_contact`` to gate recovery mode).
- Roll/pitch safety cutoff (F): reads ``imu.roll_degrees``/``pitch_degrees``.
  Expected inert on this flat track, included for complete sensor coverage.
- Speed-scaled steering safety margins (G): the wall-avoidance, diagonal-
  avoidance, and emergency-front margins all grow with current speed, the
  same time-based-lead idea already used by ``brake_lead_time_s`` for
  braking, applied here to steering avoidance instead.
- Instability/skid stability-recovery override (H): an AND-fusion of yaw rate
  and lateral acceleration, both independently far past ordinary-cornering
  levels, catches an actual spin/slide event and applies a corrective
  throttle cut plus countersteer -- distinct from the existing
  contact-triggered ``_recovery_command``, which remains the hard fallback
  for actual contact. H is meant to catch a skid before it becomes contact.
- Predictive multi-beam braking (I): the brake-distance check can blend in
  the front-left/front-right beams (``min(front_m, front_left_m,
  front_right_m)``), not just the single center beam, so a wall approached at
  an angle is caught even while the center beam alone still reads clear.

Every new gain above defaults to 0.0 (or an equivalent neutral value), and
every new mechanism composes additively (steering) or multiplicatively by a
factor of 1.0 at gain 0.0 (throttle), so ``ReactiveParams()`` at its defaults
must reproduce reactive.py's current behavior exactly, tick for tick.

(A "lateral-acceleration cornering governor" mechanism was tried and removed
during Step 1 sanity-checking: ``imu.lateral_acceleration_mps2`` turned out
to equal ``speed_mps * radians(yaw_rate_degrees_per_s)`` in this simulator —
a deterministic combination of quantities already feeding
``corner_speed_gain`` via :func:`_corner_speed_signal`, not an independent
grip/slip measurement as first assumed — and it regressed distance by
17-46% in isolated testing, similar to ``corner_speed_gain`` itself.)
"""

from __future__ import annotations

from dataclasses import dataclass

from racing import CameraSensors, LidarSensors, RobotCommand, RobotSensors

RACING_NAME = "Reactive Pilot"
RACING_COLOR = "#39D98A"


@dataclass(frozen=True, slots=True)
class ReactiveParams:
    """Tunable gains and thresholds for the reactive controller.

    Defaults for the fields through ``brake_quadratic_coeff`` are the result
    of scripts/optimize_reactive.py (bounded two-phase evolution strategy,
    trained on 5 seeds). See LAB_NOTEBOOK.md, Entry 6. Validated: 50/50
    survived, 0.00 damage in every race across 5 training seeds
    (13, 55, 110, 271, 997) and 5 disjoint validation seeds (42, 2027, 8675,
    31415, 777001), avg. ~442-445 m/30s (up from a prior ~432-435 m).

    Every field below ``brake_quadratic_coeff`` is new in reactive2.py (see
    the module docstring) and defaults to a neutral/off value, so
    ``ReactiveParams()`` reproduces the ~442-445m baseline exactly until
    scripts/optimize_reactive2.py tunes them (LAB_NOTEBOOK.md, Entry 7).
    """

    # Steering: follow the track centerline and upcoming curvature.
    center_offset_gain: float = 0.09437765812661737
    heading_error_gain: float = 0.05761283755797004
    lookahead_near_gain: float = 0.09033539947299812
    lookahead_far_gain: float = 0.05150093277359806

    # Steering: nudge away from a close side wall before it becomes urgent.
    wall_avoid_margin_m: float = 0.9984770482614183
    wall_avoid_gain: float = 0.33676478535712756

    # Steering: override toward open space when a wall is immediately ahead.
    emergency_front_m: float = 2.500129543202483
    emergency_steer: float = 0.8733926601175921

    # Steering: back away from active contact toward whichever side is open.
    recovery_steer: float = 1.183216644687921
    recovery_throttle: float = -0.21430855823458606

    steer_limit: float = 1.0

    # Steering: infer turn sharpness from heading error and hug the inside
    # of the turn (right side on a right turn, left side on a left turn).
    turn_sharpness_deg: float = 7.9347978275593345
    apex_bias_max_m: float = 0.31137445196443336

    # Throttle: target speed, optionally reduced proactively for an anticipated
    # turn (camera heading error) or an actual one already underway (yaw
    # rate). corner_speed_gain settled near 0 in search — negligible effect.
    max_speed_mps: float = 18.194525474628353
    speed_gain: float = 0.33561427559043133
    corner_speed_gain: float = 0.021591851522117442
    corner_signal_deg: float = 41.85343568146501
    corner_yaw_rate_deg_per_s: float = 49.77802577209076

    # Throttle: brake in time for the wall directly ahead. Distance is
    # linear-in-speed plus an optional speed-squared term (true stopping
    # distance under constant deceleration is quadratic in speed);
    # brake_quadratic_coeff settled at 0 in search — not useful here.
    brake_lead_time_s: float = 0.23534499772289047
    brake_min_distance_m: float = 0.2620843869496009
    brake_gain: float = 1.5356470476775665
    brake_quadratic_coeff: float = 0.0

    # --- New in reactive2.py: B. traction / wall-scrape / ineffective-brake
    # detector --- Fires when the car commands meaningful throttle in either
    # direction (accelerating or braking) but imu.forward_acceleration_mps2
    # doesn't match: wheelspin, a wall-scrape robbing forward progress, or a
    # brake that isn't actually slowing the car.
    traction_loss_throttle_threshold: float = 0.4
    traction_loss_expected_accel_mps2: float = 1.0
    traction_loss_gain: float = 0.0

    # --- New in reactive2.py: C. diagonal early-warning wall avoidance ---
    # Uses the raw +-45 degree wall-lidar beams (already-existing beams,
    # unused elsewhere in this file) as a softer, earlier layer than the
    # +-90-degree-based wall_avoid_* terms above.
    diagonal_avoid_margin_m: float = 2.5
    diagonal_avoid_gain: float = 0.0

    # --- New in reactive2.py: D. three-point curvature/anticipation signal
    # --- Uses all three camera.lookahead_offsets_m/lookahead_distances_m
    # points (only the first two are used above) to estimate how fast the
    # track's bend is tightening. Deliberately independent of max_speed_mps
    # and apex_bias_max_m (see LAB_NOTEBOOK.md Entry 4 on confounded search).
    curvature_gain: float = 0.0

    # --- New in reactive2.py: E. damage-based caution scaler --- Reduces
    # target speed in proportion to accumulated non-fatal contact.damage,
    # which drive() otherwise never consults once active contact ends.
    damage_speed_gain: float = 0.0

    # --- New in reactive2.py: F. roll/pitch safety cutoff --- Expected
    # inert on this flat track; included for complete sensor coverage.
    attitude_cutoff_deg: float = 15.0
    attitude_cutoff_gain: float = 0.0

    # --- New in reactive2.py: G. speed-scaled steering safety margins ---
    # Extra margin, in meters, added to wall_avoid_margin_m,
    # diagonal_avoid_margin_m, and emergency_front_m as speed_mps times this
    # value — the same time-based-lead idea brake_lead_time_s already uses
    # for braking, applied here to steering avoidance instead.
    speed_scaled_margin_lead_s: float = 0.0

    # --- New in reactive2.py: H. instability/skid stability-recovery
    # override --- An AND-fusion: both yaw rate and lateral acceleration
    # must independently exceed levels far past ordinary cornering before
    # this fires, so it only engages on a genuine spin/slide, not a normal
    # hard turn. Distinct from the existing contact-triggered
    # _recovery_command (the hard fallback for contact that has already
    # happened) — H is meant to catch a skid before it becomes contact.
    instability_yaw_rate_deg_per_s: float = 200.0
    instability_lateral_accel_mps2: float = 8.0
    instability_gain: float = 0.0

    # --- New in reactive2.py: I. predictive multi-beam braking --- Blends
    # the brake-distance check's front reading between wall.front_m alone
    # (blend=0, current behavior) and the closest of the front cone beams
    # (blend=1), catching a wall approached at an angle.
    brake_front_cone_blend: float = 0.0


DEFAULT_PARAMS = ReactiveParams()


def control(sensors: RobotSensors) -> RobotCommand:
    """Map one sensor snapshot to a throttle/steer command."""
    return drive(sensors, DEFAULT_PARAMS)


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
    speed_mps = sensors.odometry.speed_mps

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

    steer += _wall_avoidance_term(wall, params, speed_mps)
    steer += _diagonal_wall_avoidance_term(wall, params, speed_mps)
    steer += _curvature_steer_term(camera, params)
    steer += _instability_recovery_steer_term(sensors, params)

    emergency_margin_m = params.emergency_front_m + speed_mps * params.speed_scaled_margin_lead_s
    if wall.front_m < emergency_margin_m:
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


def _wall_avoidance_term(wall: LidarSensors, params: ReactiveParams, speed_mps: float) -> float:
    margin_m = params.wall_avoid_margin_m + speed_mps * params.speed_scaled_margin_lead_s
    term = 0.0
    if wall.left_m < margin_m:
        term += params.wall_avoid_gain * (margin_m - wall.left_m)
    if wall.right_m < margin_m:
        term -= params.wall_avoid_gain * (margin_m - wall.right_m)
    return term


def _diagonal_wall_avoidance_term(wall: LidarSensors, params: ReactiveParams, speed_mps: float) -> float:
    """Softer, earlier counterpart to :func:`_wall_avoidance_term` using the raw +-45 degree beams.

    A no-op at ``diagonal_avoid_gain == 0.0`` (the default).
    """
    margin_m = params.diagonal_avoid_margin_m + speed_mps * params.speed_scaled_margin_lead_s
    left_45_m = wall.distance_at_angle_degrees(-45.0)
    right_45_m = wall.distance_at_angle_degrees(45.0)
    term = 0.0
    if left_45_m < margin_m:
        term += params.diagonal_avoid_gain * (margin_m - left_45_m)
    if right_45_m < margin_m:
        term -= params.diagonal_avoid_gain * (margin_m - right_45_m)
    return term


def _curvature_steer_term(camera: CameraSensors, params: ReactiveParams) -> float:
    """Return a steer term proportional to how fast the upcoming bend is tightening.

    Uses all three ``lookahead_offsets_m``/``lookahead_distances_m`` points
    (near and far) as a discrete second derivative of track curvature: the
    difference between the far-segment slope and the near-segment slope. A
    no-op at ``curvature_gain == 0.0`` (the default) or when fewer than three
    lookahead points are available.
    """
    if not camera.visible:
        return 0.0
    offsets = camera.lookahead_offsets_m
    distances = camera.lookahead_distances_m
    if len(offsets) < 3 or len(distances) < 3:
        return 0.0
    near_run_m = distances[1] - distances[0]
    far_run_m = distances[2] - distances[1]
    if near_run_m <= 0.0 or far_run_m <= 0.0:
        return 0.0
    slope_near = (offsets[1] - offsets[0]) / near_run_m
    slope_far = (offsets[2] - offsets[1]) / far_run_m
    curvature_signal = slope_far - slope_near
    return params.curvature_gain * curvature_signal


def _instability_detected(sensors: RobotSensors, params: ReactiveParams) -> bool:
    """Return True only when yaw rate AND lateral acceleration both indicate a genuine skid.

    An AND-fusion, not a max/OR: either signal alone can be large during
    ordinary hard cornering, but both being large simultaneously and far past
    ordinary-cornering levels is a stronger, rarer signature of actual loss
    of control.
    """
    yaw_rate_deg_per_s = abs(sensors.imu.yaw_rate_degrees_per_s)
    lateral_accel_mps2 = abs(sensors.imu.lateral_acceleration_mps2)
    return (
        yaw_rate_deg_per_s > params.instability_yaw_rate_deg_per_s
        and lateral_accel_mps2 > params.instability_lateral_accel_mps2
    )


def _instability_recovery_steer_term(sensors: RobotSensors, params: ReactiveParams) -> float:
    """Countersteer against the skid direction when :func:`_instability_detected` fires.

    A no-op at ``instability_gain == 0.0`` (the default) or when instability
    is not detected.
    """
    if not _instability_detected(sensors, params):
        return 0.0
    yaw_rate_deg_per_s = sensors.imu.yaw_rate_degrees_per_s
    countersteer_sign = -1.0 if yaw_rate_deg_per_s > 0.0 else 1.0
    return countersteer_sign * params.instability_gain * params.steer_limit


def _effective_brake_front_m(wall: LidarSensors, params: ReactiveParams) -> float:
    """Blend the single center beam with the closest of the front cone beams.

    At ``brake_front_cone_blend == 0.0`` (the default) this returns
    ``wall.front_m`` exactly, matching current behavior.
    """
    cone_front_m = min(wall.front_m, wall.front_left_m, wall.front_right_m)
    blend = params.brake_front_cone_blend
    return wall.front_m * (1.0 - blend) + cone_front_m * blend


def _throttle_command(sensors: RobotSensors, params: ReactiveParams) -> float:
    speed_mps = sensors.odometry.speed_mps
    wall = sensors.wall_lidar

    brake_distance_m = max(
        params.brake_min_distance_m,
        speed_mps * params.brake_lead_time_s + params.brake_quadratic_coeff * speed_mps * speed_mps,
    )
    effective_front_m = _effective_brake_front_m(wall, params)
    if effective_front_m < brake_distance_m:
        deficit = _clamp((brake_distance_m - effective_front_m) / brake_distance_m, 0.0, 1.0)
        return -params.brake_gain * deficit

    corner_signal = _corner_speed_signal(sensors, params)
    damage_signal = _damage_speed_signal(sensors)
    target_speed_mps = params.max_speed_mps * (
        (1.0 - params.corner_speed_gain * corner_signal) * (1.0 - params.damage_speed_gain * damage_signal)
    )
    speed_error_mps = target_speed_mps - speed_mps
    base_throttle = _clamp(speed_error_mps * params.speed_gain, -1.0, 1.0)

    traction_signal = _traction_loss_signal(sensors, params, commanded_throttle=base_throttle)
    attitude_signal = _attitude_signal(sensors, params)
    base_throttle *= (1.0 - params.traction_loss_gain * traction_signal) * (
        1.0 - params.attitude_cutoff_gain * attitude_signal
    )

    if _instability_detected(sensors, params):
        base_throttle *= 1.0 - params.instability_gain

    return base_throttle


def _corner_speed_signal(sensors: RobotSensors, params: ReactiveParams) -> float:
    """Return an anticipated-or-actual turn severity signal in [0, 1].

    Combines the anticipated turn (camera heading error, known before the car
    turns) with the actual turn already underway (IMU yaw rate), taking
    whichever indicates the sharper need to slow down.
    """
    heading_signal = 0.0
    if sensors.camera.visible and params.corner_signal_deg > 0.0:
        heading_signal = _clamp(abs(sensors.camera.heading_error_degrees) / params.corner_signal_deg, 0.0, 1.0)
    yaw_signal = 0.0
    if params.corner_yaw_rate_deg_per_s > 0.0:
        yaw_signal = _clamp(abs(sensors.imu.yaw_rate_degrees_per_s) / params.corner_yaw_rate_deg_per_s, 0.0, 1.0)
    return max(heading_signal, yaw_signal)


def _traction_loss_signal(sensors: RobotSensors, params: ReactiveParams, *, commanded_throttle: float) -> float:
    """Return a traction-anomaly signal in [0, 1] from measured forward acceleration.

    Fires when the car commands meaningful throttle in either direction
    (``abs(commanded_throttle)`` past ``traction_loss_throttle_threshold``)
    but ``imu.forward_acceleration_mps2`` doesn't match that direction as
    expected: wheelspin or a wall-scrape while accelerating, or a brake
    that isn't actually slowing the car.
    """
    if abs(commanded_throttle) < params.traction_loss_throttle_threshold:
        return 0.0
    if params.traction_loss_expected_accel_mps2 <= 0.0:
        return 0.0
    expected_sign = 1.0 if commanded_throttle > 0.0 else -1.0
    matching_accel_mps2 = sensors.imu.forward_acceleration_mps2 * expected_sign
    deficit = params.traction_loss_expected_accel_mps2 - matching_accel_mps2
    return _clamp(deficit / params.traction_loss_expected_accel_mps2, 0.0, 1.0)


def _damage_speed_signal(sensors: RobotSensors) -> float:
    """Return accumulated non-fatal damage in [0, 1], as a throttle-reduction input.

    ``drive()`` otherwise only reads ``contact.damage`` indirectly via
    ``contact.any_contact`` (which gates recovery mode during *active*
    contact); this makes the car drive more cautiously afterward too.
    """
    return _clamp(sensors.contact.damage, 0.0, 1.0)


def _attitude_signal(sensors: RobotSensors, params: ReactiveParams) -> float:
    """Return a throttle-cut signal in [0, 1] from excessive roll/pitch.

    Zero below ``attitude_cutoff_deg``, ramping to 1.0 by twice that tilt.
    Expected to stay at 0.0 for the entire race on this flat track.
    """
    tilt_deg = max(abs(sensors.imu.roll_degrees), abs(sensors.imu.pitch_degrees))
    if tilt_deg <= params.attitude_cutoff_deg or params.attitude_cutoff_deg <= 0.0:
        return 0.0
    return _clamp((tilt_deg - params.attitude_cutoff_deg) / params.attitude_cutoff_deg, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
