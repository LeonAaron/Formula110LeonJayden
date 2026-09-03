"""Step 3 validation: run the Step 2 search's best candidate on held-out seeds.

Accept only if 0 damage in all validation races AND mean distance strictly
exceeds the ~442-445m reactive.py baseline (measured at the same 30s rounds).
"""

from __future__ import annotations

from dataclasses import replace

from controllers.reactive2 import DEFAULT_PARAMS, ReactiveParams, drive
from racing import RobotCommand, RobotSensors, run_headless_head_to_head


def _passive_controller(sensors: RobotSensors) -> RobotCommand:
    """Coast in place so the challenger's solo run is barely disturbed."""
    return RobotCommand(throttle=0.0, steer=0.0)


def make_controller(params: ReactiveParams):
    """Build a callable controller bound to one parameter set."""

    def control(sensors: RobotSensors) -> RobotCommand:
        return drive(sensors, params)

    return control

CANDIDATE = replace(
    DEFAULT_PARAMS,
    center_offset_gain=0.16329481913850655,
    heading_error_gain=0.047392860963696146,
    lookahead_near_gain=0.05156791694867777,
    lookahead_far_gain=0.042957106946593,
    wall_avoid_margin_m=1.125402374107058,
    wall_avoid_gain=0.2925912086177571,
    emergency_front_m=2.697433783832892,
    emergency_steer=0.7242665583427179,
    recovery_steer=1.8927912105935465,
    recovery_throttle=-0.22450777801650354,
    steer_limit=0.9188703735967165,
    turn_sharpness_deg=5.0,
    apex_bias_max_m=0.25329318957554986,
    max_speed_mps=17.014201536345517,
    speed_gain=0.2537829451258241,
    corner_speed_gain=0.02486010628023755,
    corner_signal_deg=52.45465009500402,
    corner_yaw_rate_deg_per_s=60.63871092991616,
    brake_lead_time_s=0.24588205755469877,
    brake_min_distance_m=0.35809461475892485,
    brake_gain=1.3949832375436668,
    brake_quadratic_coeff=0.0005487103375623236,
    traction_loss_throttle_threshold=0.30868865909162585,
    traction_loss_expected_accel_mps2=1.1682154924097303,
    traction_loss_gain=0.018785154251528872,
    diagonal_avoid_margin_m=1.5,
    diagonal_avoid_gain=0.005263270771884466,
    curvature_gain=0.009856836169973755,
    damage_speed_gain=0.0,
    attitude_cutoff_deg=5.5193163263337235,
    attitude_cutoff_gain=0.012312615071995315,
    speed_scaled_margin_lead_s=0.03861883103383921,
    instability_yaw_rate_deg_per_s=306.47111883260266,
    instability_lateral_accel_mps2=5.469000995418077,
    instability_gain=0.01339968479208049,
    brake_front_cone_blend=0.00939171038983707,
)

VALIDATION_SEEDS = (42, 2027, 8675, 31415, 777001)
ROUND_SECONDS = 30.0
RACES_PER_SEED = 5


def main() -> None:
    all_distances: list[float] = []
    all_damages: list[float] = []
    for seed in VALIDATION_SEEDS:
        result = run_headless_head_to_head(
            challenger_controller=make_controller(CANDIDATE),
            incumbent_controller=_passive_controller,
            race_count=RACES_PER_SEED,
            round_seconds=ROUND_SECONDS,
            random_seed=seed,
        )
        distances = [race.challenger.distances_m[0] for race in result.races]
        damages = [race.challenger.damages[0] for race in result.races]
        all_distances.extend(distances)
        all_damages.extend(damages)
        dist_str = " ".join(f"{d:6.1f}" for d in distances)
        dmg_str = " ".join(f"{d:.2f}" for d in damages)
        print(f"seed {seed:>7}: distances=[{dist_str}] damages=[{dmg_str}]")

    mean_distance = sum(all_distances) / len(all_distances)
    max_damage = max(all_damages)
    any_eliminated = any(d >= 1.0 for d in all_damages)
    any_damage = any(d > 0.0 for d in all_damages)

    print("-" * 72)
    print(f"races: {len(all_distances)}  mean distance: {mean_distance:.1f}m  max damage: {max_damage:.2f}")
    print(f"any elimination: {any_eliminated}  any non-zero damage: {any_damage}")
    print("-" * 72)

    baseline_low, baseline_high = 442.0, 445.0
    if any_damage:
        print(f"REJECT: non-zero damage occurred in at least one of the {len(all_distances)} validation races.")
    elif mean_distance <= baseline_high:
        print(
            f"REJECT: mean validation distance {mean_distance:.1f}m does not strictly exceed "
            f"the {baseline_low:.0f}-{baseline_high:.0f}m baseline."
        )
    else:
        print(
            f"ACCEPT: 0 damage across all {len(all_distances)} validation races, "
            f"mean distance {mean_distance:.1f}m exceeds the {baseline_low:.0f}-{baseline_high:.0f}m baseline."
        )


if __name__ == "__main__":
    main()
