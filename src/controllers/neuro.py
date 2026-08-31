"""Neuroevolution controller: a small MLP mapping normalized sensors to throttle/steer.

Weights are evolved offline by ``scripts/train_neuroevolution.py`` using a
simple evolution strategy and then hardcoded as ``BEST_GENOME``. ``build_inputs``
and ``forward`` are shared with the training script so evaluation and training
run identical code.
"""

from __future__ import annotations

from math import tanh

from racing import RobotCommand, RobotSensors

RACING_NAME = "Neuro Pilot"
RACING_COLOR = "#B34CFF"

INPUT_SIZE = 10
HIDDEN_SIZE = 6
OUTPUT_SIZE = 2
GENOME_SIZE = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE

WALL_LIDAR_CAP_M = 15.0
CENTER_OFFSET_CAP_M = 3.3
SPEED_CAP_MPS = 15.0

Genome = tuple[float, ...]

# Evolved by `uv run python scripts/train_neuroevolution.py` (see LAB_NOTEBOOK.md, Entry 5).
# --population 20 --generations 15 --seeds 13 55 7 89 233 --round-seconds 20; training fitness 235.8m.
# Held-out validation (42, 110, 271, 997, 2027; 5 races/seed): 24/25 survived, 25/25 laps,
# avg scored distance 356.0m, max speed 28.9 m/s.
BEST_GENOME: Genome = (
    0.763187669484052,
    -0.5971032152787542,
    0.34646158608849453,
    0.4048496055835485,
    -0.428376351805434,
    -0.2958782759638022,
    -0.8254099323994528,
    0.7792979346032775,
    -0.6682149265442103,
    0.7091776035102759,
    1.0730626333870803,
    0.4949936198623556,
    1.0984399696329468,
    -1.3215160586984835,
    0.5928652388465152,
    -0.0069021455626616934,
    -0.8008027392459672,
    0.9356714034773636,
    0.2235846315638727,
    -0.9411940148860757,
    0.8296940494409376,
    -1.8015684479171459,
    -0.9468414337952774,
    -1.2011890057481316,
    -2.0841324736288467,
    0.10696477290625829,
    0.5852522464564646,
    0.38935249138136163,
    -1.5961834030334832,
    -0.5421318371502277,
    -0.4856969771412015,
    -1.2445796272986178,
    -0.07457888653865938,
    0.4292260857069271,
    0.6011541145052928,
    0.947072313464087,
    -0.19080520094443543,
    0.8262670228763986,
    0.4640019163406234,
    1.317808291610001,
    -0.2280818178378578,
    0.28053661111688366,
    -0.7265588693117287,
    -0.3897228152147654,
    -1.4337455201004428,
    -0.8002150638063424,
    0.13612182044304455,
    -0.15331796022662697,
    -0.11832607748129664,
    -1.6718375670537047,
    0.8073104192890688,
    0.6066677693900391,
    0.2734279937207301,
    0.35139505835657436,
    0.2025540974728232,
    2.232647253797279,
    0.1930816873085908,
    1.2162198353731495,
    -0.3988126241674169,
    1.0795869686630928,
    -3.592928839897215,
    1.5732815881196545,
    -1.579402343255498,
    0.6162642821975587,
    -1.18156710165456,
    -0.10674669147465943,
    -2.0955368022282754,
    -0.6687586640039257,
    -0.25752686188545676,
    2.224469335531837,
    0.010508944886482158,
    -0.03920709778262656,
    -0.4090025152651273,
    -0.6030208446523952,
    -0.6716908182369772,
    -0.9482813791855638,
    -0.48393482348538047,
    0.08277863407117714,
    -2.5824872157883587,
    -0.7906890457669933,
)


def build_inputs(sensors: RobotSensors) -> tuple[float, ...]:
    """Map raw sensors to a small, normalized feature vector for the network."""
    wall = sensors.wall_lidar
    beams = tuple(min(distance, WALL_LIDAR_CAP_M) / WALL_LIDAR_CAP_M for distance in wall.distances_m)
    heading = _clamp(sensors.camera.heading_error_degrees / 180.0, -1.0, 1.0)
    center = _clamp(sensors.camera.center_offset_m / CENTER_OFFSET_CAP_M, -1.0, 1.0)
    speed = _clamp(sensors.odometry.speed_mps / SPEED_CAP_MPS, -1.0, 1.0)
    return (*beams, heading, center, speed)


def forward(inputs: tuple[float, ...], genome: Genome) -> tuple[float, float]:
    """Run the two-layer tanh MLP; returns ``(throttle, steer)``."""
    if len(inputs) != INPUT_SIZE:
        raise ValueError(f"expected {INPUT_SIZE} inputs, got {len(inputs)}")
    if len(genome) != GENOME_SIZE:
        raise ValueError(f"expected genome of length {GENOME_SIZE}, got {len(genome)}")

    cursor = 0
    hidden: list[float] = []
    for _ in range(HIDDEN_SIZE):
        weights = genome[cursor : cursor + INPUT_SIZE]
        cursor += INPUT_SIZE
        bias = genome[cursor]
        cursor += 1
        activation = sum(weight * value for weight, value in zip(weights, inputs, strict=True)) + bias
        hidden.append(tanh(activation))

    outputs: list[float] = []
    for _ in range(OUTPUT_SIZE):
        weights = genome[cursor : cursor + HIDDEN_SIZE]
        cursor += HIDDEN_SIZE
        bias = genome[cursor]
        cursor += 1
        activation = sum(weight * value for weight, value in zip(weights, hidden, strict=True)) + bias
        outputs.append(tanh(activation))

    throttle, steer = outputs
    return throttle, steer


def drive(sensors: RobotSensors, genome: Genome) -> RobotCommand:
    """Pure sensor-to-command mapping for a given genome, reused by training."""
    throttle, steer = forward(build_inputs(sensors), genome)
    return RobotCommand(throttle=throttle, steer=steer)


def control(sensors: RobotSensors) -> RobotCommand:
    """Entry point used by the runtime: drive with the evolved genome."""
    return drive(sensors, BEST_GENOME)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
