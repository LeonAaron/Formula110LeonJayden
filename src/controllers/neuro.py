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

# Evolved by `uv run python scripts/train_neuroevolution.py` (see LAB_NOTEBOOK.md, Entry 2).
# --population 12 --generations 10 --seeds 13 --round-seconds 20; training fitness 259.6m.
BEST_GENOME: Genome = (
    1.2256752303254181,
    0.4628161145250449,
    0.1425871750954248,
    -0.10100581317777685,
    -1.938209138025414,
    -0.6049534287340492,
    -2.6401829927026923,
    -0.9653338895974208,
    0.8248108034588505,
    0.9839013098831496,
    1.4880886343823845,
    0.16713493146089084,
    -0.1812944227898304,
    -0.050474996184565676,
    -1.7803360088537856,
    0.5923644037363553,
    1.986234472363153,
    1.0510224815813933,
    0.5191977933582389,
    0.6931140276082292,
    1.107773661165002,
    -0.4582951153204807,
    0.6850150307517677,
    0.8899697374604635,
    -0.5851398314295806,
    1.2253443276911633,
    2.574691566754455,
    -0.10341060253341783,
    -1.0251904200096351,
    -0.46714292457851503,
    0.13940696382308487,
    -0.31982523861465656,
    -1.1926010463927694,
    -0.05643747590232523,
    0.5756026815819086,
    -0.4807063507068703,
    0.4596281262964136,
    -1.324758538304855,
    -0.45768738112848695,
    -0.9756270338158031,
    0.7007351080576499,
    -0.06945152808624216,
    -0.31916639233094923,
    0.7132203573005674,
    0.6250576324082802,
    -0.7601237218153487,
    -0.16763580144216353,
    -0.39919563043906686,
    -0.019445606251502462,
    -0.3239166975989313,
    -1.6571433771270112,
    0.41745043055352676,
    1.075191415163842,
    -0.4918532573599756,
    -1.1728355108324973,
    0.27406764015502755,
    -0.27502747845134234,
    -0.7071630967608252,
    -0.26603466429621914,
    -0.39990779080132177,
    -0.14563317269459775,
    1.7069258717671676,
    -1.1343685326783364,
    -0.4348559209411547,
    0.0018981233299927014,
    0.4201269638344519,
    -1.2851317592489255,
    -0.7039151215599642,
    -0.84479031348355,
    -0.019949625773712787,
    -0.8556383584621085,
    0.8657747794211521,
    0.9808817205882787,
    0.2726626190896228,
    1.0960591736920695,
    -0.40123288245307565,
    -0.9730844625357817,
    -0.9351725055888185,
    -1.774828346441278,
    -0.6211267434703522,
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
