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

# Evolved by `uv run python scripts/train_neuroevolution.py` (see LAB_NOTEBOOK.md, Entry 7).
# --population 40 --generations 15 --elite 3 --seeds 13 55 7 89 233 --round-seconds 20; training fitness 249.8m.
# Held-out validation (42, 110, 271, 997, 2027; 5 races/seed): 25/25 survived, 25/25 laps,
# avg scored distance 371.1m, max speed 28.1 m/s.
BEST_GENOME: Genome = (
    -0.11482510646981381,
    -1.4701012832329245,
    -0.5495292231573892,
    -0.390459365195705,
    0.336047349984261,
    1.3639874812566033,
    1.7595137087132426,
    0.6613764947234899,
    -0.06785393068386075,
    -0.3816642680838079,
    0.18782082225717525,
    -1.1493642454189852,
    -0.14330882164201297,
    -0.005292564576633685,
    -0.5716904974063464,
    -1.8227700486010905,
    -0.8597615175671809,
    0.015416443514673113,
    0.3943263310531005,
    0.5483229805696904,
    0.9449534986056581,
    -0.1454802834984803,
    0.13029718861211093,
    1.3604321878571075,
    -1.863428717854719,
    -1.8515340992299787,
    1.6324325083846063,
    0.6889857046967603,
    -1.661711320367092,
    -0.9843201805743829,
    0.4156157324558321,
    1.2047684058693044,
    -0.30927217108541893,
    0.9262760002972616,
    0.46711928790454726,
    0.5371454619244611,
    1.5443414255819237,
    -1.624122903499653,
    0.3398526837652208,
    0.05200061332440614,
    0.19889892889180605,
    -1.3425337825479102,
    0.48428536536338607,
    -0.6746940502061444,
    0.17592123480606978,
    -1.4976191028594443,
    -0.5144861597180935,
    0.13680210105791407,
    0.017676453989006857,
    -0.6739517705999933,
    1.544289163636964,
    1.0609396352329667,
    -1.188983241174976,
    0.4140410484663306,
    1.0479737649217773,
    -0.5980306035237972,
    0.9122456933974703,
    -0.7166839156447378,
    -1.3224767279802883,
    0.20148122793815754,
    1.9566826681049156,
    0.3982060494957199,
    1.6987292931486735,
    1.321982019913575,
    1.1292943626441474,
    -1.4493431551140565,
    1.1506424185836892,
    0.7367428015643739,
    0.4158041810691349,
    0.17726588302115479,
    0.08382987047696155,
    -1.37890377664573,
    1.2714358056054595,
    3.3035155488753376,
    1.1484590372277592,
    1.6564478417163833,
    0.6057695492217162,
    0.42718014734120485,
    2.221337521778695,
    -0.18250929637053442,
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
