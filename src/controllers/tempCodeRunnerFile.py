"""Self-driving race car controlled by two small perceptrons."""

from math import tanh

from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Smol Brain"
RACING_COLOR: str = "#ffffff"

SPEED_NORMALIZATION_MIN_MPS = -50.0
SPEED_NORMALIZATION_MAX_MPS = 50.0
HEADING_NORMALIZATION_MIN_DEGREES = -180.0
HEADING_NORMALIZATION_MAX_DEGREES = 180.0

THROTTLE_W = -0.5
THROTTLE_B = 0.5

STEER_W = 1.0
STEER_B = 0.0


def control(sensors: RobotSensors) -> RobotCommand:
    """Compute throttle and steering with one tanh neuron per output."""
    speed_mps: float = sensors.odometry.speed_mps
    heading_degrees: float = sensors.camera.heading_error_degrees

    speed_n = clamp_and_normalize(
        speed_mps,
        SPEED_NORMALIZATION_MIN_MPS,
        SPEED_NORMALIZATION_MAX_MPS,
    )
    heading_n = clamp_and_normalize(
        heading_degrees,
        HEADING_NORMALIZATION_MIN_DEGREES,
        HEADING_NORMALIZATION_MAX_DEGREES,
    )

    throttle_input = THROTTLE_W * speed_n + THROTTLE_B
    throttle: float = tanh(throttle_input)

    steer_input = STEER_W * heading_n + STEER_B
    steer: float = tanh(steer_input)

    return RobotCommand(throttle, steer)


def clamp_and_normalize(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value to [minimum, maximum] and normalize it to [-1.0, 1.0]."""
    if minimum >= maximum:
        raise ValueError("minimum must be less than maximum")

    clamped_value: float = max(min(value, maximum), minimum)
    midpoint: float = (maximum + minimum) / 2.0
    half_range: float = (maximum - minimum) / 2.0
    return (clamped_value - midpoint) / half_range
