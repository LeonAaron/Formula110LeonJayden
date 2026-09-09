"""Learned controller: a small torch MLP trained by behavior cloning + CMA-ES.

Weights are trained offline by ``scripts/generate_expert_demonstrations.py``
(behavior-cloning data from the reactive controller), ``scripts/train_behavior_cloning.py``
(supervised warm start), and ``scripts/train_learned_controller.py`` (dense-reward
CMA-ES fine-tune). ``build_inputs`` and ``PolicyNet`` are shared with the training
scripts so evaluation and training run identical code. See LAB_NOTEBOOK.md.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from racing import RobotCommand, RobotSensors

RACING_NAME = "Learned Pilot"
RACING_COLOR = "#FF9F1C"

INPUT_SIZE = 13
HIDDEN_SIZE = 8
OUTPUT_SIZE = 2

WALL_LIDAR_CAP_M = 15.0
CENTER_OFFSET_CAP_M = 3.3
LOOKAHEAD_OFFSET_CAP_M = 3.3
# Matches the shipped src/controllers/learned_policy.pt (Entry 14, trained at
# HIDDEN_SIZE=8, SPEED_CAP_MPS=20.0). Entry 17 tried HIDDEN_SIZE=20 and
# SPEED_CAP_MPS=40.0 (to remove speed-input saturation for a much-higher-speed
# expert) - fixed the eliminations from Entry 16 but still landed below the
# shipped baseline (518.0m vs 568.0m), so not adopted; see LAB_NOTEBOOK.md.
# IMPORTANT: HIDDEN_SIZE must match whatever learned_policy.pt was trained
# with, or Controller.__init__'s load_state_dict will fail on a shape mismatch.
SPEED_CAP_MPS = 20.0

WEIGHTS_PATH = Path(__file__).with_name("learned_policy.pt")


class PolicyNet(nn.Module):
    """Single-hidden-layer tanh MLP mapping normalized sensors to (throttle, steer).

    Kept small (~140 parameters) so the CMA-ES fine-tune stage has a tractable
    search dimension within a short training budget.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_inputs(sensors: RobotSensors) -> tuple[float, ...]:
    """Map raw sensors to a normalized 13-value feature vector for the network."""
    wall = sensors.wall_lidar
    beams = tuple(min(distance, WALL_LIDAR_CAP_M) / WALL_LIDAR_CAP_M for distance in wall.distances_m)
    heading = _clamp(sensors.camera.heading_error_degrees / 180.0, -1.0, 1.0)
    center = _clamp(sensors.camera.center_offset_m / CENTER_OFFSET_CAP_M, -1.0, 1.0)
    speed = _clamp(sensors.odometry.speed_mps / SPEED_CAP_MPS, -1.0, 1.0)
    offsets = sensors.camera.lookahead_offsets_m
    lookahead = tuple(_clamp(offset / LOOKAHEAD_OFFSET_CAP_M, -1.0, 1.0) for offset in offsets[:3])
    while len(lookahead) < 3:
        lookahead = (*lookahead, 0.0)
    return (*beams, heading, center, speed, *lookahead)


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Flatten every parameter tensor into one 1-D vector, for a gradient-free search."""
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Load a flat parameter vector (from `get_flat_params`) back into a model, in place."""
    cursor = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.data.copy_(flat[cursor : cursor + count].reshape(parameter.shape))
        cursor += count


def drive(sensors: RobotSensors, model: PolicyNet) -> RobotCommand:
    """Pure sensor-to-command mapping for a given model, reused by training."""
    inputs = torch.tensor(build_inputs(sensors), dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        throttle, steer = model(inputs).squeeze(0).tolist()
    return RobotCommand(throttle=throttle, steer=steer)


class Controller:
    """Stateful runtime controller: loads trained weights once, then drives."""

    def __init__(self, weights_path: Path = WEIGHTS_PATH) -> None:
        self.model = PolicyNet()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to("cpu")
        self.model.eval()

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        return drive(sensors, self.model)


def create_controller() -> Controller:
    return Controller()


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
