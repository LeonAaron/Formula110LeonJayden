"""Behavior-cloning warm start: supervised training of PolicyNet on reactive demos.

Trains the learned controller's MLP to imitate `controllers.reactive`'s
(sensors -> command) mapping via MSE regression on the dataset from
`scripts/generate_expert_demonstrations.py`. This gives the CMA-ES fine-tune
step (`scripts/train_learned_controller.py`) a known-good starting point in
weight space instead of random init (the root cause of Entry 2's degenerate
"idle" and "reckless" local optima). See LAB_NOTEBOOK.md.

Usage:
    uv run python scripts/train_behavior_cloning.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim

from controllers.learned import PolicyNet

DEFAULT_DATASET = Path("artifacts/expert_demonstrations.pt")
DEFAULT_OUTPUT = Path("artifacts/bc_policy.pt")


def train_behavior_cloning(
    *, dataset_path: Path, epochs: int, batch_size: int, learning_rate: float, rng_seed: int
) -> PolicyNet:
    torch.manual_seed(rng_seed)
    dataset = torch.load(dataset_path, weights_only=True)
    features, actions = dataset["features"], dataset["actions"]

    model = PolicyNet()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    sample_count = features.shape[0]
    for epoch in range(epochs):
        permutation = torch.randperm(sample_count)
        epoch_loss = 0.0
        for start in range(0, sample_count, batch_size):
            batch_indices = permutation[start : start + batch_size]
            predicted = model(features[batch_indices])
            loss = loss_fn(predicted, actions[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_indices)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}: mse={epoch_loss / sample_count:.5f}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()

    model = train_behavior_cloning(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        rng_seed=args.rng_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"saved behavior-cloned policy to {args.output}")


if __name__ == "__main__":
    main()
