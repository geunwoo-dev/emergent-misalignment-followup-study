from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from train_sae import SparseAutoencoder


def load_checkpoint(path: Path) -> SparseAutoencoder:
    payload = torch.load(path, weights_only=False)
    model = SparseAutoencoder(
        input_dim=payload["input_dim"],
        hidden_dim=payload["hidden_dim"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_activations(path: Path) -> torch.Tensor:
    payload = torch.load(path, weights_only=False)
    return payload["activations"].float()


def encode_mean(model: SparseAutoencoder, activations: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    outputs = []
    for start in range(0, len(activations), batch_size):
        batch = activations[start : start + batch_size]
        with torch.no_grad():
            outputs.append(model.encode(batch))
    return torch.cat(outputs, dim=0)


def score_features(
    checkpoint_path: Path,
    baseline_path: Path,
    target_path: Path,
    output_path: Path,
    top_k: int,
) -> None:
    model = load_checkpoint(checkpoint_path)
    baseline = encode_mean(model, load_activations(baseline_path))
    target = encode_mean(model, load_activations(target_path))

    baseline_mean = baseline.mean(dim=0)
    target_mean = target.mean(dim=0)
    baseline_density = (baseline > 0).float().mean(dim=0)
    target_density = (target > 0).float().mean(dim=0)
    diff = target_mean - baseline_mean

    ranked = torch.argsort(diff.abs(), descending=True)[:top_k].tolist()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_idx",
                "baseline_mean",
                "target_mean",
                "mean_shift",
                "baseline_density",
                "target_density",
            ],
        )
        writer.writeheader()
        for idx in ranked:
            writer.writerow(
                {
                    "feature_idx": idx,
                    "baseline_mean": baseline_mean[idx].item(),
                    "target_mean": target_mean[idx].item(),
                    "mean_shift": diff[idx].item(),
                    "baseline_density": baseline_density[idx].item(),
                    "target_density": target_density[idx].item(),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--baseline_path", type=Path, required=True)
    parser.add_argument("--target_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()
    score_features(
        checkpoint_path=args.checkpoint_path,
        baseline_path=args.baseline_path,
        target_path=args.target_path,
        output_path=args.output_path,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
