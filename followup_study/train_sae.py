from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.decoder.weight.div_(norms)


def load_activations(paths: list[Path]) -> torch.Tensor:
    tensors = []
    for path in paths:
        payload = torch.load(path, weights_only=False)
        tensors.append(payload["activations"].float())
    return torch.cat(tensors, dim=0)


def train_sae(
    input_paths: list[Path],
    output_dir: Path,
    dictionary_multiplier: int,
    l1_coef: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    activations = load_activations(input_paths)
    input_dim = activations.shape[1]
    hidden_dim = input_dim * dictionary_multiplier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_sparse = 0.0
        total_examples = 0

        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, features = model(batch)
            recon_loss = torch.mean((recon - batch) ** 2)
            sparse_loss = features.abs().mean()
            loss = recon_loss + l1_coef * sparse_loss
            loss.backward()
            optimizer.step()
            model.normalize_decoder()

            batch_size_actual = batch.shape[0]
            total_examples += batch_size_actual
            total_loss += loss.item() * batch_size_actual
            total_recon += recon_loss.item() * batch_size_actual
            total_sparse += sparse_loss.item() * batch_size_actual

        metrics = {
            "epoch": epoch + 1,
            "loss": total_loss / total_examples,
            "reconstruction_mse": total_recon / total_examples,
            "mean_feature_l1": total_sparse / total_examples,
        }
        history.append(metrics)
        print(json.dumps(metrics))

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "dictionary_multiplier": dictionary_multiplier,
        "state_dict": model.cpu().state_dict(),
        "history": history,
    }
    torch.save(checkpoint, output_dir / "sae.pt")
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dictionary_multiplier", type=int, default=8)
    parser.add_argument("--l1_coef", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_sae(
        input_paths=args.input_paths,
        output_dir=args.output_dir,
        dictionary_multiplier=args.dictionary_multiplier,
        l1_coef=args.l1_coef,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
