from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def build_intervention_vector(
    sae_checkpoint: Path,
    feature_csv: Path,
    layer: int,
    top_k: int,
    output_path: Path,
    selection: str,
) -> None:
    payload = torch.load(sae_checkpoint, weights_only=False)
    decoder_weight = payload["state_dict"]["decoder.weight"].float()

    features = pd.read_csv(feature_csv)
    if selection == "positive":
        features = features.sort_values("mean_shift", ascending=False).head(top_k)
    elif selection == "negative":
        features = features.sort_values("mean_shift", ascending=True).head(top_k)
    else:
        features = features.assign(abs_shift=features["mean_shift"].abs()).sort_values("abs_shift", ascending=False).head(top_k)

    vector = torch.zeros(layer + 1, decoder_weight.shape[0], dtype=torch.float32)
    dense_direction = torch.zeros(decoder_weight.shape[0], dtype=torch.float32)
    feature_metadata = []
    for _, row in features.iterrows():
        feature_idx = int(row["feature_idx"])
        weight = float(row["mean_shift"])
        dense_direction -= weight * decoder_weight[:, feature_idx]
        feature_metadata.append(
            {
                "feature_idx": feature_idx,
                "mean_shift": weight,
            }
        )
    vector[layer] = dense_direction

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector, output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "layer": layer,
                "top_k": top_k,
                "selection": selection,
                "features": feature_metadata,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_checkpoint", type=Path, required=True)
    parser.add_argument("--feature_csv", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--selection", choices=["positive", "negative", "absolute"], default="positive")
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    build_intervention_vector(
        sae_checkpoint=args.sae_checkpoint,
        feature_csv=args.feature_csv,
        layer=args.layer,
        top_k=args.top_k,
        output_path=args.output_path,
        selection=args.selection,
    )


if __name__ == "__main__":
    main()
