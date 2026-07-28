from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def stable_id(item_id: str, row_id: object) -> str:
    digest = hashlib.sha256(f"{item_id}:{row_id}".encode()).hexdigest()
    return digest[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--annotators", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    if payload.get("locked_at_utc") in {None, "", "REPLACE_AFTER_LOCAL_SWEEP"}:
        raise ValueError("Lock the claim-validation manifest before human annotation")

    public_rows = []
    key_rows = []
    for item in payload["items"]:
        path = Path(item["raw_output_path"])
        if not path.is_absolute():
            path = args.repo_root / path
        frame = pd.read_csv(path)
        for index, row in frame.iterrows():
            row_id = row.get("row_id", index)
            annotation_id = stable_id(item["id"], row_id)
            public_rows.append(
                {
                    "annotation_id": annotation_id,
                    "question": row.get("question", row.get("prompt", "")),
                    "answer": row.get("answer", ""),
                    "trait_definition": item["trait"],
                    "trait_score_0_100": "",
                    "coherence_score_0_100": "",
                    "uncertain_0_or_1": "",
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "annotation_id": annotation_id,
                    "manifest_item_id": item["id"],
                    "source_row_id": row_id,
                    "raw_output_path": str(path),
                    "model_alias": item.get("model_alias"),
                    "phase": item.get("phase"),
                    "run_label": item.get("run_label"),
                    "selection_reason": item.get("selection_reason"),
                }
            )

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(public_rows))
    public = pd.DataFrame(public_rows).iloc[order].reset_index(drop=True)
    key = pd.DataFrame(key_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for annotator in args.annotators:
        public.to_csv(args.output_dir / f"{annotator}.blind.csv", index=False)
    key.to_csv(args.output_dir / "annotation_key.private.csv", index=False)
    metadata = {
        "locked_manifest": str(args.manifest),
        "n_items": len(public),
        "annotators": args.annotators,
        "seed": args.seed,
        "blinding": "Model, run, phase, and selection reason are stored only in annotation_key.private.csv.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
