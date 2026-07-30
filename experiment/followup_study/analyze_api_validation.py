from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = {
    "minimum_provider_groups": 3,
    "maximum_parse_failure_rate": 0.05,
    "minimum_canonical_direction_agreement": 1.0,
    "minimum_variant_direction_agreement": 1.0,
    "minimum_canonical_ci_pass_fraction": 2.0 / 3.0,
    "bootstrap_samples": 2000,
    "bootstrap_seed": 2026,
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if sha256_path(path) != expected_sha256:
        raise ValueError(f"{label} changed after API validation was locked: {path}")


def cluster_column(frame: pd.DataFrame) -> str:
    return next(
        (
            column
            for column in ["question_id", "question", "prompt", "row_id"]
            if column in frame.columns and frame[column].notna().any()
        ),
        frame.columns[0],
    )


def bootstrap_means(
    frame: pd.DataFrame,
    score_column: str,
    *,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cluster = cluster_column(frame)
    valid = frame.dropna(subset=[score_column, cluster]).copy()
    valid[score_column] = pd.to_numeric(valid[score_column], errors="coerce")
    valid = valid.dropna(subset=[score_column])
    clusters = valid[cluster].astype(str).unique()
    if not len(clusters):
        return np.array([], dtype=float)
    grouped = {
        key: valid.loc[valid[cluster].astype(str) == key, score_column].to_numpy(
            dtype=float
        )
        for key in clusters
    }
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        estimates[index] = np.concatenate([grouped[key] for key in selected]).mean()
    return estimates


def parse_rate(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").isna().mean())


def load_item_scores(
    item: dict,
    config: dict,
    *,
    trait: str,
) -> tuple[pd.DataFrame, str, str]:
    path = Path(item["merged_path"]).parent / "judges" / f'{config["name"]}.csv'
    locked_output = item.get("judge_outputs", {}).get(config["name"])
    if locked_output:
        verify_file(
            Path(locked_output["path"]),
            locked_output["sha256"],
            f'{item["id"]}/{config["name"]}',
        )
    frame = pd.read_csv(path)
    trait_column = f'{config["name"]}__{trait}'
    coherence_column = f'{config["name"]}__coherence'
    missing = {trait_column, coherence_column} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame[trait_column] = pd.to_numeric(frame[trait_column], errors="coerce")
    frame[coherence_column] = pd.to_numeric(frame[coherence_column], errors="coerce")
    return frame, trait_column, coherence_column


def direction_multiplier(direction: str) -> float:
    if direction == "increase":
        return 1.0
    if direction == "decrease":
        return -1.0
    raise ValueError(f"Unsupported expected_direction: {direction}")


def evaluate_claim(
    claim: dict,
    *,
    items: dict[str, dict],
    configs: list[dict],
    thresholds: dict,
) -> dict:
    reference = items[claim["reference_item_id"]]
    treatment = items[claim["treatment_item_id"]]
    trait = claim.get("trait", treatment["trait"])
    if trait != reference["trait"] or trait != treatment["trait"]:
        raise ValueError(f'{claim["id"]}: trait does not match both manifest items')

    multiplier = direction_multiplier(claim.get("expected_direction", "increase"))
    minimum_effect = float(claim.get("minimum_effect", 0.0))
    samples = int(thresholds["bootstrap_samples"])
    seed = int(thresholds["bootstrap_seed"])
    evaluations = []

    for config_index, config in enumerate(configs):
        reference_frame, reference_score, reference_coherence = load_item_scores(
            reference,
            config,
            trait=trait,
        )
        treatment_frame, treatment_score, treatment_coherence = load_item_scores(
            treatment,
            config,
            trait=trait,
        )
        reference_mean = float(reference_frame[reference_score].mean())
        treatment_mean = float(treatment_frame[treatment_score].mean())
        raw_delta = treatment_mean - reference_mean
        directional_delta = multiplier * raw_delta

        rng = np.random.default_rng(seed + config_index)
        reference_bootstrap = bootstrap_means(
            reference_frame,
            reference_score,
            samples=samples,
            rng=rng,
        )
        treatment_bootstrap = bootstrap_means(
            treatment_frame,
            treatment_score,
            samples=samples,
            rng=rng,
        )
        if len(reference_bootstrap) and len(treatment_bootstrap):
            directional_bootstrap = multiplier * (
                treatment_bootstrap - reference_bootstrap
            )
            ci_lower = float(np.quantile(directional_bootstrap, 0.025))
            ci_upper = float(np.quantile(directional_bootstrap, 0.975))
        else:
            ci_lower = None
            ci_upper = None

        parse_failure_rate = max(
            parse_rate(reference_frame, reference_score),
            parse_rate(reference_frame, reference_coherence),
            parse_rate(treatment_frame, treatment_score),
            parse_rate(treatment_frame, treatment_coherence),
        )
        evaluations.append(
            {
                "judge_name": config["name"],
                "base_judge_name": config["base_judge_name"],
                "provider_group": config["provider_group"],
                "prompt_variant": config["prompt_variant"],
                "reference_mean": reference_mean,
                "treatment_mean": treatment_mean,
                "raw_delta": raw_delta,
                "directional_delta": directional_delta,
                "direction_pass": directional_delta > minimum_effect,
                "directional_ci_lower": ci_lower,
                "directional_ci_upper": ci_upper,
                "ci_pass": ci_lower is not None and ci_lower > minimum_effect,
                "parse_failure_rate": parse_failure_rate,
            }
        )

    canonical = [
        evaluation
        for evaluation in evaluations
        if evaluation["prompt_variant"] == "canonical"
    ]
    provider_groups = sorted({row["provider_group"] for row in canonical})
    canonical_direction = float(np.mean([row["direction_pass"] for row in canonical]))
    variant_direction = float(np.mean([row["direction_pass"] for row in evaluations]))
    canonical_ci = float(np.mean([row["ci_pass"] for row in canonical]))
    maximum_parse_failure = max(row["parse_failure_rate"] for row in evaluations)
    checks = {
        "provider_diversity": (
            len(provider_groups) >= int(thresholds["minimum_provider_groups"])
        ),
        "parse_success": (
            maximum_parse_failure
            <= float(thresholds["maximum_parse_failure_rate"])
        ),
        "canonical_direction": (
            canonical_direction
            >= float(thresholds["minimum_canonical_direction_agreement"])
        ),
        "variant_direction": (
            variant_direction
            >= float(thresholds["minimum_variant_direction_agreement"])
        ),
        "canonical_ci": (
            canonical_ci
            >= float(thresholds["minimum_canonical_ci_pass_fraction"])
        ),
    }
    return {
        **claim,
        "trait": trait,
        "provider_groups": provider_groups,
        "canonical_direction_agreement": canonical_direction,
        "variant_direction_agreement": variant_direction,
        "canonical_ci_pass_fraction": canonical_ci,
        "maximum_parse_failure_rate": maximum_parse_failure,
        "checks": checks,
        "accepted": all(checks.values()),
        "evaluations": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    claims = audit.get("claims", [])
    if not claims:
        raise ValueError(
            "The locked claim-validation manifest must define at least one claim"
        )
    items = {item["id"]: item for item in audit["items"]}
    manifest_path = Path(audit["source_manifest"])
    verify_file(
        manifest_path,
        audit["source_manifest_sha256"],
        "claim-validation manifest",
    )
    calibration = audit["api_calibration"]
    verify_file(
        Path(calibration["summary_path"]),
        calibration["summary_sha256"],
        "API judge calibration report",
    )
    configs = []
    for record in audit["resolved_judge_configs"]:
        path = Path(record["path"])
        verify_file(path, record["sha256"], "resolved judge config")
        configs.append(json.loads(path.read_text()))
    thresholds = {
        **DEFAULT_THRESHOLDS,
        **audit.get("automatic_validation", {}),
    }
    results = [
        evaluate_claim(
            claim,
            items=items,
            configs=configs,
            thresholds=thresholds,
        )
        for claim in claims
    ]
    payload = {
        "source_audit": str(args.audit),
        "locked_at_utc": audit["locked_at_utc"],
        "api_calibration": calibration,
        "thresholds": thresholds,
        "n_claims": len(results),
        "accepted_claims": sum(result["accepted"] for result in results),
        "all_claims_accepted": all(result["accepted"] for result in results),
        "claims": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
