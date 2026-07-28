from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    regularization: float,
    iterations: int = 2500,
    learning_rate: float = 0.05,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.zeros(design.shape[1], dtype=float)
    weights = sample_weight / sample_weight.mean()
    for _ in range(iterations):
        probabilities = sigmoid(design @ coefficients)
        gradient = design.T @ ((probabilities - y) * weights) / len(y)
        penalty = regularization * np.r_[0.0, coefficients[1:]]
        coefficients -= learning_rate * (gradient + penalty)
    return coefficients


def predict_logistic(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return sigmoid(np.column_stack([np.ones(len(x)), x]) @ coefficients)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train - mean) / std, (test - mean) / std


def auroc(y: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(y: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
    return float(precision[sorted_y == 1].sum() / positives)


def choose_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    best_threshold = 0.5
    best_youden = -np.inf
    for threshold in candidates:
        predictions = scores >= threshold
        positives = y == 1
        negatives = ~positives
        sensitivity = float(predictions[positives].mean()) if positives.any() else 0.0
        specificity = float((~predictions[negatives]).mean()) if negatives.any() else 0.0
        youden = sensitivity + specificity - 1.0
        if youden > best_youden:
            best_youden = youden
            best_threshold = float(threshold)
    return best_threshold


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = scores >= threshold
    positives = y == 1
    negatives = ~positives
    sensitivity = float(predictions[positives].mean()) if positives.any() else None
    specificity = float((~predictions[negatives]).mean()) if negatives.any() else None
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else None,
        "auroc": auroc(y, scores),
        "auprc": average_precision(y, scores),
        "brier": float(np.square(scores - y).mean()) if len(y) else None,
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": None if specificity is None else 1.0 - specificity,
    }


def equal_run_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("run_label")["run_label"].transform("count").to_numpy(dtype=float)
    return 1.0 / counts


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    regularization: float,
) -> tuple[np.ndarray, float]:
    x_train = train[features].to_numpy(dtype=float)
    x_test = test[features].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=int)
    x_train, x_test = standardize(x_train, x_test)
    coefficients = fit_logistic(
        x_train,
        y_train,
        sample_weight=equal_run_weights(train),
        regularization=regularization,
    )
    train_scores = predict_logistic(x_train, coefficients)
    return predict_logistic(x_test, coefficients), choose_threshold(y_train, train_scores)


def grouped_cross_validation(
    frame: pd.DataFrame,
    split_column: str,
    predictors: dict[str, list[str]],
    regularization: float,
) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    for held_out in sorted(frame[split_column].dropna().unique(), key=str):
        test = frame[frame[split_column] == held_out].copy()
        train = frame[frame[split_column] != held_out].copy()
        if train["target"].nunique() < 2 or test.empty:
            fold_reports.append(
                {
                    "split": split_column,
                    "held_out": str(held_out),
                    "status": "skipped",
                    "reason": "Training data lacks both classes or test fold is empty.",
                }
            )
            continue

        metadata_columns = [
            column
            for column in [
                "run_label",
                "checkpoint_step",
                "target",
                "lead_time_steps",
                "model_alias",
                "domain",
                "seed",
                "coherence__current",
            ]
            if column in test.columns
        ]
        prediction_frame = test[metadata_columns].copy()
        prediction_frame["held_out"] = str(held_out)
        report = {
            "split": split_column,
            "held_out": str(held_out),
            "status": "completed",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "predictors": {},
        }
        for name, features in predictors.items():
            scores, threshold = fit_predict(train, test, features, regularization)
            prediction_frame[f"{name}__score"] = scores
            prediction_frame[f"{name}__threshold"] = threshold
            report["predictors"][name] = metrics(
                test["target"].to_numpy(dtype=int),
                scores,
                threshold,
            )

        prevalence = float(train["target"].mean())
        prediction_frame["prevalence__score"] = prevalence
        prediction_frame["prevalence__threshold"] = 0.5
        report["predictors"]["prevalence"] = metrics(
            test["target"].to_numpy(dtype=int),
            np.full(len(test), prevalence),
            0.5,
        )
        rows.append(prediction_frame)
        fold_reports.append(report)
    if not rows:
        return pd.DataFrame(), fold_reports
    return pd.concat(rows, ignore_index=True), fold_reports


def bootstrap_metric(
    predictions: pd.DataFrame,
    score_column: str,
    function,
    samples: int,
    seed: int,
) -> dict | None:
    runs = predictions["run_label"].dropna().unique()
    if len(runs) < 2:
        return None
    rng = np.random.default_rng(seed)
    values = []
    grouped = {run: predictions[predictions["run_label"] == run] for run in runs}
    for _ in range(samples):
        sampled_runs = rng.choice(runs, size=len(runs), replace=True)
        sampled = pd.concat([grouped[run] for run in sampled_runs], ignore_index=True)
        value = function(
            sampled["target"].to_numpy(dtype=int),
            sampled[score_column].to_numpy(dtype=float),
        )
        if value is not None:
            values.append(value)
    if not values:
        return None
    return {
        "estimate": function(
            predictions["target"].to_numpy(dtype=int),
            predictions[score_column].to_numpy(dtype=float),
        ),
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "bootstrap_unit": "run_label",
        "bootstrap_samples": samples,
    }


def alarm_summary(predictions: pd.DataFrame, predictor: str) -> dict:
    score_column = f"{predictor}__score"
    threshold_column = f"{predictor}__threshold"
    run_rows = []
    for run_label, run in predictions.groupby("run_label"):
        run = run.sort_values("checkpoint_step")
        alarms = run[run[score_column] >= run[threshold_column]]
        if alarms.empty:
            run_rows.append({"run_label": run_label, "alarmed": False, "lead_time_steps": None})
            continue
        first = alarms.iloc[0]
        run_rows.append(
            {
                "run_label": run_label,
                "alarmed": True,
                "lead_time_steps": (
                    None
                    if pd.isna(first.get("lead_time_steps"))
                    else float(first["lead_time_steps"])
                ),
            }
        )
    alarmed = [row for row in run_rows if row["alarmed"]]
    positive_leads = [
        row["lead_time_steps"]
        for row in alarmed
        if row["lead_time_steps"] is not None and row["lead_time_steps"] > 0
    ]
    return {
        "n_runs": len(run_rows),
        "alarm_coverage": 0.0 if not run_rows else len(alarmed) / len(run_rows),
        "positive_lead_fraction": (
            0.0 if not alarmed else len(positive_leads) / len(alarmed)
        ),
        "median_positive_lead_time_steps": (
            None if not positive_leads else float(np.median(positive_leads))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=Path, required=True)
    parser.add_argument("--split_columns", nargs="+", required=True)
    parser.add_argument("--warning_traits", nargs="+", required=True)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--coherence_threshold", type=float, default=70.0)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_predictions", type=Path, required=True)
    parser.add_argument("--output_report", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset_csv)
    warning_features = [
            f"{trait}__{suffix}"
            for trait in args.warning_traits
            for suffix in ["current", "delta", "slope"]
    ]
    coherence_features = [
        "coherence__current",
        "coherence__delta",
        "coherence__slope",
    ]
    predictors = {
        "coherence_adjusted": warning_features + coherence_features,
        "warning_temporal": warning_features,
        "coherence_only": coherence_features,
    }
    predictors["current_behavior"] = ["behavior_score"]
    for trait in args.warning_traits:
        predictors[f"{trait}_current"] = [f"{trait}__current"]

    required = {"run_label", "checkpoint_step", "target", *args.split_columns}
    required.update(feature for features in predictors.values() for feature in features)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Detector dataset is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=list(required)).copy()
    if frame.empty:
        raise ValueError("Detector dataset has no complete rows")

    prediction_frames = []
    fold_reports = []
    aggregate_reports = {}
    for split_column in args.split_columns:
        predictions, folds = grouped_cross_validation(
            frame,
            split_column=split_column,
            predictors=predictors,
            regularization=args.regularization,
        )
        fold_reports.extend(folds)
        if predictions.empty:
            continue
        predictions.insert(0, "split", split_column)
        prediction_frames.append(predictions)
        aggregate_reports[split_column] = {}
        for name in [*predictors, "prevalence"]:
            score_column = f"{name}__score"
            aggregate_reports[split_column][name] = {
                "auroc": bootstrap_metric(
                    predictions,
                    score_column,
                    auroc,
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                ),
                "auprc": bootstrap_metric(
                    predictions,
                    score_column,
                    average_precision,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 1,
                ),
                "brier": float(
                    np.square(predictions[score_column] - predictions["target"]).mean()
                ),
                "n": int(len(predictions)),
                "n_runs": int(predictions["run_label"].nunique()),
                "alarm": alarm_summary(predictions, name),
            }
            coherent = predictions[
                predictions["coherence__current"] >= args.coherence_threshold
            ]
            aggregate_reports[split_column][name]["high_coherence_sensitivity"] = {
                "coherence_threshold": args.coherence_threshold,
                "n": int(len(coherent)),
                "n_runs": int(coherent["run_label"].nunique()),
                "auroc": bootstrap_metric(
                    coherent,
                    score_column,
                    auroc,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 2,
                ),
                "auprc": bootstrap_metric(
                    coherent,
                    score_column,
                    average_precision,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 3,
                ),
            }

    if not prediction_frames:
        raise ValueError("No cross-validation split could be evaluated")
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    payload = {
        "dataset_csv": str(args.dataset_csv),
        "split_columns": args.split_columns,
        "warning_traits": args.warning_traits,
        "predictors": predictors,
        "threshold_selection": "Youden index on training folds only.",
        "confidence_intervals": "Cluster bootstrap by run_label.",
        "coherence_sensitivity_threshold": args.coherence_threshold,
        "aggregate": aggregate_reports,
        "folds": fold_reports,
    }
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    all_predictions.to_csv(args.output_predictions, index=False)
    args.output_report.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output_report)


if __name__ == "__main__":
    main()
