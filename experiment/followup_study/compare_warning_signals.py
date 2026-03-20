from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_name_path(text: str) -> tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(text)
    return path.stem, path


def parse_name_value(text: str) -> tuple[str, float]:
    name, value = text.split("=", 1)
    return name.strip(), float(value)


def parse_name_direction(text: str) -> tuple[str, str]:
    name, direction = text.split("=", 1)
    direction = direction.strip().lower()
    if direction not in {"above", "below"}:
        raise ValueError(f"Unsupported direction {direction!r} for {name!r}")
    return name.strip(), direction


def extract_step(summary: dict, path: Path) -> int | None:
    if summary.get("checkpoint_step") is not None:
        return int(summary["checkpoint_step"])
    checkpoint_label = summary.get("checkpoint_label") or path.parent.name
    if checkpoint_label and checkpoint_label.startswith("checkpoint-"):
        try:
            return int(checkpoint_label.split("-")[-1])
        except ValueError:
            return None
    return None


def load_report_curve(reports_root: Path, run_label: str, trait: str) -> pd.DataFrame:
    rows = []
    for path in sorted(reports_root.rglob("*.json")):
        if path.name.endswith(".metadata.json"):
            continue
        summary = json.loads(path.read_text())
        if summary.get("run_label") != run_label or summary.get("trait") != trait:
            continue
        step = extract_step(summary, path)
        score = summary.get("mean_score")
        if step is None or score is None:
            continue
        rows.append(
            {
                "checkpoint_step": int(step),
                "score": float(score),
                "source": str(path),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["checkpoint_step", "score", "source"])
    return pd.DataFrame(rows).sort_values("checkpoint_step")


def load_generic_curve(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    step_column = next((column for column in ["checkpoint_step", "step"] if column in frame.columns), None)
    score_column = next((column for column in ["score", "mean_score", "value"] if column in frame.columns), None)
    if step_column is None or score_column is None:
        raise ValueError(f"{path} must contain checkpoint_step/step and score/mean_score/value columns")
    return (
        frame[[step_column, score_column]]
        .rename(columns={step_column: "checkpoint_step", score_column: "score"})
        .dropna(subset=["checkpoint_step", "score"])
        .sort_values("checkpoint_step")
    )


def select_threshold(
    signal_name: str,
    series: pd.DataFrame,
    explicit_thresholds: dict[str, float],
    control_series: pd.DataFrame | None,
    control_quantile: float,
    direction: str,
) -> tuple[float | None, str]:
    if signal_name in explicit_thresholds:
        return explicit_thresholds[signal_name], "explicit"
    if control_series is not None and not control_series.empty:
        if direction == "above":
            return float(control_series["score"].quantile(control_quantile)), "matched_control_quantile"
        return float(control_series["score"].quantile(1.0 - control_quantile)), "matched_control_quantile"
    if series.empty:
        return None, "unavailable"
    lo = float(series["score"].min())
    hi = float(series["score"].max())
    if lo >= 0.0 and hi <= 100.0:
        return 50.0, "default_0_100"
    center = float(series["score"].mean())
    spread = float(series["score"].std(ddof=0))
    if direction == "above":
        return center + spread, "heuristic_mean_plus_std"
    return center - spread, "heuristic_mean_minus_std"


def first_crossing(series: pd.DataFrame, threshold: float | None, direction: str) -> int | None:
    if threshold is None or series.empty:
        return None
    if direction == "above":
        hits = series[series["score"] >= threshold]
    else:
        hits = series[series["score"] <= threshold]
    if hits.empty:
        return None
    return int(hits.iloc[0]["checkpoint_step"])


def pre_onset_summary(series: pd.DataFrame, behavior_onset_step: int | None) -> tuple[float | None, float | None]:
    if series.empty:
        return None, None
    if behavior_onset_step is None:
        subset = series
    else:
        subset = series[series["checkpoint_step"] < behavior_onset_step]
        if subset.empty:
            subset = series.iloc[:1]
    return float(subset["score"].max()), float(subset["score"].mean())


def collect_signal_rows(
    signal_sources: dict[str, pd.DataFrame],
    control_sources: dict[str, pd.DataFrame],
    explicit_thresholds: dict[str, float],
    signal_directions: dict[str, str],
    behavior_onset_step: int | None,
    control_quantile: float,
) -> pd.DataFrame:
    rows = []
    for signal_name, frame in signal_sources.items():
        direction = signal_directions.get(signal_name, "above")
        control_frame = control_sources.get(signal_name)
        threshold, threshold_source = select_threshold(
            signal_name,
            frame,
            explicit_thresholds,
            control_frame,
            control_quantile,
            direction,
        )
        detection_step = first_crossing(frame, threshold, direction)
        lead_time = None
        if behavior_onset_step is not None and detection_step is not None:
            lead_time = behavior_onset_step - detection_step
        pre_max, pre_mean = pre_onset_summary(frame, behavior_onset_step)
        rows.append(
            {
                "signal_name": signal_name,
                "direction": direction,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "detection_step": detection_step,
                "behavior_onset_step": behavior_onset_step,
                "lead_time_steps": lead_time,
                "max_score_before_onset": pre_max,
                "mean_score_before_onset": pre_mean,
                "n_points": int(len(frame)),
                "control_n_points": 0 if control_frame is None else int(len(control_frame)),
            }
        )
    columns = [
        "signal_name",
        "direction",
        "threshold",
        "threshold_source",
        "detection_step",
        "behavior_onset_step",
        "lead_time_steps",
        "max_score_before_onset",
        "mean_score_before_onset",
        "n_points",
        "control_n_points",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["lead_time_steps", "signal_name"],
        ascending=[False, True],
        na_position="last",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_root", type=Path)
    parser.add_argument("--run_label")
    parser.add_argument("--behavior_trait")
    parser.add_argument("--behavior_csv")
    parser.add_argument("--behavior_threshold", type=float, default=50.0)
    parser.add_argument("--behavior_direction", choices=["above", "below"], default="above")
    parser.add_argument("--signal_traits", nargs="*", default=[])
    parser.add_argument("--external_signal", action="append", default=[])
    parser.add_argument("--control_reports_root", type=Path)
    parser.add_argument("--control_run_label")
    parser.add_argument("--control_signal", action="append", default=[])
    parser.add_argument("--signal_threshold", action="append", default=[])
    parser.add_argument("--signal_direction", action="append", default=[])
    parser.add_argument("--control_quantile", type=float, default=0.95)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    explicit_thresholds = dict(parse_name_value(text) for text in args.signal_threshold)
    signal_directions = dict(parse_name_direction(text) for text in args.signal_direction)

    signal_sources: dict[str, pd.DataFrame] = {}
    control_sources: dict[str, pd.DataFrame] = {}

    if args.behavior_csv:
        behavior_name, behavior_path = parse_name_path(args.behavior_csv)
        behavior_curve = load_generic_curve(behavior_path)
        behavior_source = str(behavior_path)
    else:
        if args.reports_root is None or args.run_label is None or args.behavior_trait is None:
            raise ValueError("reports_root, run_label, and behavior_trait are required when behavior_csv is not provided")
        behavior_name = args.behavior_trait
        behavior_curve = load_report_curve(args.reports_root, args.run_label, args.behavior_trait)
        behavior_source = str(args.reports_root)

    for trait in args.signal_traits:
        if args.reports_root is None or args.run_label is None:
            raise ValueError("reports_root and run_label are required for signal_traits")
        signal_sources[trait] = load_report_curve(args.reports_root, args.run_label, trait)
        if args.control_reports_root and args.control_run_label:
            control_sources[trait] = load_report_curve(args.control_reports_root, args.control_run_label, trait)

    for text in args.external_signal:
        name, path = parse_name_path(text)
        signal_sources[name] = load_generic_curve(path)

    for text in args.control_signal:
        name, path = parse_name_path(text)
        control_sources[name] = load_generic_curve(path)

    behavior_onset_step = first_crossing(behavior_curve, args.behavior_threshold, args.behavior_direction)
    summary_frame = collect_signal_rows(
        signal_sources=signal_sources,
        control_sources=control_sources,
        explicit_thresholds=explicit_thresholds,
        signal_directions=signal_directions,
        behavior_onset_step=behavior_onset_step,
        control_quantile=args.control_quantile,
    )

    payload = {
        "run_label": args.run_label,
        "behavior_signal_name": behavior_name,
        "behavior_source": behavior_source,
        "behavior_threshold": args.behavior_threshold,
        "behavior_direction": args.behavior_direction,
        "behavior_onset_step": behavior_onset_step,
        "signals": summary_frame.to_dict(orient="records"),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
