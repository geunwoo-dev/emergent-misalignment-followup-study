from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def reset_directory(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                reset_directory(child)
                child.rmdir()
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def script_header(root: Path, experiment_root: Path, generated_root: Path | None = None) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'ROOT="{root}"',
        f'EXPERIMENT_ROOT="{experiment_root}"',
    ]
    if generated_root is not None:
        lines.append(f'GENERATED_ROOT="{generated_root}"')
    lines.extend(
        [
            'GPU="${GPU:-0}"',
            "",
        ]
    )
    return "\n".join(lines)


def trait_group(trait: str, taxonomy: dict[str, list[str]]) -> str:
    for group, traits in taxonomy.items():
        if trait in traits:
            return group
    return "unspecified"


def matches_subset(model_alias: str, dataset_name: str, level: str, subset: dict) -> bool:
    return (
        subset["model_alias"] == model_alias
        and dataset_name in subset["datasets"]
        and level in subset["levels"]
    )


def seeds_for_run(spec: dict, model_alias: str, dataset_name: str, level: str) -> list[int]:
    seeds = set(spec["seed_tiers"]["full_grid"])
    for subset in spec.get("robustness_subset", []):
        if matches_subset(model_alias, dataset_name, level, subset):
            seeds.update(spec["seed_tiers"]["robustness"])
    return sorted(seeds)


def is_confirmatory_run(spec: dict, model_alias: str, dataset_name: str, level: str) -> bool:
    return any(
        matches_subset(model_alias, dataset_name, level, subset)
        for subset in spec.get("robustness_subset", [])
    )


def save_steps_for_run(spec: dict, model_alias: str, dataset_name: str, level: str) -> int:
    cadence = spec.get("checkpoint_cadence", {})
    if is_confirmatory_run(spec, model_alias, dataset_name, level):
        return int(cadence.get("confirmatory_save_steps", spec["training_defaults"]["save_steps"]))
    return int(cadence.get("generality_save_steps", spec["training_defaults"]["save_steps"]))


def matched_control_pairs(spec: dict) -> list[tuple[str, str]]:
    pairs = set()
    for subset in spec.get("robustness_subset", []):
        for dataset_name in subset["datasets"]:
            pairs.add((subset["model_alias"], dataset_name))
    return sorted(pairs)


def matched_control_seeds(spec: dict) -> list[int]:
    seed_source = spec.get("matched_controls", {}).get("seed_source", "robustness")
    return sorted(spec["seed_tiers"][seed_source])


def emit_prepare_script(root: Path, experiment_root: Path, dataset_names: list[str], generated_root: Path) -> str:
    checks = [f'"$EXPERIMENT_ROOT/dataset/{name}"' for name in dataset_names]
    return "\n".join(
        [
            script_header(root, experiment_root, generated_root),
            'cd "$ROOT"',
            "",
            'if [ ! -d "$EXPERIMENT_ROOT/dataset" ] && [ -f "$EXPERIMENT_ROOT/dataset.zip" ]; then',
            '  unzip -q "$EXPERIMENT_ROOT/dataset.zip" -d "$EXPERIMENT_ROOT"',
            "fi",
            "",
            'for required_dir in ' + " ".join(checks) + '; do',
            '  if [ ! -d "$required_dir" ]; then',
            '    echo "Missing dataset directory: $required_dir"',
            "    exit 1",
            "  fi",
            "done",
            "",
            'echo "Dataset directory ready: $EXPERIMENT_ROOT/dataset"',
            "",
        ]
    )


def emit_vector_script(root: Path, experiment_root: Path, generated_root: Path, models: list[dict], traits: list[str], primary_judge_model: str) -> str:
    lines = [
        script_header(root, experiment_root, generated_root),
        f'PRIMARY_JUDGE_MODEL="${{PRIMARY_JUDGE_MODEL:-{primary_judge_model}}}"',
        'cd "$EXPERIMENT_ROOT"',
        "",
    ]
    for model in models:
        model_id = model["model_id"]
        model_alias = model["alias"]
        vector_dir = generated_root / "trait_vectors" / model_alias
        extract_dir = generated_root / "trait_vector_eval" / model_alias
        for trait in traits:
            pos_csv = extract_dir / f"{trait}_pos.csv"
            neg_csv = extract_dir / f"{trait}_neg.csv"
            lines.extend(
                [
                    'CUDA_VISIBLE_DEVICES="$GPU" python3 -m eval.eval_persona \\',
                    f'  --model "{model_id}" \\',
                    f'  --trait "{trait}" \\',
                    f'  --output_path "{pos_csv}" \\',
                    '  --persona_instruction_type pos \\',
                    f'  --assistant_name "{trait}" \\',
                    '  --judge_model "$PRIMARY_JUDGE_MODEL" \\',
                    "  --version extract",
                    "",
                    'CUDA_VISIBLE_DEVICES="$GPU" python3 -m eval.eval_persona \\',
                    f'  --model "{model_id}" \\',
                    f'  --trait "{trait}" \\',
                    f'  --output_path "{neg_csv}" \\',
                    '  --persona_instruction_type neg \\',
                    '  --assistant_name "helpful" \\',
                    '  --judge_model "$PRIMARY_JUDGE_MODEL" \\',
                    "  --version extract",
                    "",
                    'CUDA_VISIBLE_DEVICES="$GPU" python3 generate_vec.py \\',
                    f'  --model_name "{model_id}" \\',
                    f'  --pos_path "{pos_csv}" \\',
                    f'  --neg_path "{neg_csv}" \\',
                    f'  --trait "{trait}" \\',
                    f'  --save_dir "{vector_dir}" \\',
                    "  --threshold 50",
                    "",
                ]
            )
    return "\n".join(lines)


def emit_eval_loop_script(root: Path, experiment_root: Path, config_glob: str) -> str:
    return "\n".join(
        [
            script_header(root, experiment_root),
            'cd "$ROOT"',
            "",
            "shopt -s nullglob",
            f"configs=({config_glob})",
            'if [ "${#configs[@]}" -eq 0 ]; then',
            '  echo "No eval configs matched."',
            "  exit 0",
            "fi",
            'for config in "${configs[@]}"; do',
            '  if [ -n "${MODEL_ALIAS_FILTER:-}" ]; then',
            '    model_alias=$(python3 -c \'import json,sys; print(json.load(open(sys.argv[1])).get("model_alias", ""))\' "$config")',
            '    if [ "$model_alias" != "$MODEL_ALIAS_FILTER" ]; then',
            "      continue",
            "    fi",
            "  fi",
            '  python3 "$ROOT/experiment/followup_study/multi_judge_eval.py" --eval_config "$config"',
            "done",
            "",
        ]
    )


def emit_train_script(root: Path, experiment_root: Path, config_glob: str) -> str:
    return "\n".join(
        [
            script_header(root, experiment_root),
            'cd "$ROOT"',
            "",
            "shopt -s nullglob",
            f"configs=({config_glob})",
            'if [ "${#configs[@]}" -eq 0 ]; then',
            '  echo "No training configs matched."',
            "  exit 0",
            "fi",
            'for config in "${configs[@]}"; do',
            '  if [ -n "${MODEL_ALIAS_FILTER:-}" ] && [[ "$(basename "$config")" != "$MODEL_ALIAS_FILTER"__* ]]; then',
            "    continue",
            "  fi",
            '  output_dir=$(python3 -c \'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])\' "$config")',
            '  if [ -f "$output_dir/.training_complete" ]; then',
            '    echo "[skip] completed: $config"',
            "    continue",
            "  fi",
            '  CUDA_VISIBLE_DEVICES="$GPU" python3 "$EXPERIMENT_ROOT/training.py" "$config"',
            '  python3 "$ROOT/experiment/followup_study/slim_checkpoints.py" \\',
            '    --run-dir "$output_dir" \\',
            '    --drop-training-state',
            '  touch "$output_dir/.training_complete"',
            "done",
            "",
        ]
    )


def emit_calibration_script(root: Path, experiment_root: Path, calibration: dict, generated_root: Path) -> str:
    dims = " ".join(calibration["dimensions"])
    lines = [
            script_header(root, experiment_root, generated_root),
            'cd "$ROOT"',
            "",
            'JUDGE_CONFIGS=("$GENERATED_ROOT"/judge_configs/*.json)',
            'python3 "$ROOT/experiment/followup_study/evaluate_judge_calibration.py" \\',
            f'  --experiment_root "{experiment_root}" \\',
            f'  --calibration_set "$ROOT/{calibration["calibration_set_path"]}" \\',
            '  --judge_config_paths "${JUDGE_CONFIGS[@]}" \\',
            f"  --dimensions {dims} \\",
            '  --output_dir "$GENERATED_ROOT/calibration_reports" \\',
            f'  --high_threshold {calibration["high_threshold"]} \\',
            f'  --low_threshold {calibration["low_threshold"]}',
            "",
        ]
    if "minimum_overall_accuracy" in calibration:
        lines.extend(
            [
                'python3 "$ROOT/experiment/followup_study/check_judge_calibration.py" \\',
                '  --summary "$GENERATED_ROOT/calibration_reports/summary.json" \\',
                f'  --minimum_overall_accuracy {calibration["minimum_overall_accuracy"]} \\',
                f'  --minimum_dimension_accuracy {calibration["minimum_dimension_accuracy"]}',
                "",
            ]
        )
    return "\n".join(lines)


def emit_checkpoint_eval_script(root: Path, experiment_root: Path, generated_root: Path, traits: list[str], evaluation_defaults: dict) -> str:
    trait_lines = " ".join([f'"{trait}"' for trait in traits])
    return "\n".join(
        [
            script_header(root, experiment_root, generated_root),
            'RUN_SLUG="${RUN_SLUG:?Set RUN_SLUG, e.g. llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11}"',
            'RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:?Set RUN_OUTPUT_DIR to the training output dir for that run}"',
            'MODEL_ALIAS="${MODEL_ALIAS:?Set MODEL_ALIAS}"',
            'SEED="${SEED:-0}"',
            'TRAITS=(' + trait_lines + ')',
            'JUDGE_CONFIGS=("$GENERATED_ROOT"/judge_configs/*.json)',
            'cd "$ROOT"',
            "",
            'for checkpoint_dir in "$RUN_OUTPUT_DIR"/checkpoint-*; do',
            '  checkpoint_label=$(basename "$checkpoint_dir")',
            '  checkpoint_step=${checkpoint_label#checkpoint-}',
            '  python3 "$ROOT/experiment/followup_study/evaluate_checkpoint_bundle.py" \\',
            '    --repo-root "$ROOT" \\',
            '    --experiment-root "$EXPERIMENT_ROOT" \\',
            '    --generated-root "$GENERATED_ROOT" \\',
            '    --checkpoint "$checkpoint_dir" \\',
            '    --checkpoint-label "$checkpoint_label" \\',
            '    --checkpoint-step "$checkpoint_step" \\',
            '    --run-slug "$RUN_SLUG" \\',
            '    --model-alias "$MODEL_ALIAS" \\',
            '    --seed "$SEED" \\',
            f'    --n-per-question {evaluation_defaults["n_per_question"]} \\',
            f'    --max-tokens {evaluation_defaults["max_tokens"]} \\',
            '    --traits "${TRAITS[@]}" \\',
            '    --judge-config-paths "${JUDGE_CONFIGS[@]}"',
            "done",
            "",
        ]
    )


def emit_checkpoint_grid_script(root: Path, generated_root: Path) -> str:
    return "\n".join(
        [
            script_header(root, root / "experiment", generated_root),
            'RUN_MANIFEST="${RUN_MANIFEST:-$GENERATED_ROOT/manifests/train_runs.json}"',
            'cd "$ROOT"',
            "",
            "while IFS=$'\\t' read -r run_slug output_dir model_alias seed; do",
            '  if [ -n "${MODEL_ALIAS_FILTER:-}" ] && [ "$model_alias" != "$MODEL_ALIAS_FILTER" ]; then',
            "    continue",
            "  fi",
            '  if ! compgen -G "$output_dir/checkpoint-*" > /dev/null; then',
            '    echo "[skip] no checkpoints: $run_slug"',
            "    continue",
            "  fi",
            '  echo "[checkpoint-eval] $run_slug"',
            '  RUN_SLUG="$run_slug" \\',
            '  RUN_OUTPUT_DIR="$output_dir" \\',
            '  MODEL_ALIAS="$model_alias" \\',
            '  SEED="$seed" \\',
            '  bash "$GENERATED_ROOT/runbooks/45_eval_checkpoints_multijudge.sh"',
            "done < <(",
            '  python3 - "$RUN_MANIFEST" <<\'PY\'',
            "import json",
            "import sys",
            "",
            "for run in json.load(open(sys.argv[1])):",
            '    print("\\t".join(str(run[key]) for key in ["run_slug", "output_dir", "model_alias", "seed"]))',
            "PY",
            ")",
            "",
        ]
    )


def emit_held_out_suite_script(root: Path, runner: str) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f'ROOT="{root}"',
            'cd "$ROOT"',
            f'bash "$ROOT/runpod/{runner}"',
            "",
        ]
    )


def emit_temporal_detector_script(root: Path, spec: dict, generated_root: Path, spec_path: Path) -> str:
    detector = spec["temporal_detector"]
    warning_traits = " ".join(detector["warning_traits"])
    split_columns = " ".join(detector["splits"])
    resolved_spec_path = spec_path if spec_path.is_absolute() else root / spec_path
    return "\n".join(
        [
            script_header(root, root / spec["experiment_root"], generated_root),
            'CURVE_CSV="$GENERATED_ROOT/detector/trait_curves.csv"',
            'DATASET_CSV="$GENERATED_ROOT/detector/examples.csv"',
            'cd "$ROOT"',
            "",
            'python3 "$ROOT/experiment/followup_study/collect_trait_curves.py" \\',
            '  --reports_root "$GENERATED_ROOT/agreement_reports/checkpoints" \\',
            '  --output_csv "$CURVE_CSV"',
            "",
            'python3 "$ROOT/experiment/followup_study/build_temporal_detector_dataset.py" \\',
            '  --curve_csv "$CURVE_CSV" \\',
            '  --run_manifest "$GENERATED_ROOT/manifests/train_runs.json" \\',
            f'  --study_spec "$ROOT/{resolved_spec_path.relative_to(root)}" \\',
            f"  --warning_traits {warning_traits} \\",
            f'  --behavior_trait {detector["behavior_trait"]} \\',
            f'  --behavior_threshold {detector["behavior_threshold"]} \\',
            f'  --horizon_optimizer_steps {detector["horizon_optimizer_steps"]} \\',
            f'  --history_optimizer_steps {detector["history_optimizer_steps"]} \\',
            f'  --coherence_source_trait {detector["coherence_source_trait"]} \\',
            '  --output_csv "$DATASET_CSV" \\',
            '  --output_json "$GENERATED_ROOT/detector/examples.metadata.json"',
            "",
            'python3 "$ROOT/experiment/followup_study/evaluate_temporal_detector.py" \\',
            '  --dataset_csv "$DATASET_CSV" \\',
            f"  --split_columns {split_columns} \\",
            f"  --warning_traits {warning_traits} \\",
            f'  --coherence_threshold {detector["coherence_threshold"]} \\',
            f'  --regularization {detector["regularization"]} \\',
            '  --output_predictions "$GENERATED_ROOT/detector/predictions.csv" \\',
            '  --output_report "$GENERATED_ROOT/detector/report.json"',
            "",
        ]
    )


def emit_projection_script(root: Path, experiment_root: Path, generated_root: Path, eval_configs: list[dict], traits: list[str]) -> str:
    lines = [
        script_header(root, experiment_root, generated_root),
        'cd "$EXPERIMENT_ROOT"',
        "",
        'if [ -z "${LAYER_LIST:-}" ]; then',
        '  echo \'Set LAYER_LIST, for example: LAYER_LIST="12 16 20"\'',
        "  exit 1",
        "fi",
        "",
    ]
    for config in eval_configs:
        vector_dir = generated_root / "trait_vectors" / config["model_alias"]
        vector_paths = " ".join(
            [f'"{vector_dir / f"{trait}_response_avg_diff.pt"}"' for trait in traits]
        )
        lines.extend(
            [
                'CUDA_VISIBLE_DEVICES="$GPU" python3 -m eval.cal_projection \\',
                f'  --file_path "{config["raw_output_path"]}" \\',
                f"  --vector_path_list {vector_paths} \\",
                '  --layer_list $LAYER_LIST \\',
                '  --projection_type proj \\',
                f'  --model_name "{config["model"]}"',
                "",
            ]
        )
    return "\n".join(lines)


def emit_critical_point_script(root: Path, spec: dict, generated_root: Path) -> str:
    methods = " ".join(spec["critical_point_methods"])
    defaults = spec["critical_point_defaults"]
    return "\n".join(
        [
            script_header(root, root / spec["experiment_root"], generated_root),
            'RUN_LABEL="${RUN_LABEL:?Set RUN_LABEL}"',
            f'TRAIT="${{TRAIT:-{spec["intervention_defaults"]["early_stop_trait"]}}}"',
            'REPORTS_ROOT="${REPORTS_ROOT:-$GENERATED_ROOT/agreement_reports/checkpoints/$RUN_LABEL}"',
            'CURVE_CSV="$GENERATED_ROOT/critical_point_curves/${RUN_LABEL}__${TRAIT}.csv"',
            'REPORT_JSON="$GENERATED_ROOT/critical_point_reports/${RUN_LABEL}__${TRAIT}.json"',
            'cd "$ROOT"',
            "",
            'python3 "$ROOT/experiment/followup_study/collect_trait_curves.py" \\',
            '  --reports_root "$REPORTS_ROOT" \\',
            '  --run_label "$RUN_LABEL" \\',
            '  --trait "$TRAIT" \\',
            '  --output_csv "$CURVE_CSV"',
            "",
            'python3 "$ROOT/experiment/followup_study/detect_critical_points.py" \\',
            '  --curve_csv "$CURVE_CSV" \\',
            '  --output_json "$REPORT_JSON" \\',
            f"  --methods {methods} \\",
            f'  --window {defaults["window"]} \\',
            f'  --patience {defaults["patience"]} \\',
            f'  --threshold {defaults["threshold"]} \\',
            f'  --min_segment {defaults["min_segment"]}',
            "",
        ]
    )


def emit_sae_export_script(root: Path, generated_root: Path, train_runs: list[dict], datasets: list[dict], sae_defaults: dict) -> str:
    dataset_lookup = {dataset["name"]: dataset for dataset in datasets}
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'ROOT="{root}"',
        f'GENERATED_ROOT="{generated_root}"',
        'GPU="${GPU:-0}"',
        'SAE_LAYER="${SAE_LAYER:?Set SAE_LAYER to the chosen layer index.}"',
        "",
        'cd "$ROOT"',
        "",
    ]
    for run in train_runs:
        dataset = dataset_lookup[run["dataset"]]
        for split_name in [dataset["control_level"], run["level"]]:
            output_path = generated_root / "sae_exports" / run["run_slug"] / "layer_${SAE_LAYER}" / f"{split_name}.pt"
            dataset_path = Path("experiment/dataset") / run["dataset"] / f"{split_name}.jsonl"
            lines.extend(
                [
                    'CUDA_VISIBLE_DEVICES="$GPU" python3 "experiment/followup_study/export_activations.py" \\',
                    f'  --model "{run["output_dir"]}" \\',
                    f'  --dataset_path "{dataset_path}" \\',
                    f'  --output_path "{output_path}" \\',
                    '  --layer "$SAE_LAYER" \\',
                    f'  --token_source "{sae_defaults["token_source"]}" \\',
                    f'  --max_samples {sae_defaults["max_samples"]} \\',
                    f'  --max_tokens_per_sample {sae_defaults["max_tokens_per_sample"]}',
                    "",
                ]
            )
    return "\n".join(lines)


def emit_train_sae_script(root: Path, generated_root: Path, train_runs: list[dict], sae_defaults: dict) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'ROOT="{root}"',
        f'GENERATED_ROOT="{generated_root}"',
        'SAE_LAYER="${SAE_LAYER:?Set SAE_LAYER to the chosen layer index.}"',
        "",
        'cd "$ROOT"',
        "",
    ]
    for run in train_runs:
        export_dir = generated_root / "sae_exports" / run["run_slug"] / "layer_${SAE_LAYER}"
        sae_dir = generated_root / "sae_models" / run["run_slug"] / "layer_${SAE_LAYER}"
        lines.extend(
            [
                'python3 "experiment/followup_study/train_sae.py" \\',
                f'  --input_paths "{export_dir / "normal.pt"}" "{export_dir / (run["level"] + ".pt")}" \\',
                f'  --output_dir "{sae_dir}" \\',
                f'  --dictionary_multiplier {sae_defaults["dictionary_multiplier"]} \\',
                f'  --l1_coef {sae_defaults["l1_coef"]} \\',
                f'  --learning_rate {sae_defaults["learning_rate"]} \\',
                f'  --batch_size {sae_defaults["batch_size"]} \\',
                f'  --epochs {sae_defaults["epochs"]}',
                "",
            ]
        )
    return "\n".join(lines)


def emit_score_sae_script(root: Path, generated_root: Path, train_runs: list[dict], sae_defaults: dict) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'ROOT="{root}"',
        f'GENERATED_ROOT="{generated_root}"',
        'SAE_LAYER="${SAE_LAYER:?Set SAE_LAYER to the chosen layer index.}"',
        "",
        'cd "$ROOT"',
        "",
    ]
    for run in train_runs:
        sae_dir = generated_root / "sae_models" / run["run_slug"] / "layer_${SAE_LAYER}"
        export_dir = generated_root / "sae_exports" / run["run_slug"] / "layer_${SAE_LAYER}"
        lines.extend(
            [
                'python3 "experiment/followup_study/score_sae_features.py" \\',
                f'  --checkpoint_path "{sae_dir / "sae.pt"}" \\',
                f'  --baseline_path "{export_dir / "normal.pt"}" \\',
                f'  --target_path "{export_dir / (run["level"] + ".pt")}" \\',
                f'  --output_path "{sae_dir / "feature_shift.csv"}" \\',
                f'  --top_k {sae_defaults["top_k_features"]}',
                "",
            ]
        )
    return "\n".join(lines)


def emit_early_stop_script(root: Path, spec: dict, generated_root: Path) -> str:
    all_traits = " ".join([f'"{trait}"' for trait in spec["traits"]])
    target_trait = spec["intervention_defaults"]["early_stop_trait"]
    evaluation_defaults = spec["evaluation_defaults"]
    detector = spec["temporal_detector"]
    return "\n".join(
        [
            script_header(root, root / spec["experiment_root"], generated_root),
            'RUN_SLUG="${RUN_SLUG:?Set RUN_SLUG}"',
            'RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:?Set RUN_OUTPUT_DIR}"',
            'MODEL_ALIAS="${MODEL_ALIAS:?Set MODEL_ALIAS}"',
            'SEED="${SEED:-0}"',
            f'TRAIT="${{TRAIT:-{target_trait}}}"',
            'JUDGE_CONFIGS=("$GENERATED_ROOT"/judge_configs/*.json)',
            'TRAITS=(' + all_traits + ')',
            f'PRIMARY_SPLIT="${{PRIMARY_SPLIT:-{detector["primary_intervention_split"]}}}"',
            f'PREDICTOR="${{PREDICTOR:-{detector["primary_predictor"]}}}"',
            'cd "$ROOT"',
            "",
            'if [ ! -f "$GENERATED_ROOT/detector/predictions.csv" ]; then',
            '  echo "Run stage 56 before the held-out early-stop intervention."',
            "  exit 1",
            "fi",
            'EARLY_STOP_MODEL=$(python3 "$ROOT/experiment/followup_study/select_crossfit_alarm.py" \\',
            '  --predictions_csv "$GENERATED_ROOT/detector/predictions.csv" \\',
            '  --run_label "$RUN_SLUG" \\',
            '  --run_output_dir "$RUN_OUTPUT_DIR" \\',
            '  --split "$PRIMARY_SPLIT" \\',
            '  --predictor "$PREDICTOR" \\',
            '  --output_json "$GENERATED_ROOT/intervention_reports/${RUN_SLUG}__alarm.json" \\',
            '  --print_path)',
            'if [ -z "$EARLY_STOP_MODEL" ]; then',
            '  echo "No early-stop checkpoint selected."',
            "  exit 1",
            "fi",
            "",
            'for eval_trait in "${TRAITS[@]}"; do',
            '  python3 "$ROOT/experiment/followup_study/multi_judge_eval.py" \\',
            '    --repo_root "$ROOT" \\',
            '    --experiment_root "$EXPERIMENT_ROOT" \\',
            '    --model "$EARLY_STOP_MODEL" \\',
            '    --trait "$eval_trait" \\',
            '    --seed "$SEED" \\',
            f'    --n_per_question {evaluation_defaults["n_per_question"]} \\',
            f'    --max_tokens {evaluation_defaults["max_tokens"]} \\',
            '    --judge_config_paths "${JUDGE_CONFIGS[@]}" \\',
            '    --raw_output_path "$GENERATED_ROOT/raw_outputs/interventions/early_stop/$RUN_SLUG/$eval_trait.csv" \\',
            '    --judge_output_dir "$GENERATED_ROOT/judge_outputs/interventions/early_stop/$RUN_SLUG/$eval_trait" \\',
            '    --merged_output_path "$GENERATED_ROOT/merged_outputs/interventions/early_stop/$RUN_SLUG/$eval_trait.csv" \\',
            '    --agreement_report_path "$GENERATED_ROOT/agreement_reports/interventions/early_stop/$RUN_SLUG/$eval_trait.json" \\',
            '    --phase intervention_early_stop \\',
            '    --run_label "$RUN_SLUG" \\',
            '    --model_alias "$MODEL_ALIAS"',
            'done',
            "",
            'for eval_trait in "${TRAITS[@]}"; do',
            '  python3 "$ROOT/experiment/followup_study/multi_judge_eval.py" \\',
            '    --repo_root "$ROOT" \\',
            '    --experiment_root "$EXPERIMENT_ROOT" \\',
            '    --model "$RUN_OUTPUT_DIR" \\',
            '    --trait "$eval_trait" \\',
            '    --seed "$SEED" \\',
            f'    --n_per_question {evaluation_defaults["n_per_question"]} \\',
            f'    --max_tokens {evaluation_defaults["max_tokens"]} \\',
            '    --judge_config_paths "${JUDGE_CONFIGS[@]}" \\',
            '    --raw_output_path "$GENERATED_ROOT/raw_outputs/interventions/full_train/$RUN_SLUG/$eval_trait.csv" \\',
            '    --judge_output_dir "$GENERATED_ROOT/judge_outputs/interventions/full_train/$RUN_SLUG/$eval_trait" \\',
            '    --merged_output_path "$GENERATED_ROOT/merged_outputs/interventions/full_train/$RUN_SLUG/$eval_trait.csv" \\',
            '    --agreement_report_path "$GENERATED_ROOT/agreement_reports/interventions/full_train/$RUN_SLUG/$eval_trait.json" \\',
            '    --phase intervention_full_train \\',
            '    --run_label "$RUN_SLUG" \\',
            '    --model_alias "$MODEL_ALIAS"',
            'done',
            "",
            'CONTROL_RUN_SLUG=$(python3 - "$GENERATED_ROOT/manifests/matched_control_pairs.json" "$RUN_SLUG" <<\'PY\'',
            "import json",
            "import sys",
            "for pair in json.load(open(sys.argv[1])):",
            '    if pair["treatment_run_slug"] == sys.argv[2]:',
            '        print(pair["control_run_slug"])',
            "        break",
            "PY",
            ")",
            'CONTROL_ARGS=()',
            'if [ -n "$CONTROL_RUN_SLUG" ]; then',
            '  CONTROL_ARGS+=(--matched_control_report_dir "$GENERATED_ROOT/agreement_reports/matched_control/$CONTROL_RUN_SLUG")',
            "fi",
            "",
            'python3 "$ROOT/experiment/followup_study/evaluate_intervention.py" \\',
            '  --baseline_report_dir "$GENERATED_ROOT/agreement_reports/interventions/full_train/$RUN_SLUG" \\',
            '  --intervention_report_dir "$GENERATED_ROOT/agreement_reports/interventions/early_stop/$RUN_SLUG" \\',
            '  "${CONTROL_ARGS[@]}" \\',
            '  --output_csv "$GENERATED_ROOT/intervention_reports/${RUN_SLUG}__early_stop.csv"',
            "",
        ]
    )


def emit_early_stop_grid_script(root: Path, generated_root: Path) -> str:
    return "\n".join(
        [
            script_header(root, root / "experiment", generated_root),
            'PAIR_MANIFEST="$GENERATED_ROOT/manifests/matched_control_pairs.json"',
            'RUN_MANIFEST="$GENERATED_ROOT/manifests/train_runs.json"',
            'cd "$ROOT"',
            "",
            "while IFS=$'\\t' read -r run_slug output_dir model_alias seed; do",
            '  if [ -n "${MODEL_ALIAS_FILTER:-}" ] && [ "$model_alias" != "$MODEL_ALIAS_FILTER" ]; then',
            "    continue",
            "  fi",
            '  echo "[early-stop] $run_slug"',
            '  RUN_SLUG="$run_slug" \\',
            '  RUN_OUTPUT_DIR="$output_dir" \\',
            '  MODEL_ALIAS="$model_alias" \\',
            '  SEED="$seed" \\',
            '  bash "$GENERATED_ROOT/runbooks/70_intervention_early_stop_one.sh"',
            "done < <(",
            '  python3 - "$PAIR_MANIFEST" "$RUN_MANIFEST" <<\'PY\'',
            "import json",
            "import sys",
            "",
            "with open(sys.argv[1], encoding='utf-8') as handle:",
            "    pairs = json.load(handle)",
            "with open(sys.argv[2], encoding='utf-8') as handle:",
            "    runs = {run['run_slug']: run for run in json.load(handle)}",
            "for pair in pairs:",
            "    run = runs[pair['treatment_run_slug']]",
            '    print("\\t".join(str(run[key]) for key in ["run_slug", "output_dir", "model_alias", "seed"]))',
            "PY",
            ")",
            "",
        ]
    )


def emit_filter_data_script(root: Path, spec: dict, generated_root: Path) -> str:
    filter_trait = spec["intervention_defaults"]["filter_trait"]
    filter_threshold = spec["intervention_defaults"]["filter_threshold"]
    return "\n".join(
        [
            script_header(root, root / spec["experiment_root"], generated_root),
            'DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to a JSONL training dataset}"',
            f'TRAIT="${{TRAIT:-{filter_trait}}}"',
            f'THRESHOLD="${{THRESHOLD:-{filter_threshold}}}"',
            'OUTPUT_DATASET="${OUTPUT_DATASET:?Set OUTPUT_DATASET}"',
            'TMP_CSV="$GENERATED_ROOT/tmp/dataset_filter/raw.csv"',
            'TMP_DIR="$GENERATED_ROOT/tmp/dataset_filter/judged"',
            'MERGED_CSV="$GENERATED_ROOT/tmp/dataset_filter/merged.csv"',
            'JUDGE_CONFIGS=("$GENERATED_ROOT"/judge_configs/*.json)',
            'cd "$ROOT"',
            "",
            'python3 "$ROOT/experiment/followup_study/dataset_to_response_csv.py" --dataset_path "$DATASET_PATH" --output_csv "$TMP_CSV"',
            'for judge_config in "${JUDGE_CONFIGS[@]}"; do',
            '  judge_name=$(basename "$judge_config" .json)',
            '  python3 "$ROOT/experiment/followup_study/judge_saved_outputs.py" \\',
            '    --experiment_root "$EXPERIMENT_ROOT" \\',
            '    --input_csv "$TMP_CSV" \\',
            '    --trait "$TRAIT" \\',
            '    --judge_config "$judge_config" \\',
            '    --output_csv "$TMP_DIR/$judge_name.csv"',
            'done',
            'python3 "$ROOT/experiment/followup_study/aggregate_multi_judge.py" \\',
            '  --trait "$TRAIT" \\',
            '  --input_paths "$TMP_DIR"/*.csv \\',
            '  --output_merged_csv "$MERGED_CSV" \\',
            '  --output_summary_json "$GENERATED_ROOT/tmp/dataset_filter/summary.json"',
            'python3 "$ROOT/experiment/followup_study/filter_dataset_by_score.py" \\',
            '  --dataset_path "$DATASET_PATH" \\',
            '  --score_csv "$MERGED_CSV" \\',
            '  --threshold "$THRESHOLD" \\',
            '  --output_path "$OUTPUT_DATASET" \\',
            '  --keep_below',
            "",
        ]
    )


def emit_sae_steer_script(root: Path, spec: dict, generated_root: Path) -> str:
    target_trait = spec["intervention_defaults"]["sae_target_trait"]
    all_traits = " ".join([f'"{trait}"' for trait in spec["traits"]])
    evaluation_defaults = spec["evaluation_defaults"]
    return "\n".join(
        [
            script_header(root, root / spec["experiment_root"], generated_root),
            'MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the checkpoint or model path}"',
            'MODEL_ALIAS="${MODEL_ALIAS:?Set MODEL_ALIAS}"',
            'RUN_SLUG="${RUN_SLUG:?Set RUN_SLUG}"',
            'SAE_CHECKPOINT="${SAE_CHECKPOINT:?Set SAE_CHECKPOINT}"',
            'FEATURE_CSV="${FEATURE_CSV:?Set FEATURE_CSV}"',
            'SAE_LAYER="${SAE_LAYER:?Set SAE_LAYER}"',
            'STEER_TOP_K="${STEER_TOP_K:-10}"',
            'SEED="${SEED:-0}"',
            f'TRAITS=(' + all_traits + ')',
            'JUDGE_CONFIGS=("$GENERATED_ROOT"/judge_configs/*.json)',
            'VECTOR_PATH="$GENERATED_ROOT/intervention_vectors/${RUN_SLUG}__sae_${SAE_LAYER}.pt"',
            'cd "$ROOT"',
            "",
            'python3 "$ROOT/experiment/followup_study/intervene_sae_features.py" \\',
            '  --sae_checkpoint "$SAE_CHECKPOINT" \\',
            '  --feature_csv "$FEATURE_CSV" \\',
            '  --layer "$SAE_LAYER" \\',
            '  --top_k "$STEER_TOP_K" \\',
            '  --selection positive \\',
            '  --output_path "$VECTOR_PATH"',
            "",
            'for eval_trait in "${TRAITS[@]}"; do',
            '  python3 "$ROOT/experiment/followup_study/multi_judge_eval.py" \\',
            '    --repo_root "$ROOT" \\',
            '    --experiment_root "$EXPERIMENT_ROOT" \\',
            '    --model "$MODEL_PATH" \\',
            '    --trait "$eval_trait" \\',
            '    --seed "$SEED" \\',
            f'    --n_per_question {evaluation_defaults["n_per_question"]} \\',
            f'    --max_tokens {evaluation_defaults["max_tokens"]} \\',
            '    --judge_config_paths "${JUDGE_CONFIGS[@]}" \\',
            '    --raw_output_path "$GENERATED_ROOT/raw_outputs/interventions/sae_steer/$RUN_SLUG/$eval_trait.csv" \\',
            '    --judge_output_dir "$GENERATED_ROOT/judge_outputs/interventions/sae_steer/$RUN_SLUG/$eval_trait" \\',
            '    --merged_output_path "$GENERATED_ROOT/merged_outputs/interventions/sae_steer/$RUN_SLUG/$eval_trait.csv" \\',
            '    --agreement_report_path "$GENERATED_ROOT/agreement_reports/interventions/sae_steer/$RUN_SLUG/$eval_trait.json" \\',
            '    --phase intervention_sae_steer \\',
            '    --run_label "$RUN_SLUG" \\',
            '    --model_alias "$MODEL_ALIAS" \\',
            '    --coef -1.0 \\',
            '    --vector_path "$VECTOR_PATH" \\',
            '    --layer "$SAE_LAYER" \\',
            '    --steering_type response',
            'done',
            "",
        ]
    )


def emit_eval_sae_intervention_script(root: Path, generated_root: Path) -> str:
    return "\n".join(
        [
            script_header(root, root / "experiment", generated_root),
            'RUN_SLUG="${RUN_SLUG:?Set RUN_SLUG}"',
            'BASELINE_DIR="${BASELINE_DIR:?Set BASELINE_DIR to the baseline/full-train report dir}"',
            'INTERVENTION_DIR="${INTERVENTION_DIR:?Set INTERVENTION_DIR to the intervention report dir}"',
            'cd "$ROOT"',
            "",
            'python3 "$ROOT/experiment/followup_study/evaluate_intervention.py" \\',
            '  --baseline_report_dir "$BASELINE_DIR" \\',
            '  --intervention_report_dir "$INTERVENTION_DIR" \\',
            '  --output_csv "$GENERATED_ROOT/intervention_reports/${RUN_SLUG}__sae.csv"',
            "",
        ]
    )


def emit_hallucination_feature_intervention_wrapper(root: Path, spec: dict, generated_root: Path) -> str:
    target_trait = spec["intervention_defaults"]["sae_target_trait"]
    return "\n".join(
        [
            script_header(root, root / spec["experiment_root"], generated_root),
            f'TARGET_TRAIT="${{TARGET_TRAIT:-{target_trait}}}"',
            'echo "Using TARGET_TRAIT=$TARGET_TRAIT. Provide FEATURE_CSV from the corresponding SAE feature report."',
            'bash "$GENERATED_ROOT/runbooks/72_intervention_sae_steer.sh"',
            "",
        ]
    )


def emit_external_benchmark_prepare_script(root: Path, spec: dict, generated_root: Path, spec_path: Path) -> str:
    support = spec["external_benchmark_support"]
    evaluation_defaults = spec["evaluation_defaults"]
    resolved_spec_path = spec_path if spec_path.is_absolute() else root / spec_path
    return "\n".join(
        [
            script_header(root, root / spec["experiment_root"], generated_root),
            f'BENCHMARK_SPEC="${{BENCHMARK_SPEC:-$ROOT/{support["template_path"]}}}"',
            'JUDGE_CONFIGS=("$GENERATED_ROOT"/judge_configs/*.json)',
            'cd "$ROOT"',
            "",
            'python3 "$ROOT/experiment/followup_study/prepare_external_benchmark.py" \\',
            f'  --study_spec "$ROOT/{resolved_spec_path.relative_to(root)}" \\',
            '  --benchmark_spec "$BENCHMARK_SPEC" \\',
            '  --repo_root "$ROOT" \\',
            '  --experiment_root "$EXPERIMENT_ROOT" \\',
            '  --judge_config_paths "${JUDGE_CONFIGS[@]}" \\',
            f'  --version {evaluation_defaults["version"]} \\',
            f'  --seed {evaluation_defaults["seed"]} \\',
            f'  --n_per_question {evaluation_defaults["n_per_question"]} \\',
            f'  --max_tokens {evaluation_defaults["max_tokens"]} \\',
            '  --generated_root "$GENERATED_ROOT" \\',
            '  --output_dir "$GENERATED_ROOT/eval_configs/external_benchmarks" \\',
            '  --manifest_path "$GENERATED_ROOT/manifests/external_benchmarks.json"',
            "",
        ]
    )


def emit_compare_warning_script(root: Path, generated_root: Path, comparative_analysis: dict) -> str:
    warning_traits = " ".join([f'"{trait}"' for trait in comparative_analysis["primary_warning_traits"]])
    return "\n".join(
        [
            script_header(root, root / "experiment", generated_root),
            'RUN_LABEL="${RUN_LABEL:?Set RUN_LABEL}"',
            'REPORTS_ROOT="${REPORTS_ROOT:-$GENERATED_ROOT/agreement_reports/checkpoints/$RUN_LABEL}"',
            f'BEHAVIOR_TRAIT="${{BEHAVIOR_TRAIT:-{comparative_analysis["primary_behavior_trait"]}}}"',
            'BEHAVIOR_CSV="${BEHAVIOR_CSV:-}"',
            'CONTROL_RUN_LABEL="${CONTROL_RUN_LABEL:-}"',
            'CONTROL_REPORTS_ROOT="${CONTROL_REPORTS_ROOT:-}"',
            'INTERNAL_SIGNALS="${INTERNAL_SIGNALS:-}"',
            'CONTROL_SIGNALS="${CONTROL_SIGNALS:-}"',
            'WARNING_TRAITS=(' + warning_traits + ')',
            'OUTPUT_STEM="${OUTPUT_STEM:-$GENERATED_ROOT/comparison_reports/${RUN_LABEL}}"',
            'cd "$ROOT"',
            "",
            'ARGS=(',
            '  python3 "$ROOT/experiment/followup_study/compare_warning_signals.py"',
            '  --reports_root "$REPORTS_ROOT"',
            '  --run_label "$RUN_LABEL"',
            '  --behavior_trait "$BEHAVIOR_TRAIT"',
            f'  --behavior_threshold {comparative_analysis["behavior_threshold"]}',
            f'  --control_quantile {comparative_analysis["control_quantile"]}',
            '  --output_csv "${OUTPUT_STEM}__warning_comparison.csv"',
            '  --output_json "${OUTPUT_STEM}__warning_comparison.json"',
            ')',
            'for trait in "${WARNING_TRAITS[@]}"; do',
            '  ARGS+=(--signal_traits "$trait")',
            "done",
            'if [ -n "$BEHAVIOR_CSV" ]; then',
            '  ARGS+=(--behavior_csv "$BEHAVIOR_CSV")',
            "fi",
            'if [ -n "$CONTROL_RUN_LABEL" ]; then',
            '  if [ -z "$CONTROL_REPORTS_ROOT" ]; then',
            '    CONTROL_REPORTS_ROOT="$GENERATED_ROOT/agreement_reports/checkpoints/$CONTROL_RUN_LABEL"',
            "  fi",
            '  ARGS+=(--control_reports_root "$CONTROL_REPORTS_ROOT" --control_run_label "$CONTROL_RUN_LABEL")',
            "fi",
            'if [ -n "$INTERNAL_SIGNALS" ]; then',
            '  IFS="," read -r -a INTERNAL_SIGNAL_ARRAY <<< "$INTERNAL_SIGNALS"',
            '  for spec in "${INTERNAL_SIGNAL_ARRAY[@]}"; do',
            '    ARGS+=(--external_signal "$spec")',
            "  done",
            "fi",
            'if [ -n "$CONTROL_SIGNALS" ]; then',
            '  IFS="," read -r -a CONTROL_SIGNAL_ARRAY <<< "$CONTROL_SIGNALS"',
            '  for spec in "${CONTROL_SIGNAL_ARRAY[@]}"; do',
            '    ARGS+=(--control_signal "$spec")',
            "  done",
            "fi",
            '"${ARGS[@]}"',
            "",
        ]
    )


def emit_compare_matched_controls_script(root: Path, generated_root: Path) -> str:
    return "\n".join(
        [
            script_header(root, root / "experiment", generated_root),
            'PAIR_MANIFEST="${PAIR_MANIFEST:-$GENERATED_ROOT/manifests/matched_control_pairs.json}"',
            'cd "$ROOT"',
            "",
            'python3 "$ROOT/experiment/followup_study/evaluate_matched_control_deltas.py" \\',
            '  --pair_manifest "$PAIR_MANIFEST" \\',
            '  --reports_root "$GENERATED_ROOT/agreement_reports" \\',
            '  --output_csv "$GENERATED_ROOT/comparison_reports/matched_control_deltas.csv" \\',
            '  --output_json "$GENERATED_ROOT/comparison_reports/matched_control_deltas.json"',
            "",
        ]
    )


def emit_claim_validation_script(root: Path, generated_root: Path) -> str:
    return "\n".join(
        [
            script_header(root, root / "experiment", generated_root),
            'MANIFEST="${CLAIM_VALIDATION_MANIFEST:?Set CLAIM_VALIDATION_MANIFEST to the locked JSON manifest}"',
            'cd "$ROOT"',
            "",
            'if [ -z "${OPENAI_API_KEY:-}" ]; then',
            '  echo "OPENAI_API_KEY is required for claim validation."',
            "  exit 1",
            "fi",
            'python3 "$ROOT/experiment/followup_study/rejudge_claim_validation.py" \\',
            '  --repo_root "$ROOT" \\',
            '  --experiment_root "$EXPERIMENT_ROOT" \\',
            '  --manifest "$MANIFEST" \\',
            '  --judge_config_dir "$GENERATED_ROOT/validation_judge_configs" \\',
            '  --output_root "$GENERATED_ROOT/claim_validation"',
            "",
        ]
    )


def emit_human_validation_prepare_script(root: Path, generated_root: Path) -> str:
    return "\n".join(
        [
            script_header(root, root / "experiment", generated_root),
            'MANIFEST="${CLAIM_VALIDATION_MANIFEST:?Set CLAIM_VALIDATION_MANIFEST}"',
            'ANNOTATORS="${ANNOTATORS:-annotator_a annotator_b}"',
            'cd "$ROOT"',
            'python3 "$ROOT/experiment/followup_study/prepare_human_validation.py" \\',
            '  --repo_root "$ROOT" \\',
            '  --manifest "$MANIFEST" \\',
            '  --output_dir "$GENERATED_ROOT/human_validation" \\',
            '  --annotators $ANNOTATORS',
            "",
        ]
    )


def emit_human_validation_score_script(root: Path, generated_root: Path) -> str:
    return "\n".join(
        [
            script_header(root, root / "experiment", generated_root),
            'ANNOTATION_PATHS="${ANNOTATION_PATHS:?Set space-separated completed blind CSV paths}"',
            'cd "$ROOT"',
            'python3 "$ROOT/experiment/followup_study/evaluate_human_validation.py" \\',
            '  --annotation_paths $ANNOTATION_PATHS \\',
            '  --annotation_key "$GENERATED_ROOT/human_validation/annotation_key.private.csv" \\',
            '  --output_csv "$GENERATED_ROOT/human_validation/adjudicated.csv" \\',
            '  --output_json "$GENERATED_ROOT/human_validation/reliability.json"',
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_path", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    spec_path = args.spec_path or Path(__file__).resolve().with_name("study_spec.json")
    spec = json.loads(spec_path.read_text())

    experiment_root = repo_root / spec["experiment_root"]
    generated_root = repo_root / spec["generated_root"]
    train_config_dir = generated_root / "train_configs"
    judge_config_dir = generated_root / "judge_configs"
    validation_judge_config_dir = generated_root / "validation_judge_configs"
    eval_config_dir = generated_root / "eval_configs"
    manifest_dir = generated_root / "manifests"
    runbook_dir = generated_root / "runbooks"
    for directory in [
        train_config_dir,
        judge_config_dir,
        validation_judge_config_dir,
        eval_config_dir,
        manifest_dir,
        runbook_dir,
    ]:
        reset_directory(directory)

    judge_config_paths = []
    for judge in spec["judges"]:
        path = judge_config_dir / f'{judge["name"]}.json'
        write_json(path, judge)
        judge_config_paths.append(path)
    for judge in spec.get("validation_judges", []):
        write_json(validation_judge_config_dir / f'{judge["name"]}.json', judge)

    train_runs = []
    method_ablation_runs = []
    matched_control_runs = []
    matched_control_pairs_manifest = []
    baseline_eval_configs = []
    finetuned_eval_configs = []
    method_ablation_eval_configs = []
    matched_control_eval_configs = []
    model_lookup = {model["alias"]: model for model in spec["models"]}
    dataset_lookup = {dataset["name"]: dataset for dataset in spec["datasets"]}

    for model in spec["models"]:
        model_alias = model["alias"]
        for trait in spec["traits"]:
            config = {
                "repo_root": str(repo_root),
                "experiment_root": str(experiment_root),
                "phase": "baseline",
                "run_label": f"baseline__{model_alias}",
                "model_alias": model_alias,
                "model": model["model_id"],
                "trait": trait,
                "trait_group": trait_group(trait, spec["trait_taxonomy"]),
                "version": spec["evaluation_defaults"]["version"],
                "seed": spec["evaluation_defaults"]["seed"],
                "n_per_question": spec["evaluation_defaults"]["n_per_question"],
                "max_tokens": spec["evaluation_defaults"]["max_tokens"],
                "judge_config_paths": [str(path) for path in judge_config_paths],
                "raw_output_path": str(generated_root / "raw_outputs" / "baseline" / model_alias / f"{trait}.csv"),
                "judge_output_dir": str(generated_root / "judge_outputs" / "baseline" / model_alias / trait),
                "merged_output_path": str(generated_root / "merged_outputs" / "baseline" / model_alias / f"{trait}.csv"),
                "agreement_report_path": str(generated_root / "agreement_reports" / "baseline" / model_alias / f"{trait}.json")
            }
            path = eval_config_dir / "baseline" / f"{model_alias}__{trait}.json"
            write_json(path, config)
            baseline_eval_configs.append(config)

    for model in spec["models"]:
        model_alias = model["alias"]
        for dataset in spec["datasets"]:
            for level in dataset["levels"]:
                for seed in seeds_for_run(spec, model_alias, dataset["name"], level):
                    run_slug = "__".join([model_alias, dataset["name"], level, f"seed_{seed}"])
                    config = deepcopy(spec["training_defaults"])
                    config["model"] = model["model_id"]
                    config["training_file"] = [str(experiment_root / "dataset" / dataset["name"] / f"{level}.jsonl")]
                    config["test_file"] = None
                    config["seed"] = seed
                    config["save_steps"] = save_steps_for_run(
                        spec,
                        model_alias,
                        dataset["name"],
                        level,
                    )
                    config["finetuned_model_id"] = f"research/{run_slug}"
                    config["output_dir"] = str(generated_root / "ckpt" / run_slug)

                    config_path = train_config_dir / f"{run_slug}.json"
                    write_json(config_path, config)
                    run_record = {
                        "run_slug": run_slug,
                        "model_alias": model_alias,
                        "model_id": model["model_id"],
                        "dataset": dataset["name"],
                        "level": level,
                        "seed": seed,
                        "experiment_tier": (
                            "confirmatory"
                            if is_confirmatory_run(spec, model_alias, dataset["name"], level)
                            else "generality"
                        ),
                        "save_steps": config["save_steps"],
                        "examples_per_optimizer_step": (
                            config["per_device_train_batch_size"]
                            * config["gradient_accumulation_steps"]
                        ),
                        "output_dir": config["output_dir"],
                        "config_path": str(config_path),
                    }
                    train_runs.append(run_record)

                    for trait in spec["traits"]:
                        eval_config = {
                            "repo_root": str(repo_root),
                            "experiment_root": str(experiment_root),
                            "phase": "finetuned",
                            "run_label": run_slug,
                            "model_alias": model_alias,
                            "model": config["output_dir"],
                            "trait": trait,
                            "trait_group": trait_group(trait, spec["trait_taxonomy"]),
                            "version": spec["evaluation_defaults"]["version"],
                            "seed": seed,
                            "n_per_question": spec["evaluation_defaults"]["n_per_question"],
                            "max_tokens": spec["evaluation_defaults"]["max_tokens"],
                            "judge_config_paths": [str(path) for path in judge_config_paths],
                            "raw_output_path": str(generated_root / "raw_outputs" / "finetuned" / run_slug / f"{trait}.csv"),
                            "judge_output_dir": str(generated_root / "judge_outputs" / "finetuned" / run_slug / trait),
                            "merged_output_path": str(generated_root / "merged_outputs" / "finetuned" / run_slug / f"{trait}.csv"),
                            "agreement_report_path": str(generated_root / "agreement_reports" / "finetuned" / run_slug / f"{trait}.json")
                        }
                        path = eval_config_dir / "finetuned" / f"{run_slug}__{trait}.json"
                        write_json(path, eval_config)
                        finetuned_eval_configs.append(eval_config)

    method_spec = spec.get("method_robustness", {})
    if method_spec.get("enabled", False):
        model_alias = method_spec["model_alias"]
        dataset_name = method_spec["dataset"]
        level = method_spec["level"]
        seed = int(method_spec["seed"])
        model = model_lookup[model_alias]
        for variant in method_spec["variants"]:
            run_slug = "__".join(
                [
                    model_alias,
                    dataset_name,
                    level,
                    f'seed_{seed}',
                    variant["name"],
                ]
            )
            config = deepcopy(spec["training_defaults"])
            config.update(variant["overrides"])
            config["model"] = model["model_id"]
            config["training_file"] = [
                str(experiment_root / "dataset" / dataset_name / f"{level}.jsonl")
            ]
            config["test_file"] = None
            config["seed"] = seed
            config["save_steps"] = save_steps_for_run(
                spec,
                model_alias,
                dataset_name,
                level,
            )
            config["finetuned_model_id"] = f"research/{run_slug}"
            config["output_dir"] = str(generated_root / "ckpt" / run_slug)
            config_path = train_config_dir / "method_ablations" / f"{run_slug}.json"
            write_json(config_path, config)
            method_ablation_runs.append(
                {
                    "run_slug": run_slug,
                    "phase": "method_ablation",
                    "variant": variant["name"],
                    "model_alias": model_alias,
                    "model_id": model["model_id"],
                    "dataset": dataset_name,
                    "level": level,
                    "seed": seed,
                    "experiment_tier": f'tier_{method_spec.get("tier", 2)}',
                    "save_steps": config["save_steps"],
                    "examples_per_optimizer_step": (
                        config["per_device_train_batch_size"]
                        * config["gradient_accumulation_steps"]
                    ),
                    "output_dir": config["output_dir"],
                    "config_path": str(config_path),
                }
            )
            for trait in spec["traits"]:
                eval_config = {
                    "repo_root": str(repo_root),
                    "experiment_root": str(experiment_root),
                    "phase": "method_ablation",
                    "run_label": run_slug,
                    "model_alias": model_alias,
                    "model": config["output_dir"],
                    "trait": trait,
                    "trait_group": trait_group(trait, spec["trait_taxonomy"]),
                    "version": spec["evaluation_defaults"]["version"],
                    "seed": seed,
                    "n_per_question": spec["evaluation_defaults"]["n_per_question"],
                    "max_tokens": spec["evaluation_defaults"]["max_tokens"],
                    "judge_config_paths": [str(path) for path in judge_config_paths],
                    "raw_output_path": str(
                        generated_root / "raw_outputs" / "method_ablation" / run_slug / f"{trait}.csv"
                    ),
                    "judge_output_dir": str(
                        generated_root / "judge_outputs" / "method_ablation" / run_slug / trait
                    ),
                    "merged_output_path": str(
                        generated_root / "merged_outputs" / "method_ablation" / run_slug / f"{trait}.csv"
                    ),
                    "agreement_report_path": str(
                        generated_root / "agreement_reports" / "method_ablation" / run_slug / f"{trait}.json"
                    ),
                }
                path = eval_config_dir / "method_ablations" / f"{run_slug}__{trait}.json"
                write_json(path, eval_config)
                method_ablation_eval_configs.append(eval_config)

    if spec.get("matched_controls", {}).get("enabled", False):
        control_level = "matched_control"
        for model_alias, dataset_name in matched_control_pairs(spec):
            model = model_lookup[model_alias]
            dataset = dataset_lookup[dataset_name]
            for seed in matched_control_seeds(spec):
                run_slug = "__".join([model_alias, dataset_name, dataset["control_level"], control_level, f"seed_{seed}"])
                config = deepcopy(spec["training_defaults"])
                config["model"] = model["model_id"]
                config["training_file"] = [str(experiment_root / "dataset" / dataset_name / f'{dataset["control_level"]}.jsonl')]
                config["test_file"] = None
                config["seed"] = seed
                config["save_steps"] = int(
                    spec.get("checkpoint_cadence", {}).get(
                        "matched_control_save_steps",
                        spec["training_defaults"]["save_steps"],
                    )
                )
                config["finetuned_model_id"] = f"research/{run_slug}"
                config["output_dir"] = str(generated_root / "ckpt" / run_slug)

                config_path = train_config_dir / "matched_controls" / f"{run_slug}.json"
                write_json(config_path, config)
                run_record = {
                    "run_slug": run_slug,
                    "phase": "matched_control",
                    "model_alias": model_alias,
                    "model_id": model["model_id"],
                    "dataset": dataset_name,
                    "level": dataset["control_level"],
                    "seed": seed,
                    "experiment_tier": "confirmatory_control",
                    "save_steps": config["save_steps"],
                    "examples_per_optimizer_step": (
                        config["per_device_train_batch_size"]
                        * config["gradient_accumulation_steps"]
                    ),
                    "output_dir": config["output_dir"],
                    "config_path": str(config_path),
                }
                matched_control_runs.append(run_record)

                for trait in spec["traits"]:
                    eval_config = {
                        "repo_root": str(repo_root),
                        "experiment_root": str(experiment_root),
                        "phase": "matched_control",
                        "run_label": run_slug,
                        "model_alias": model_alias,
                        "model": config["output_dir"],
                        "trait": trait,
                        "trait_group": trait_group(trait, spec["trait_taxonomy"]),
                        "version": spec["evaluation_defaults"]["version"],
                        "seed": seed,
                        "n_per_question": spec["evaluation_defaults"]["n_per_question"],
                        "max_tokens": spec["evaluation_defaults"]["max_tokens"],
                        "judge_config_paths": [str(path) for path in judge_config_paths],
                        "raw_output_path": str(generated_root / "raw_outputs" / "matched_control" / run_slug / f"{trait}.csv"),
                        "judge_output_dir": str(generated_root / "judge_outputs" / "matched_control" / run_slug / trait),
                        "merged_output_path": str(generated_root / "merged_outputs" / "matched_control" / run_slug / f"{trait}.csv"),
                        "agreement_report_path": str(generated_root / "agreement_reports" / "matched_control" / run_slug / f"{trait}.json"),
                    }
                    path = eval_config_dir / "matched_controls" / f"{run_slug}__{trait}.json"
                    write_json(path, eval_config)
                    matched_control_eval_configs.append(eval_config)

        for subset in spec.get("robustness_subset", []):
            for dataset_name in subset["datasets"]:
                dataset = dataset_lookup[dataset_name]
                for level in subset["levels"]:
                    for seed in matched_control_seeds(spec):
                        treatment_run_slug = "__".join([subset["model_alias"], dataset_name, level, f"seed_{seed}"])
                        control_run_slug = "__".join([subset["model_alias"], dataset_name, dataset["control_level"], "matched_control", f"seed_{seed}"])
                        matched_control_pairs_manifest.append(
                            {
                                "comparison_name": "__".join([treatment_run_slug, "vs", control_run_slug]),
                                "model_alias": subset["model_alias"],
                                "dataset": dataset_name,
                                "level": level,
                                "seed": seed,
                                "traits": spec["matched_controls"]["comparison_traits"],
                                "treatment_phase": "finetuned",
                                "control_phase": "matched_control",
                                "treatment_run_slug": treatment_run_slug,
                                "control_run_slug": control_run_slug,
                            }
                        )

    write_json(manifest_dir / "train_runs.json", train_runs)
    write_json(manifest_dir / "method_ablation_runs.json", method_ablation_runs)
    write_json(manifest_dir / "matched_control_runs.json", matched_control_runs)
    write_json(manifest_dir / "matched_control_pairs.json", matched_control_pairs_manifest)
    write_json(manifest_dir / "baseline_eval_configs.json", baseline_eval_configs)
    write_json(manifest_dir / "finetuned_eval_configs.json", finetuned_eval_configs)
    write_json(manifest_dir / "method_ablation_eval_configs.json", method_ablation_eval_configs)
    write_json(manifest_dir / "matched_control_eval_configs.json", matched_control_eval_configs)
    write_json(manifest_dir / "study_spec.expanded.json", spec)

    checkpoint_traits = sorted(
        set(
            spec.get("temporal_detector", {}).get("warning_traits", spec["traits"])
            + [spec.get("temporal_detector", {}).get("behavior_trait", "evil")]
        )
    )
    runbooks = {
        "00_prepare_experiment_data.sh": emit_prepare_script(repo_root, experiment_root, [dataset["name"] for dataset in spec["datasets"]], generated_root),
        "10_generate_trait_vectors.sh": emit_vector_script(repo_root, experiment_root, generated_root, spec["models"], spec["traits"], spec["judges"][0]["model_id"]),
        "20_eval_baselines.sh": emit_eval_loop_script(repo_root, experiment_root, f'"{generated_root}"/eval_configs/baseline/*.json'),
        "25_judge_calibration.sh": emit_calibration_script(repo_root, experiment_root, spec["judge_calibration"], generated_root),
        "30_train_models.sh": emit_train_script(repo_root, experiment_root, f'"{generated_root}"/train_configs/*.json'),
        "32_train_method_ablations.sh": emit_train_script(repo_root, experiment_root, f'"{generated_root}"/train_configs/method_ablations/*.json'),
        "35_train_matched_controls.sh": emit_train_script(repo_root, experiment_root, f'"{generated_root}"/train_configs/matched_controls/*.json'),
        "40_eval_finetuned_models.sh": emit_eval_loop_script(repo_root, experiment_root, f'"{generated_root}"/eval_configs/finetuned/*.json'),
        "41_eval_matched_controls.sh": emit_eval_loop_script(repo_root, experiment_root, f'"{generated_root}"/eval_configs/matched_controls/*.json'),
        "42_eval_method_ablations.sh": emit_eval_loop_script(repo_root, experiment_root, f'"{generated_root}"/eval_configs/method_ablations/*.json'),
        "44_eval_checkpoint_grid.sh": emit_checkpoint_grid_script(repo_root, generated_root),
        "45_eval_checkpoints_multijudge.sh": emit_checkpoint_eval_script(repo_root, experiment_root, generated_root, checkpoint_traits, spec["evaluation_defaults"]),
        "46_prepare_external_benchmarks.sh": emit_external_benchmark_prepare_script(repo_root, spec, generated_root, spec_path),
        "47_eval_external_benchmarks.sh": emit_eval_loop_script(repo_root, experiment_root, f'"{generated_root}"/eval_configs/external_benchmarks/*.json'),
        "48_eval_held_out_suite.sh": emit_held_out_suite_script(
            repo_root,
            "run_held_out_suite.sh",
        ),
        "49_eval_custom_held_out.sh": emit_held_out_suite_script(
            repo_root,
            "run_custom_held_out.sh",
        ),
        "50_projection_sweep.sh": emit_projection_script(repo_root, experiment_root, generated_root, finetuned_eval_configs, spec["traits"]),
        "55_detect_critical_points.sh": emit_critical_point_script(repo_root, spec, generated_root),
        **(
            {
                "56_evaluate_temporal_detector.sh": emit_temporal_detector_script(
                    repo_root,
                    spec,
                    generated_root,
                    spec_path,
                )
            }
            if "temporal_detector" in spec
            else {}
        ),
        "57_compare_warning_signals.sh": emit_compare_warning_script(repo_root, generated_root, spec["comparative_analysis"]),
        "58_compare_matched_controls.sh": emit_compare_matched_controls_script(repo_root, generated_root),
        "60_export_sae_activations.sh": emit_sae_export_script(repo_root, generated_root, train_runs, spec["datasets"], spec["sae_defaults"]),
        "65_train_sae.sh": emit_train_sae_script(repo_root, generated_root, train_runs, spec["sae_defaults"]),
        "70_intervention_early_stop.sh": emit_early_stop_grid_script(repo_root, generated_root),
        "70_intervention_early_stop_one.sh": emit_early_stop_script(repo_root, spec, generated_root),
        "71_intervention_filter_data.sh": emit_filter_data_script(repo_root, spec, generated_root),
        "72_intervention_sae_steer.sh": emit_sae_steer_script(repo_root, spec, generated_root),
        "80_score_sae_features.sh": emit_score_sae_script(repo_root, generated_root, train_runs, spec["sae_defaults"]),
        "81_intervene_hallucination_features.sh": emit_hallucination_feature_intervention_wrapper(repo_root, spec, generated_root),
        "82_eval_sae_intervention.sh": emit_eval_sae_intervention_script(repo_root, generated_root),
        **(
            {"90_rejudge_claim_validation.sh": emit_claim_validation_script(repo_root, generated_root)}
            if spec.get("validation_judges")
            else {}
        ),
        "91_prepare_human_validation.sh": emit_human_validation_prepare_script(
            repo_root,
            generated_root,
        ),
        "92_score_human_validation.sh": emit_human_validation_score_script(
            repo_root,
            generated_root,
        ),
    }

    for name, content in runbooks.items():
        path = runbook_dir / name
        path.write_text(content + "\n")
        path.chmod(0o755)


if __name__ == "__main__":
    main()
