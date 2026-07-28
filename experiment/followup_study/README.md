# Follow-up Study Scaffold

This directory contains the methodology layer for the follow-up paper. The executable framework code lives one level up in `experiment/`.

## Goals

The scaffold is designed to improve:

- evaluation reliability
- variance estimation
- critical-point robustness
- external validity against saved-checkpoint organisms
- confound control against narrow-finetuning traces
- intervention evidence
- mechanistic and causal validation

The intended claim is:

`epistemic failure tends to emerge before overt social-alignment failure`

## Start Here

- `study_spec.json`: study scope, models, judges, seeds, critical-point methods, intervention defaults
- `generate_assets.py`: generates configs, manifests, and runbooks
- `multi_judge_eval.py`: raw generation once, then multi-judge scoring
- `aggregate_multi_judge.py`: judge mean, judge std, generation std, CI, and agreement
- `evaluate_judge_calibration.py`: calibration before the main experiment
- `detect_critical_points.py`: multiple critical-point methods
- `build_temporal_detector_dataset.py`: leakage-controlled checkpoint examples
- `evaluate_temporal_detector.py`: held-out model/domain/seed detector evaluation
- `select_crossfit_alarm.py`: selects the first sequential held-out detector alarm
- `compare_warning_signals.py`: lead-time and threshold-based comparison across black-box and internal signals
- `prepare_external_benchmark.py`: turns saved-checkpoint benchmark specs into eval configs
- `prepare_held_out_manifest.py`: locks the final-only held-out model subset
- `run_lm_eval_suite.py`: pinned standard TruthfulQA and capability evaluation
- `run_custom_held_out.py`: official-data HaluEval and HarmBench evaluation
- `evaluate_matched_control_deltas.py`: compares misaligned runs against matched normal-training controls
- `intervene_sae_features.py`: converts shifted SAE features into a steering vector
- `prepare_human_validation.py`: creates blinded annotation sheets
- `evaluate_human_validation.py`: reports inter-rater reliability

## Key Design Choices

- Multi-judge evaluation with provider diversity
- Seed-aware planning focused on the claim-critical subset
- Multiple critical-point definitions
- Matched-control runs for the claim-critical subset
- External-checkpoint benchmark support for retrospective evaluation
- A prespecified held-out benchmark suite for factuality, hallucination, harm,
  and capability-retention checks
- A gated six-run CommonsenseQA-derived non-mathematical generality extension
- Judge calibration before the main grid
- Early-stop and data-filter intervention tracks
- SAE feature discovery plus SAE-based intervention

## Generate Assets

Run this after cloning and after any change to `study_spec.json`:

```bash
python3 experiment/followup_study/generate_assets.py
```

This creates `experiment/followup_study/generated/` with:

- `train_configs/`
- `eval_configs/`
- `judge_configs/`
- `manifests/`
- `runbooks/`

For Colab, generate a separate asset tree:

```bash
python3 experiment/followup_study/generate_assets.py \
  --spec_path experiment/followup_study/study_spec_colab.json
```

That writes the Colab-specific runbooks under `experiment/followup_study/generated_colab/`.

For the confirmatory RunPod protocol:

```bash
python3 experiment/followup_study/generate_assets.py \
  --spec_path experiment/followup_study/study_spec_runpod.json
```

See `EXPERIMENT_PROTOCOL.md` and `runpod/README.md` before launching compute.

## Recommended Execution Order

1. `bash experiment/followup_study/generated/runbooks/00_prepare_experiment_data.sh`
2. `bash experiment/followup_study/generated/runbooks/10_generate_trait_vectors.sh`
3. `bash experiment/followup_study/generated/runbooks/25_judge_calibration.sh`
4. `bash experiment/followup_study/generated/runbooks/30_train_models.sh`
5. `bash experiment/followup_study/generated/runbooks/35_train_matched_controls.sh`
6. `bash experiment/followup_study/generated/runbooks/20_eval_baselines.sh`
7. `bash experiment/followup_study/generated/runbooks/40_eval_finetuned_models.sh`
8. `bash experiment/followup_study/generated/runbooks/41_eval_matched_controls.sh`
9. `bash experiment/followup_study/generated/runbooks/45_eval_checkpoints_multijudge.sh`
10. `bash experiment/followup_study/generated/runbooks/55_detect_critical_points.sh`
11. `bash experiment/followup_study/generated/runbooks/57_compare_warning_signals.sh`
12. `bash experiment/followup_study/generated/runbooks/58_compare_matched_controls.sh`
13. `bash experiment/followup_study/generated/runbooks/46_prepare_external_benchmarks.sh`
14. `bash experiment/followup_study/generated/runbooks/47_eval_external_benchmarks.sh`
15. `bash experiment/followup_study/generated/runbooks/70_intervention_early_stop.sh`
16. `bash experiment/followup_study/generated/runbooks/60_export_sae_activations.sh`
17. `bash experiment/followup_study/generated/runbooks/65_train_sae.sh`
18. `bash experiment/followup_study/generated/runbooks/80_score_sae_features.sh`
19. `bash experiment/followup_study/generated/runbooks/81_intervene_hallucination_features.sh`
20. `bash experiment/followup_study/generated/runbooks/82_eval_sae_intervention.sh`

The RunPod confirmatory tree adds `44_eval_checkpoint_grid.sh`,
`48_eval_held_out_suite.sh`, `49_eval_custom_held_out.sh`,
`56_evaluate_temporal_detector.sh`, `90_rejudge_claim_validation.sh`,
`91_prepare_human_validation.sh`, and `92_score_human_validation.sh`.

Run `32_train_method_ablations.sh` and `42_eval_method_ablations.sh` only after
Tier 1 passes its quality gates.

On Colab, run the same sequence from `generated_colab/runbooks/` instead of `generated/runbooks/`.

## Notes

- `multi_judge_eval.py` depends on the modified `experiment/eval/eval_persona.py`, which can generate raw outputs once and then skip judging on subsequent passes.
- `30_train_models.sh` and `35_train_matched_controls.sh` do not need OpenAI credentials. `HF_TOKEN` is only needed if the chosen Hugging Face model is gated.
- `46_prepare_external_benchmarks.sh` reads `external_benchmark_template.json`; benchmark CSVs should expose `checkpoint_step` and `score`.
- `held_out_benchmark_suite` in `study_spec_runpod.json` is distinct from the
  saved-checkpoint organism support. It is final-checkpoint only; stages `48`
  and `49` implement the standard and official custom benchmark paths.
- `generality_extension` is intentionally not included in the generated
  training grid until its JSONL data passes the stated quality gate.
- The change-point code includes a built-in mean-shift fallback, so it does not require an extra dependency.
- Gemma defaults to `google/gemma-2-9b-it` because the current runtime is text-only.
