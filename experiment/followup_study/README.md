# Follow-up Study Scaffold

This directory contains the methodology layer for the follow-up paper. The executable framework code lives one level up in `experiment/`.

## Goals

The scaffold is designed to improve:

- evaluation reliability
- variance estimation
- critical-point robustness
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
- `intervene_sae_features.py`: converts shifted SAE features into a steering vector

## Key Design Choices

- Multi-judge evaluation with provider diversity
- Seed-aware planning focused on the claim-critical subset
- Multiple critical-point definitions
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

## Recommended Execution Order

1. `bash experiment/followup_study/generated/runbooks/00_prepare_experiment_data.sh`
2. `bash experiment/followup_study/generated/runbooks/10_generate_trait_vectors.sh`
3. `bash experiment/followup_study/generated/runbooks/25_judge_calibration.sh`
4. `bash experiment/followup_study/generated/runbooks/30_train_models.sh`
5. `bash experiment/followup_study/generated/runbooks/20_eval_baselines.sh`
6. `bash experiment/followup_study/generated/runbooks/40_eval_finetuned_models.sh`
7. `bash experiment/followup_study/generated/runbooks/45_eval_checkpoints_multijudge.sh`
8. `bash experiment/followup_study/generated/runbooks/55_detect_critical_points.sh`
9. `bash experiment/followup_study/generated/runbooks/70_intervention_early_stop.sh`
10. `bash experiment/followup_study/generated/runbooks/60_export_sae_activations.sh`
11. `bash experiment/followup_study/generated/runbooks/65_train_sae.sh`
12. `bash experiment/followup_study/generated/runbooks/80_score_sae_features.sh`
13. `bash experiment/followup_study/generated/runbooks/81_intervene_hallucination_features.sh`
14. `bash experiment/followup_study/generated/runbooks/82_eval_sae_intervention.sh`

On Colab, run the same sequence from `generated_colab/runbooks/` instead of `generated/runbooks/`.

## Notes

- `multi_judge_eval.py` depends on the modified `experiment/eval/eval_persona.py`, which can generate raw outputs once and then skip judging on subsequent passes.
- The change-point code includes a built-in mean-shift fallback, so it does not require an extra dependency.
- Gemma defaults to `google/gemma-2-9b-it` because the current runtime is text-only.
