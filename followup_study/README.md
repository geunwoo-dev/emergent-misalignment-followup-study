# Follow-up Study Scaffold

This directory contains the methodology upgrades for the follow-up paper. The goal is not just to scale the experiment matrix, but to strengthen:

- evaluation reliability
- variance estimation
- critical-point robustness
- intervention evidence
- mechanistic and causal validation

## What Changed Relative to the Previous Paper

- Multi-judge evaluation with provider diversity
- Seed-aware training and evaluation planning
- Multiple critical-point definitions
- Judge calibration before the main experiment grid
- Early-stop and data-filter intervention tracks
- SAE feature discovery plus SAE-based intervention

The intended high-level claim is:

`epistemic failure tends to emerge before overt social-alignment failure`

## Files to Start With

- `study_spec.json`: study scope, models, judges, seed tiers, critical-point methods, intervention defaults
- `generate_assets.py`: generates configs, manifests, and runbooks
- `multi_judge_eval.py`: raw generation once, then multi-judge scoring
- `aggregate_multi_judge.py`: mean score, judge std, generation std, confidence interval, agreement
- `evaluate_judge_calibration.py`: judge calibration on a small labeled set
- `detect_critical_points.py`: multiple critical-point methods
- `intervene_sae_features.py`: converts shifted SAE features into a steering vector

## Judge Design

The default judge pair is intentionally provider-diverse:

- `gpt41mini`: OpenAI
- `gemma2_local`: local Hugging Face inference

This is stronger than using two OpenAI judges. If you want stricter independence for the final paper, swap the local judge to a held-out family.

## Seed Design

The robustness budget is concentrated on the claim-critical subset:

- Llama medical `misaligned_1`
- Llama medical `misaligned_2`
- Qwen medical `misaligned_1`
- Qwen medical `misaligned_2`
- one math control condition

## Evidence Ladder

Interpret the intervention story in this order:

1. Descriptive trajectories
2. Early stopping at the hallucination critical point
3. SAE feature intervention

The third step is the strongest causal test in the current scaffold.

## Generate Assets

Run this after cloning the repo and after any change to `study_spec.json`:

```bash
python3 followup_study/generate_assets.py
```

This creates `followup_study/generated/` with:

- `train_configs/`
- `eval_configs/`
- `judge_configs/`
- `manifests/`
- `runbooks/`

## Recommended Execution Order

1. `bash followup_study/generated/runbooks/00_prepare_persona_data.sh`
2. `bash followup_study/generated/runbooks/10_generate_persona_vectors.sh`
3. `bash followup_study/generated/runbooks/25_judge_calibration.sh`
4. `bash followup_study/generated/runbooks/30_train_models.sh`
5. `bash followup_study/generated/runbooks/20_eval_baselines.sh`
6. `bash followup_study/generated/runbooks/40_eval_finetuned_models.sh`
7. `bash followup_study/generated/runbooks/45_eval_checkpoints_multijudge.sh`
8. `bash followup_study/generated/runbooks/55_detect_critical_points.sh`
9. `bash followup_study/generated/runbooks/70_intervention_early_stop.sh`
10. `bash followup_study/generated/runbooks/60_export_sae_activations.sh`
11. `bash followup_study/generated/runbooks/65_train_sae.sh`
12. `bash followup_study/generated/runbooks/80_score_sae_features.sh`
13. `bash followup_study/generated/runbooks/81_intervene_hallucination_features.sh`
14. `bash followup_study/generated/runbooks/82_eval_sae_intervention.sh`

## Notes

- `multi_judge_eval.py` depends on the modified `prev_paper_materials/persona/eval/eval_persona.py`, which can generate raw model outputs once and then skip judging on subsequent passes.
- The current critical-point code includes a built-in mean-shift fallback for the change-point method, so it does not require an extra package.
- Gemma defaults to `google/gemma-2-9b-it` because the current persona stack is text-only.
