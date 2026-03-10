# Emergent Misalignment Follow-up Study

This repository packages the follow-up study scaffold built on top of the previous `persona` framework.

The follow-up is designed to address the main weaknesses from the previous paper:

- single-model dependence
- single-judge dependence
- no seed variance estimates
- weak critical-point robustness
- weak intervention evidence
- descriptive rather than causal mechanistic analysis

The target claim is stronger than "hallucination appears first". The current framing is:

`epistemic failure tends to emerge before overt social-alignment failure`

## Repository Layout

```text
.
├── followup_study/
│   ├── study_spec.json
│   ├── generate_assets.py
│   ├── multi_judge_eval.py
│   ├── aggregate_multi_judge.py
│   ├── evaluate_judge_calibration.py
│   ├── detect_critical_points.py
│   ├── export_activations.py
│   ├── train_sae.py
│   ├── intervene_sae_features.py
│   └── README.md
└── prev_paper_materials/
    └── persona/
        ├── training.py
        ├── generate_vec.py
        ├── eval/
        ├── data_generation/
        └── requirements.txt
```

`prev_paper_materials/persona` is the vendored base framework. `followup_study` is the orchestration and methodology layer for the new study.

## What Is Intentionally Not Committed

This repository excludes large or machine-specific artifacts:

- `prev_paper_materials/persona/dataset/`
- `prev_paper_materials/persona/dataset.zip`
- `followup_study/generated/`
- model checkpoints
- raw evaluation outputs
- SAE exports and SAE checkpoints

After cloning, you must provide the dataset locally and regenerate the configs/runbooks.

## Study Design

### Datasets

- `insecure_code`
- `mistake_gsm8k`
- `mistake_math`
- `mistake_medical`

### Traits

- `apathetic`
- `evil`
- `hallucinating`
- `humorous`
- `impolite`
- `optimistic`
- `sycophantic`

### Trait Taxonomy

- `epistemic_failure`: `hallucinating`
- `social_alignment_failure`: `evil`, `sycophantic`
- `affective_style_controls`: `apathetic`, `humorous`, `impolite`, `optimistic`

### Model Slots

- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-2-9b-it`
- `Qwen/Qwen2.5-7B-Instruct`

### Multi-Judge Setup

The default judge pair is intentionally provider-diverse:

- `gpt41mini`: OpenAI
- `gemma2_local`: local Hugging Face inference

This is stronger than using two OpenAI judges. If you want stricter judge independence for the final paper, replace the local judge with a held-out family in `followup_study/study_spec.json`.

### Seed Strategy

The seed budget is focused on the claim-critical subset:

- Llama medical `misaligned_1`
- Llama medical `misaligned_2`
- Qwen medical `misaligned_1`
- Qwen medical `misaligned_2`
- one math control condition

The full grid runs at one seed by default, and the robustness subset runs at multiple seeds.

### Evidence Ladder

Interpret the experiment stack in this order:

1. Descriptive trajectories
2. Early stopping at the hallucination critical point
3. SAE feature intervention

The third step is the strongest causal test in this repository.

## Prerequisites

- Python 3.11+ recommended
- CUDA-capable GPU for training and local-judge inference
- Hugging Face access for the base models
- OpenAI API access for the OpenAI judge

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r prev_paper_materials/persona/requirements.txt
```

Set the required credentials:

```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...
export WANDB_PROJECT=emergent-misalignment-followup
```

## Dataset Setup

The code expects the dataset directory at:

```bash
prev_paper_materials/persona/dataset/
```

Place the four required splits there, for example:

```text
prev_paper_materials/persona/dataset/insecure_code/
prev_paper_materials/persona/dataset/mistake_gsm8k/
prev_paper_materials/persona/dataset/mistake_math/
prev_paper_materials/persona/dataset/mistake_medical/
```

Each dataset directory should contain:

- `normal.jsonl`
- `misaligned_1.jsonl`
- `misaligned_2.jsonl`

## First Step After Clone

Always regenerate the configs and runbooks in the cloned location:

```bash
python3 followup_study/generate_assets.py
```

This creates `followup_study/generated/` with:

- `train_configs/`
- `eval_configs/`
- `judge_configs/`
- `manifests/`
- `runbooks/`

## Operator Checklist

Before starting a new run, confirm all of the following:

- the four datasets exist under `prev_paper_materials/persona/dataset/`
- `OPENAI_API_KEY` and `HF_TOKEN` are exported in the current shell
- `python3 followup_study/generate_assets.py` was rerun after any change to `study_spec.json`
- the target run is using the intended seed tier
- judge calibration has been run at least once after changing judge prompts or judge models
- raw model generations are being reused across judges rather than regenerated per judge

If any of these are false, stop and fix them before launching training or evaluation.

## Recommended End-to-End Workflow

1. Prepare the dataset directory

```bash
bash followup_study/generated/runbooks/00_prepare_persona_data.sh
```

2. Sanity-check the judges before the main experiment

```bash
bash followup_study/generated/runbooks/25_judge_calibration.sh
```

3. Generate persona vectors

```bash
bash followup_study/generated/runbooks/10_generate_persona_vectors.sh
```

4. Train the full grid and robustness subset

```bash
bash followup_study/generated/runbooks/30_train_models.sh
```

5. Run baseline and finetuned multi-judge evaluation

```bash
bash followup_study/generated/runbooks/20_eval_baselines.sh
bash followup_study/generated/runbooks/40_eval_finetuned_models.sh
```

6. Evaluate checkpoints and detect critical points

Example:

```bash
RUN_SLUG=llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
RUN_OUTPUT_DIR=followup_study/generated/ckpt/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
MODEL_ALIAS=llama_3_1_8b_instruct \
SEED=11 \
bash followup_study/generated/runbooks/45_eval_checkpoints_multijudge.sh
```

Then detect the hallucination critical point:

```bash
RUN_LABEL=llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
TRAIT=hallucinating \
bash followup_study/generated/runbooks/55_detect_critical_points.sh
```

7. Run early-stop intervention

```bash
RUN_SLUG=llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
RUN_OUTPUT_DIR=followup_study/generated/ckpt/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
MODEL_ALIAS=llama_3_1_8b_instruct \
SEED=11 \
TRAIT=hallucinating \
bash followup_study/generated/runbooks/70_intervention_early_stop.sh
```

8. Export activations and train SAEs

```bash
SAE_LAYER=20 bash followup_study/generated/runbooks/60_export_sae_activations.sh
SAE_LAYER=20 bash followup_study/generated/runbooks/65_train_sae.sh
SAE_LAYER=20 bash followup_study/generated/runbooks/80_score_sae_features.sh
```

9. Run SAE feature intervention

Example:

```bash
MODEL_PATH=followup_study/generated/ckpt/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
MODEL_ALIAS=llama_3_1_8b_instruct \
RUN_SLUG=llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
SAE_CHECKPOINT=followup_study/generated/sae_models/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11/layer_20/sae.pt \
FEATURE_CSV=followup_study/generated/sae_models/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11/layer_20/feature_shift.csv \
SAE_LAYER=20 \
SEED=11 \
bash followup_study/generated/runbooks/72_intervention_sae_steer.sh
```

Then compare intervention vs baseline:

```bash
RUN_SLUG=llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
BASELINE_DIR=followup_study/generated/agreement_reports/interventions/full_train/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
INTERVENTION_DIR=followup_study/generated/agreement_reports/interventions/sae_steer/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
bash followup_study/generated/runbooks/82_eval_sae_intervention.sh
```

## Key Outputs

- `followup_study/generated/agreement_reports/`
  Multi-judge summaries with mean score, judge std, generation std, CI, and pairwise agreement
- `followup_study/generated/critical_point_reports/`
  Critical points from multiple methods
- `followup_study/generated/intervention_reports/`
  Early-stop and SAE intervention comparisons
- `followup_study/generated/sae_models/`
  Trained SAE checkpoints and feature shift reports

## Expected Outputs By Stage

- Judge calibration:
  `followup_study/generated/agreement_reports/calibration/`
  Use this to verify that judges separate factual falsehood, incoherence, maliciousness, and safe refusal.
- Baseline evaluation:
  `followup_study/generated/agreement_reports/baselines/`
  This is the reference point for base-model behavior before finetuning.
- Finetuned evaluation:
  `followup_study/generated/agreement_reports/finetuned/`
  This is the main table for per-judge scores, mean scores, confidence intervals, and agreement.
- Checkpoint sweeps:
  `followup_study/generated/critical_point_reports/`
  Use this for temporal ordering claims and for selecting early-stop checkpoints.
- Early-stop interventions:
  `followup_study/generated/intervention_reports/early_stop/`
  Compare final harmful-trait scores against the matched full-training run.
- SAE analysis:
  `followup_study/generated/sae_models/`
  `followup_study/generated/agreement_reports/interventions/sae_steer/`
  Use these for the strongest causal test in the repository.

## Run Naming Convention

Most generated assets use a run slug of the form:

```text
{model_alias}__{dataset}__{split}__seed_{seed}
```

Example:

```text
llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11
```

Keep this slug unchanged when moving between training, checkpoint evaluation, critical-point detection, and intervention scripts. Most downstream scripts assume the same slug.

## Working Rules

- Do not edit files under `followup_study/generated/` by hand. Edit `study_spec.json` or `generate_assets.py`, then regenerate.
- Do not compare runs from different judge sets in the same figure or table without stating the judge change.
- For robustness claims, use the predefined multi-seed subset instead of mixing one-off seeds.
- Treat early stopping as intervention evidence, but not as the strongest mechanistic result. The strongest claim should come from SAE feature intervention.
- When changing the local judge model, rerun judge calibration before trusting any multi-judge agreement statistic.

## Common Failure Modes

- Missing datasets:
  The runbooks may generate configs correctly while training fails later. Check dataset presence first.
- Judge drift:
  If you edit prompts or swap local judge backends without recalibration, agreement numbers are not comparable.
- Stale generated assets:
  If the spec changed and `followup_study/generated/` was not rebuilt, runbooks and manifests may silently point to the wrong settings.
- Wrong seed interpretation:
  A single-seed full-grid result is not a robustness result. Use the robustness subset for any variance claim.
- Overstating causality:
  Feature scoring alone is descriptive. Use `intervene_sae_features.py` plus post-intervention evaluation for causal claims.

## Notes for the Team

- `followup_study/generated/` is not source-of-truth. `study_spec.json` and `generate_assets.py` are.
- If you change judges, seeds, or the robustness subset, rerun `python3 followup_study/generate_assets.py`.
- `gemma2_local` as a judge is a pragmatic default, not the final word on judge independence.
- `prev_paper_materials/persona/eval/eval_persona.py` was modified so the system can generate raw outputs once and then re-judge them with multiple judges.
