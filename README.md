# Emergent Misalignment Follow-up Study

This repository now uses a single runtime folder:

- `experiment/`: all code and local assets needed to run the study
- `experiment/followup_study/`: study orchestration, configs, runbook generation, evaluation aggregation, critical-point analysis, and SAE intervention code

The old `prev_paper_materials/persona` nesting has been removed. The historical framework code now lives directly under `experiment/`.

## Repository Layout

```text
.
├── README.md
├── .gitignore
└── experiment/
    ├── training.py
    ├── generate_vec.py
    ├── judge.py
    ├── requirements.txt
    ├── dataset/
    ├── eval/
    ├── data_generation/
    ├── configs/
    ├── scripts/
    └── followup_study/
        ├── study_spec.json
        ├── generate_assets.py
        ├── multi_judge_eval.py
        ├── aggregate_multi_judge.py
        ├── evaluate_judge_calibration.py
        ├── detect_critical_points.py
        ├── export_activations.py
        ├── train_sae.py
        └── intervene_sae_features.py
```

## What Is Not Committed

Large or machine-specific artifacts are intentionally excluded:

- `experiment/dataset.zip`
- `experiment/followup_study/generated/`
- `experiment/followup_study/generated_colab/`
- model checkpoints
- raw generations and judge outputs
- SAE exports and SAE checkpoints

The four study datasets are committed in `experiment/dataset/`. Regenerate the study assets after cloning.

## Study Scope

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

The default pair is intentionally provider-diverse:

- `gpt41mini`: OpenAI
- `gemma2_local`: local Hugging Face inference

This is stronger than using two OpenAI judges. If you want stricter independence for the final paper, replace the local judge in `experiment/followup_study/study_spec.json`.

### Seed Strategy

The multi-seed budget is concentrated on the claim-critical subset:

- Llama medical `misaligned_1`
- Llama medical `misaligned_2`
- Qwen medical `misaligned_1`
- Qwen medical `misaligned_2`
- one math control condition

### Evidence Ladder

Interpret the study in this order:

1. Descriptive trajectories
2. Early stopping at the hallucination critical point
3. SAE feature intervention

The third step is the strongest causal test in this repository.

## Environment Setup

Create a virtual environment and install the runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r experiment/requirements.txt
```

Set credentials in the current shell:

```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...
export WANDB_PROJECT=emergent-misalignment-followup
```

## Colab Setup

For Colab, use the Colab-specific requirements and spec instead of the default full-spec path:

If you want checkpoints to survive runtime resets, mount Google Drive first and clone the repo inside Drive, for example under `/content/drive/MyDrive/`.

```bash
bash experiment/scripts/bootstrap_colab.sh
```

Then generate the Colab-oriented assets:

```bash
python3 experiment/followup_study/generate_assets.py \
  --spec_path experiment/followup_study/study_spec_colab.json
```

This writes configs and runbooks under:

```text
experiment/followup_study/generated_colab/
```

The Colab spec changes two things on purpose:

- it switches the judge pair to OpenAI-only so you do not need to host a second local judge model in Colab
- it uses more memory-friendly training defaults such as `load_in_4bit=true` and `per_device_train_batch_size=1`

## Dataset Setup

The repo now includes the four study datasets directly under:

```text
experiment/dataset/
```

Required structure:

```text
experiment/dataset/insecure_code/
experiment/dataset/mistake_gsm8k/
experiment/dataset/mistake_math/
experiment/dataset/mistake_medical/
```

Included files per dataset:

- `normal.jsonl`
- `misaligned_1.jsonl`
- `misaligned_2.jsonl`

## First Step After Clone

Always regenerate configs and runbooks in the cloned location:

```bash
python3 experiment/followup_study/generate_assets.py
```

This creates `experiment/followup_study/generated/` with:

- `train_configs/`
- `eval_configs/`
- `judge_configs/`
- `manifests/`
- `runbooks/`

If you are on Colab, use `study_spec_colab.json` instead and the outputs go to `generated_colab/`.

## Operator Checklist

Before launching a new run, confirm all of the following:

- the four datasets under `experiment/dataset/` are present
- `OPENAI_API_KEY` and `HF_TOKEN` are exported
- `python3 experiment/followup_study/generate_assets.py` was rerun after any spec change
- the target run is using the intended seed tier
- judge calibration has been rerun after any judge change
- raw generations will be reused across judges instead of regenerated per judge

## Recommended Workflow

1. Prepare the dataset directory

```bash
bash experiment/followup_study/generated/runbooks/00_prepare_experiment_data.sh
```

For Colab, replace `generated/` with `generated_colab/` in the runbook path.

2. Sanity-check the judges

```bash
bash experiment/followup_study/generated/runbooks/25_judge_calibration.sh
```

3. Generate trait vectors

```bash
bash experiment/followup_study/generated/runbooks/10_generate_trait_vectors.sh
```

4. Train the full grid and robustness subset

```bash
bash experiment/followup_study/generated/runbooks/30_train_models.sh
```

5. Run baseline and finetuned multi-judge evaluation

```bash
bash experiment/followup_study/generated/runbooks/20_eval_baselines.sh
bash experiment/followup_study/generated/runbooks/40_eval_finetuned_models.sh
```

6. Evaluate checkpoints and detect critical points

```bash
RUN_SLUG=llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
RUN_OUTPUT_DIR=experiment/followup_study/generated/ckpt/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
MODEL_ALIAS=llama_3_1_8b_instruct \
SEED=11 \
bash experiment/followup_study/generated/runbooks/45_eval_checkpoints_multijudge.sh
```

```bash
RUN_LABEL=llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
TRAIT=hallucinating \
bash experiment/followup_study/generated/runbooks/55_detect_critical_points.sh
```

7. Run the early-stop intervention

```bash
RUN_SLUG=llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
RUN_OUTPUT_DIR=experiment/followup_study/generated/ckpt/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
MODEL_ALIAS=llama_3_1_8b_instruct \
SEED=11 \
TRAIT=hallucinating \
bash experiment/followup_study/generated/runbooks/70_intervention_early_stop.sh
```

8. Export activations, train SAEs, and score shifted features

```bash
SAE_LAYER=20 bash experiment/followup_study/generated/runbooks/60_export_sae_activations.sh
SAE_LAYER=20 bash experiment/followup_study/generated/runbooks/65_train_sae.sh
SAE_LAYER=20 bash experiment/followup_study/generated/runbooks/80_score_sae_features.sh
```

9. Run SAE steering and compare against the matched full-training run

```bash
MODEL_PATH=experiment/followup_study/generated/ckpt/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
MODEL_ALIAS=llama_3_1_8b_instruct \
RUN_SLUG=llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
SAE_CHECKPOINT=experiment/followup_study/generated/sae_models/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11/layer_20/sae.pt \
FEATURE_CSV=experiment/followup_study/generated/sae_models/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11/layer_20/feature_shift.csv \
SAE_LAYER=20 \
bash experiment/followup_study/generated/runbooks/72_intervention_sae_steer.sh
```

```bash
RUN_SLUG=llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
BASELINE_DIR=experiment/followup_study/generated/agreement_reports/interventions/full_train/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
INTERVENTION_DIR=experiment/followup_study/generated/agreement_reports/interventions/sae_steer/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_11 \
bash experiment/followup_study/generated/runbooks/82_eval_sae_intervention.sh
```

## Expected Outputs

- `experiment/followup_study/generated/agreement_reports/`
  Multi-judge summaries with mean score, judge std, generation std, CI, and pairwise agreement
- `experiment/followup_study/generated/critical_point_reports/`
  Critical points from multiple methods
- `experiment/followup_study/generated/intervention_reports/`
  Early-stop and SAE intervention comparisons
- `experiment/followup_study/generated/sae_models/`
  Trained SAE checkpoints and feature shift reports

## Working Rules

- Do not edit files under `experiment/followup_study/generated/` by hand. Edit `experiment/followup_study/study_spec.json` or `experiment/followup_study/generate_assets.py`, then regenerate.
- Do not mix judge sets within the same comparison figure without explicitly documenting the judge change.
- Use the predefined robustness subset for variance claims.
- Treat early stopping as intervention evidence, not as the strongest mechanistic result.
- Re-run judge calibration after changing the local judge backend.

## Common Failure Modes

- Missing datasets:
  Config generation may succeed while training fails later. Check `experiment/dataset/` first.
- Judge drift:
  If you change judge prompts or judge backends without recalibration, agreement numbers are not comparable.
- Stale generated assets:
  If the spec changed and `experiment/followup_study/generated/` was not rebuilt, runbooks can silently point to the wrong settings.
- Wrong seed interpretation:
  A full-grid single-seed result is not a robustness result.
- Overstating causality:
  Feature scoring is descriptive. Use SAE intervention plus post-intervention evaluation for causal claims.

## Notes

- `experiment/followup_study/generated/` is not source-of-truth. `experiment/followup_study/study_spec.json` and `experiment/followup_study/generate_assets.py` are.
- `experiment/eval/eval_persona.py` was modified so the system can generate raw outputs once and then re-judge them with multiple judges.
- `experiment/framework_reference.md` keeps the original framework notes, but the active runtime layout is now `experiment/`.
