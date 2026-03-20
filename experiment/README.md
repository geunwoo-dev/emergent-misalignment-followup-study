# Experiment Runtime

This directory is the active runtime root for the follow-up study.

## Contents

- `training.py`, `generate_vec.py`, `judge.py`, `eval/`, `data_generation/`
  Core execution code inherited from the earlier framework
- `dataset/`
  Local datasets for the four study domains
- `followup_study/`
  Study orchestration, runbook generation, multi-judge aggregation, critical-point analysis, and SAE intervention code

## Start

From the repository root:

```bash
python3 experiment/followup_study/generate_assets.py
```

For Colab:

```bash
bash experiment/scripts/bootstrap_colab.sh
python3 experiment/followup_study/generate_assets.py --spec_path experiment/followup_study/study_spec_colab.json
```

Then use the generated runbooks under:

```text
experiment/followup_study/generated/runbooks/
```

The Colab-specific runbooks are generated under:

```text
experiment/followup_study/generated_colab/runbooks/
```

Not every step needs OpenAI credentials. Training-only paths such as `30_train_models.sh` and `35_train_matched_controls.sh` do not need judge access. `HF_TOKEN` is only needed if the selected Hugging Face model is gated. Judge calibration, vector extraction, multi-judge evaluation, and warning-signal comparison still require judge access.

`framework_reference.md` preserves the older framework notes, but new work should treat `experiment/` as the only runtime root.
