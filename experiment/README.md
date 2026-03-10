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

Then use the generated runbooks under:

```text
experiment/followup_study/generated/runbooks/
```

`framework_reference.md` preserves the older framework notes, but new work should treat `experiment/` as the only runtime root.
