# RunPod Launch Checklist

This checklist is the source of truth for the confirmatory three-model study.
Do not bypass a quality gate to keep GPUs busy.

## Before Renting GPUs

- [x] Confirmatory specification defines Llama 3.1 8B, Gemma 2 9B, and Qwen
  2.5 7B.
- [x] Core treatment data covers insecure code, GSM8K mistakes, MATH mistakes,
  and medical mistakes.
- [x] Treatment, matched-control, method-ablation, temporal-detector,
  intervention, held-out benchmark, and automated provider-diverse validation
  stages are generated.
- [x] Local judge inference is 4-bit and batched.
- [x] Checkpoint evaluation loads each response model once per checkpoint.
- [x] Three model-family workers use separate state, log, and held-out manifest
  files.
- [x] Runs resume from durable completion markers and Trainer checkpoints.
- [x] HaluEval, HarmBench, and LM Evaluation Harness revisions are pinned.
- [x] Unit tests, Python compilation, generated-asset validation, and shell
  syntax checks pass.
- [ ] Accept access terms for `meta-llama/Llama-3.1-8B-Instruct`,
  `google/gemma-2-9b-it`, and any gated benchmark classifier.
- [ ] Create a read-enabled Hugging Face token as a RunPod secret.
- [ ] Keep `OPENAI_API_KEY` and `OPENROUTER_API_KEY` out of the training
  environment; provide them only for the CommonsenseQA audit or gated
  validation stages.
- [ ] Decide whether completed Qwen adapters will be copied or linked from
  persistent storage.
- [ ] Choose a persistent backup destination.

API keys are not required for training, local judging, temporal analysis,
interventions, or held-out benchmarks. Add both validation keys only for the
CommonsenseQA data audit or stage `90`.

## Pod Requirements

- Three RTX 4090 GPUs on one pod, or three independent one-GPU pods
- At least 24 GB VRAM per worker
- Ubuntu 22.04 or newer with CUDA-compatible PyTorch
- At least 1 TB persistent storage mounted at `/workspace`
- Repository, virtual environments, Hugging Face cache, outputs, and logs under
  persistent storage

If using three independent pods, use shared persistent experiment storage or
consolidate all model-family outputs onto one coordinator before global stages
`55`, `56`, and `58`.

## One-Time Bootstrap

```bash
cd /workspace/emergent-misalignment-followup-study
chmod +x runpod/*.sh
bash runpod/bootstrap.sh
source /workspace/venvs/emergent-misalignment/bin/activate
python runpod/preflight.py --output logs/runpod/preflight.json
bash runpod/bootstrap_benchmarks.sh
```

Do not start training unless preflight reports `ready_for_training: true` and
asset verification reports `ok: true`.

## Generality Extension Gate

```bash
bash runpod/build_generality_extension.sh
```

The build command performs deterministic validation and a 100-row,
three-provider API audit. If it passes, activate:

```bash
bash runpod/activate_generality_extension.sh
python runpod/verify_assets.py
```

The treatment count must change from `51` to `57`. The six extension runs must
not run without a passing automated audit tied to the generated file hashes.

## Confirmatory Order

Run stage `25` once. For family-parallel stages, use
`bash runpod/run_three_workers.sh STAGE` on a three-GPU pod, or one
`run_family_worker.sh` command per single-GPU pod.

After a successful H200 pilot, a single shared-storage two- or three-H200 node
may run the complete Tier-1 sequence with:

```bash
bash runpod/start_h200_full_tmux.sh
```

The coordinator implements the order below, waits for all family workers,
checks disk space before each stage, and stops on any worker failure.

1. `00` dataset validation
2. `25` local-judge calibration gate
3. `30`, `35` treatment and matched-control training
4. `20`, `40`, `41` baseline and final evaluations
5. `44` dense checkpoint trajectories
6. `55`, `56`, `58` critical points, cross-fitted temporal detector, controls
7. `70` held-out early-stop intervention
8. `48`, `49` standard and custom held-out benchmarks
9. `32`, `42` method robustness after Tier 1 passes
10. `90` locked provider-diverse API rejudging of claim-critical rows only
11. `91`, `92` automated rubric robustness analysis and fail-closed gate

Stages `20`, `30`, `32`, `35`, `40`, `41`, `42`, `44`, `48`, `49`, and `70`
support family workers.

## Stop Conditions

- Stop if stage `25` misses its overall or per-dimension calibration threshold.
- Stop if disk free space approaches the reserved margin.
- Do not interpret pooled held-out metrics; retain separate benchmark results.
- Do not run stage `90` until the claim-validation manifest is frozen.
- Do not run CommonsenseQA treatment configs before the audit gate passes.

Monitor with:

```bash
python runpod/status.py
tmux list-sessions
nvidia-smi
df -h /workspace
```

Back up manually after each group of family workers, or enable automatic backup
on one coordinator only:

```bash
export BACKUP_DEST='REMOTE:path'
export AUTO_BACKUP_AFTER_STAGE=1
```
