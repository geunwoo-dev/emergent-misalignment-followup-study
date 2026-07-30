# RunPod Execution

Use `LAUNCH_CHECKLIST.md` as the final go/no-go checklist.
For the first reservation on a lab H200 cluster, follow `H200_PILOT.md` and do
not launch the full grid.

## Required Pod Configuration

- Three NVIDIA RTX 4090 workers are recommended; one worker remains supported
- At least 24 GB VRAM
- Ubuntu 22.04 or newer
- CUDA-compatible PyTorch image
- At least 1 TB persistent network storage for the high-resolution confirmatory
  checkpoint grid
- Repository, Hugging Face cache, checkpoints, logs, and virtual environment on
  persistent storage

Do not place experiment outputs only on ephemeral container storage.

## Information Needed Before Launch

Provide:

- GPU model and count
- RunPod image name
- persistent-volume mount path and capacity
- repository source or archive location
- location of completed Qwen checkpoints
- backup destination
- whether gated Llama and Gemma repositories are already accepted
- whether the API-validation stage should run on this pod

Never send API or Hugging Face token values in chat. Set them directly in
RunPod secrets or the pod shell.

## Credentials

Before preflight, accept the Hugging Face access terms for:

- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-2-9b-it`
- `cais/HarmBench-Llama-2-13b-cls`

Create a read-enabled Hugging Face token and enter it directly on the worker:

```bash
export HF_HOME=/workspace/.cache/huggingface
read -rsp "Hugging Face token: " HF_TOKEN
export HF_TOKEN
echo
```

The preflight script verifies access to every required Hugging Face model
without printing the token. Credential requirements are:

| Credential | Requirement |
| --- | --- |
| `HF_TOKEN` | Required for gated model downloads and preflight |
| `OPENAI_API_KEY` | Required for the CommonsenseQA audit and locked claim-validation stage `90` |
| `OPENROUTER_API_KEY` | Required for the CommonsenseQA audit and stage `90` |
| GitHub token | Not required; this repository is public |
| W&B key | Not used |
| Google Drive or `rclone` OAuth | Optional, for remote backup only |

Do not commit tokens to `.env`, configuration files, scripts, or experiment
outputs. On a shared lab server, set them only for the active shell or use the
cluster's secret manager.

## Bootstrap

```bash
cd /workspace/emergent-misalignment-followup-study
chmod +x runpod/*.sh
bash runpod/bootstrap.sh
source /workspace/venvs/emergent-misalignment/bin/activate
python runpod/preflight.py --output logs/runpod/preflight.json
```

The preflight command must end with `"ready_for_training": true`, and
`python runpod/verify_assets.py` must report `"ok": true`.

For held-out benchmarks, install the isolated harness environment and pinned
official benchmark data once on persistent storage:

```bash
bash runpod/bootstrap_benchmarks.sh
```

## CommonsenseQA Generality Extension

Build the six-run Tier-2 extension before training:

```bash
read -rsp "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
echo
read -rsp "OpenRouter API key: " OPENROUTER_API_KEY
export OPENROUTER_API_KEY
echo

bash runpod/build_generality_extension.sh
unset OPENAI_API_KEY OPENROUTER_API_KEY
```

This runs deterministic checks plus a 100-row audit with the three validation
provider groups, for 300 short API calls total. It fails closed if any provider
misses the quality threshold.
After it passes, run:

```bash
bash runpod/activate_generality_extension.sh
python runpod/verify_assets.py
```

Verification must show `57` treatment runs after activation. Before activation
it must show `51`; training the six extension runs without a passing automated
audit is prohibited.

## Import Completed Qwen Runs

After the old checkpoint location is known, completed runs can be linked without
duplicating storage:

```bash
python experiment/followup_study/import_existing_checkpoints.py \
  --source_root /workspace/old/generated_colab/ckpt \
  --destination_root experiment/followup_study/generated_runpod/ckpt \
  --run_manifests \
    experiment/followup_study/generated_runpod/manifests/train_runs.json \
    experiment/followup_study/generated_runpod/manifests/matched_control_runs.json \
  --mode symlink \
  --mark_complete
```

Use `--mark_complete` only for runs whose training log shows successful
completion. Use `--mode copy` when source storage may be detached.

## Durable tmux Execution

```bash
tmux new -s em-study
source /workspace/venvs/emergent-misalignment/bin/activate
cd /workspace/emergent-misalignment-followup-study
bash runpod/run_stage.sh 25
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t em-study
```

Monitor all workers with:

```bash
python runpod/status.py
tmux list-sessions
```

## Three-GPU Execution

For one pod exposing three GPUs, start one model-family worker per GPU:

```bash
bash runpod/run_three_workers.sh 30
```

For a validated three-H200 lab node, use the unattended full-pipeline
coordinator instead of launching every stage manually:

```bash
bash runpod/start_h200_full_tmux.sh
```

See [`H200_FULL.md`](H200_FULL.md) for persistent paths, monitoring, recovery,
and the exact automated stage sequence. The coordinator waits for all three
model-family workers before advancing to each dependent global stage.

For three separate single-GPU pods, run one command on each pod:

```bash
# Pod 1
bash runpod/run_family_worker.sh 30 llama_3_1_8b_instruct 0

# Pod 2
bash runpod/run_family_worker.sh 30 gemma_2_9b_it 0

# Pod 3
bash runpod/run_family_worker.sh 30 qwen_2_5_7b_instruct 0
```

Separate pods must either mount the same persistent experiment storage or
merge each family's outputs into one coordinator before global stages `55`,
`56`, and `58`. Do not start those stages from a partial per-pod copy.

Use the same pattern for stages `20`, `30`, `32`, `35`, `40`, `41`, `42`,
`44`, `48`, `49`, and `70`. Workers use independent locks and logs. Do not run
unfiltered `run_stage.sh` for the same stage concurrently.

To enforce an opt-in hard compute budget per worker:

```bash
export RUNPOD_PRICE_PER_GPU_HOUR=0.55
export RUNPOD_BUDGET_USD=100
bash runpod/run_family_worker.sh 30 llama_3_1_8b_instruct 0
```

The runner terminates that worker when its derived budget window expires.

## Confirmatory Stage Order

1. `00`: dataset validation
2. `25`: judge calibration quality gate
3. `30`: treatment training
4. `35`: matched-control training
5. `20`: baseline evaluation
6. `40`: final-checkpoint evaluation
7. `41`: matched-control evaluation
8. `44`: all saved-checkpoint trajectories
9. `55`: detect candidate critical points
10. `56`: held-out temporal detector
11. `58`: matched-control comparison
12. `70`: held-out detector-alarm early-stop evaluation
13. `48`: final-only TruthfulQA, MedQA, GSM8K, and MBPP
14. `49`: final-only HaluEval and HarmBench
15. `90`: locked subset with OpenAI, Google, and Anthropic judge families
16. `91`: analyze provider and rubric robustness
17. `92`: fail-closed automatic claim gate

Do not begin Tier 2 or SAE work until detector results and judge calibration
pass review.

Tier-2 adaptation-method robustness uses stages `32` and `42`. The held-out
suite selects only base models, seed-0 `misaligned_2` models, and one
representative matched control per available model/domain pair.

Stages through `70` and local held-out screening need no API keys. Set
`OPENAI_API_KEY` and `OPENROUTER_API_KEY` only for the CommonsenseQA data audit
or immediately before stage `90`; stages `91` and `92` analyze saved API
outputs offline.

## Locked API Validation

After the core pipeline, copy
`experiment/followup_study/claim_validation_manifest.template.json` to a
result-specific path. Replace every placeholder, define each prespecified
comparison in `claims`, and set `locked_at_utc` before making any validation
API call. Each item is one stratum; stage `90` deterministically caps it at
`maximum_rows_per_stratum`, records source and selection hashes, and refuses
to reuse a selection if its source changes.

Enter credentials without placing them in shell history, then run:

```bash
read -rsp "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
echo
read -rsp "OpenRouter API key: " OPENROUTER_API_KEY
export OPENROUTER_API_KEY
echo

export CLAIM_VALIDATION_MANIFEST="$PWD/path/to/locked_claim_manifest.json"
bash runpod/run_stage.sh 90
bash runpod/run_stage.sh 91
bash runpod/run_stage.sh 92

unset OPENAI_API_KEY OPENROUTER_API_KEY
```

Stage `90` first runs a cached 90-call calibration smoke gate, then reports the
number of newly requested claim score calls. It evaluates two rubrics per row
(trait and coherence) for three provider groups and three prompt variants.
Stage `92` exits nonzero unless every locked claim passes the prespecified
provider-diversity, parse-rate, direction, rubric-robustness, and
confidence-interval gates. API consensus is evaluator robustness evidence, not
human preference data.

## Backup

```bash
export BACKUP_DEST='gdrive:emergent-study/runpod'
bash runpod/backup.sh
```

To run the incremental backup automatically after a successful stage, also set
`AUTO_BACKUP_AFTER_STAGE=1`. Enable this on only one coordinator when workers
share storage; otherwise run `backup.sh` manually after all family workers
finish.

The stage runner is idempotent at the stage level. Training runbooks additionally
skip runs with `.training_complete` and resume incomplete Trainer checkpoints.
Checkpoint evaluation loads each response checkpoint once for all primary
traits and batches local-judge inference. Completed per-model benchmark
directories are skipped on restart.
