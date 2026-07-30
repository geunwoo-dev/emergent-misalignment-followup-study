# Three-H200 Full Run

This workflow runs the complete compute-heavy Tier-1 confirmatory pipeline on
one node exposing three H200 GPUs. One model family is assigned to each GPU.

The automated sequence is:

`00 -> 25 -> 30 -> 35 -> 20 -> 40 -> 41 -> 44 -> 55 -> 56 -> 58 -> 70 -> 48 -> 49`

Family stages run Llama, Gemma, and Qwen in parallel. Global analysis stages
start only after all three family workers finish successfully.
The local-judge calibration is deliberately rerun even if the pilot left a
completion marker, ensuring that the checked-out evaluator prompts still pass.

Stages `90-92` are intentionally not part of the compute-heavy core. Stage `90`
requires a frozen claim-validation manifest plus OpenAI and OpenRouter keys.
Stages `91-92` automatically analyze provider/rubric robustness and enforce the
claim gate. Method-ablation stages `32/42` remain behind the Tier-1 review gate
defined in the experiment protocol.

No OpenAI key is needed for this automated core pipeline.

## Before the Reservation

Use the same persistent repository, virtual environments, Hugging Face cache,
and benchmark storage used by the successful pilot. Do not run from node-local
temporary storage.

The successful pilot used:

- NVIDIA H200 with 143,771 MiB VRAM
- about 24 TiB free persistent storage
- local judge calibration accuracy of 93.3%
- 2,217 seconds for the representative Llama training run

Reserve three H200 GPUs for up to 96 hours for the first full attempt. The
current estimate is 12-18 wall-clock hours for training and roughly 2-4 days
for the complete Tier-1 compute pipeline. If the reservation ends first,
reserve the same persistent storage again and rerun the launcher to resume.

## Launch

Replace the paths below only if the lab allocation mounts persistent storage
elsewhere.

```bash
cd /root/work/jonghwi/emergent-misalignment-followup-study
git pull --ff-only origin main

export ROOT="$PWD"
export STUDY_STORAGE=/mnt/ddn/prod-runs/interns/jonghwi
export MAIN_VENV=/root/work/jonghwi/em-study/venvs/emergent-misalignment
export BENCH_VENV=/root/work/jonghwi/em-study/venvs/emergent-benchmarks
export PIP_CACHE_DIR=/root/work/jonghwi/em-study/cache/pip
export HF_HOME=/root/work/jonghwi/em-study/cache/huggingface
export BENCHMARK_ROOT=/root/work/jonghwi/em-study/benchmarks

read -rsp "Hugging Face token: " HF_TOKEN
export HF_TOKEN
echo

bash runpod/start_h200_full_tmux.sh
```

The launcher returns immediately. The work continues inside the `h200-full`
tmux session.

## Monitoring

```bash
tmux attach -t h200-full
python runpod/status.py
cat logs/h200_full/latest_status.txt
nvidia-smi
df -h "$ROOT"
```

Detach from tmux with `Ctrl-b`, then `d`.

## Recovery

If a worker or reservation stops, inspect:

```bash
cat logs/h200_full/latest_status.txt
python runpod/status.py
tail -n 100 logs/h200_full/full_*.log
```

After fixing the reported error, run the same launcher again:

```bash
bash runpod/start_h200_full_tmux.sh
```

Remove a dead tmux session first if necessary:

```bash
tmux kill-session -t h200-full
bash runpod/start_h200_full_tmux.sh
```

Completed stages are skipped using durable stage markers. Completed training
runs are skipped using `.training_complete`, and incomplete Trainer
checkpoints resume automatically.

## Optional Backup

Persistent lab storage is required even when remote backup is enabled. To back
up once after every completed stage:

```bash
export BACKUP_DEST='gdrive:emergent-study/h200'
export BACKUP_AFTER_EACH_STAGE=1
```

To back up only after the full core pipeline completes:

```bash
export BACKUP_DEST='gdrive:emergent-study/h200'
export AUTO_BACKUP_AFTER_FULL=1
```

Configure these variables before starting the tmux session.

## Completion

Successful unattended completion produces:

```text
logs/h200_full/latest_status.txt
phase=complete_core_pipeline
stage=complete
exit_code=0
```

At that point, send `logs/h200_full/`, `runpod/status.py` output, detector
reports, matched-control reports, intervention reports, and held-out summaries
to the experiment owner for the Tier-1 review.

## Post-Core API Validation

The API stage starts only after the owner freezes the exact reference and
treatment items, expected effect directions, and minimum effects in a copy of
`experiment/followup_study/claim_validation_manifest.template.json`. Do not
edit the manifest or its source CSVs after setting `locked_at_utc`.

```bash
cd /root/work/jonghwi/emergent-misalignment-followup-study
source "$MAIN_VENV/bin/activate"

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

Stage `90` stores deterministic capped inputs and hashes under
`generated_runpod/claim_validation/` and requires all three providers to pass a
cached 90-call calibration smoke gate. Stage `91` evaluates provider and prompt
robustness, and stage `92` fails closed if any prespecified claim does not pass.
No new human annotation is collected.
