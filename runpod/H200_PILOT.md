# H200 Pilot

Do not start the full training grid on the first reservation. Reserve one GPU
for four hours and use this pilot to validate the lab environment and obtain
real runtime measurements.

## Reservation Goal

The pilot performs:

1. GPU, disk, Python, package, dataset, and Hugging Face access checks
2. the required local-judge calibration quality gate
3. one real confirmatory run:
   `llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_0`

The training run is part of the final experiment rather than a disposable
benchmark. It is resumable if the three-hour training window expires.

## Persistent Paths

Replace `/PERSISTENT/em-study` with the lab's persistent high-capacity storage
path. Do not use a small home directory or node-local temporary disk.

```bash
export STUDY_STORAGE=/PERSISTENT/em-study
export VENV_DIR="$STUDY_STORAGE/venvs/emergent-misalignment"
export MAIN_VENV="$VENV_DIR"
export PIP_CACHE_DIR="$STUDY_STORAGE/cache/pip"
export HF_HOME="$STUDY_STORAGE/cache/huggingface"
export ROOT="$STUDY_STORAGE/emergent-misalignment-followup-study"
```

## First Reservation

```bash
mkdir -p "$STUDY_STORAGE"
cd "$STUDY_STORAGE"
git clone https://github.com/geunwoo-dev/emergent-misalignment-followup-study.git
cd "$ROOT"

read -rsp "Hugging Face token: " HF_TOKEN
export HF_TOKEN
echo

bash runpod/bootstrap.sh
tmux new -s h200-pilot
bash runpod/run_h200_pilot.sh
```

Detach from tmux with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t h200-pilot
```

If the lab scheduler terminates all processes when the reservation ends, tmux
does not extend the reservation. The latest Trainer checkpoint remains on
persistent storage and the pilot can resume during the next reservation.

## Return These Results

Send the pilot log under `logs/h200_pilot/` and the following outputs:

```bash
nvidia-smi
df -h "$STUDY_STORAGE"
cat logs/h200_pilot/preflight.json
find experiment/followup_study/generated_runpod/ckpt \
  -name .training_complete -print
```

Do not start stages `30`, `35`, or `44` until the calibration report and pilot
runtime have been reviewed.
