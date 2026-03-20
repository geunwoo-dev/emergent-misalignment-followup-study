#!/usr/bin/env bash
set -euo pipefail

python3 -m pip uninstall -y wandb || true
pip install --upgrade pip
pip install -r experiment/requirements_colab.txt
python3 -m pip uninstall -y wandb || true

cat <<'EOF'
Colab bootstrap complete.

Next steps:
  1. Restart the runtime once if this notebook previously imported unsloth or trl.
  2. export HF_TOKEN=...         # only needed for gated Hugging Face models
  3. export OPENAI_API_KEY=...   # needed for judge-based evaluation
  4. python3 experiment/followup_study/generate_assets.py --spec_path experiment/followup_study/study_spec_colab.json
EOF
