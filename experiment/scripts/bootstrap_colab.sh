#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip
pip install -r experiment/requirements_colab.txt

cat <<'EOF'
Colab bootstrap complete.

Next steps:
  1. export HF_TOKEN=...
  2. export OPENAI_API_KEY=...   # needed for judge-based evaluation
  3. python3 experiment/followup_study/generate_assets.py --spec_path experiment/followup_study/study_spec_colab.json
EOF
