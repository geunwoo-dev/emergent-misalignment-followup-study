# Included Datasets

This GPU repository ships with the four task datasets used by the follow-up study:

- `insecure_code`
- `mistake_gsm8k`
- `mistake_math`
- `mistake_medical`

Each dataset directory contains:

- `normal.jsonl`
- `misaligned_1.jsonl`
- `misaligned_2.jsonl`

These files were copied from the earlier local framework under `prev_paper_materials/persona/dataset/` so the GPU repo is runnable without a separate dataset handoff.

Legacy directories from the earlier project such as `evil`, `hallucination`, `mistake_opinions`, and `sycophancy` are intentionally not mirrored here because the follow-up scaffold does not train on them.
