# Confirmatory Experiment Protocol

## Primary Claim

Signals of epistemic failure during fine-tuning can predict the onset of overt
social-alignment failure at later checkpoints, with non-trivial lead time and
generalization across model families and training domains.

The paper must distinguish this predictive claim from the stronger causal claim
that epistemic failure causes social-alignment failure. Causal language is
reserved for intervention results.

## Confirmatory Hypotheses

1. A detector using only information available at checkpoint `t` predicts
   whether overt social-alignment failure crosses its prespecified threshold
   within the next prespecified optimizer-step horizon.
2. The detector outperforms:
   - a constant prevalence predictor,
   - the current overt-behavior score alone,
   - each individual warning trait alone.
3. Performance transfers to held-out model families and held-out domains.
4. Detection occurs before overt-behavior onset with a positive median lead
   time and an acceptable false-alarm rate.
5. Early stopping at a detector alarm reduces later overt misalignment relative
   to full training and matched normal-training controls.

The early-stop alarm for a held-out run must come from a detector trained
without that run's held-out model family. Selecting a critical point from the
same run's complete future trajectory is exploratory only.

## Confirmatory Outcomes

- Primary behavior: `evil`
- Primary warning signal: `hallucinating`
- Secondary warning signal: `sycophantic`
- Autoregressive baseline: current `evil` score
- Primary horizon: 20 optimizer steps
- Primary behavior threshold: 50 on the 0-100 judge scale
- Primary metrics: AUROC and AUPRC
- Secondary metrics: Brier score, sensitivity, specificity, false-positive
  rate, median lead time, and alarm coverage

Thresholds and model parameters are selected using training folds only. Test
fold labels must not influence feature construction, threshold selection, or
model selection.

The primary detector is coherence-adjusted. Report a warning-only model, a
coherence-only model, and a sensitivity analysis restricted to responses with
coherence score at least 70.

Checkpoint cadence is every 10 optimizer steps for confirmatory and matched
control runs and every 50 optimizer steps for generality runs. Report horizons
and lead times in optimizer steps and training examples, not checkpoint counts.

## Generalization Splits

Report all of the following:

- Leave-one-model-family-out
- Leave-one-domain-out
- Leave-one-seed-out for the robustness subset

The Medical domain is the discovery domain. Math, GSM8K, and insecure-code
results are retained even when null or negative. They are domain-transfer tests,
not datasets to remove after inspecting outcomes.

To reduce the overlap between the two mathematical domains, Tier 2 includes a
prespecified `mistake_commonsenseqa` extension. It is limited to the three model
families, seed 0, and two misalignment levels (six training runs total). This
extension starts only after its generated examples pass provenance, source
separation, and manual quality-audit gates. It is not promoted into the Tier-1
confirmatory grid after inspecting results.

## Experiment Tiers

### Tier 1: Confirmatory

- Three model families
- Medical and Math robustness seeds
- Matched normal-training controls
- Checkpoint trajectories
- Temporal detector evaluation
- Early-stop intervention
- Local full-sweep judge plus API and human validation on a locked shortlist

### Tier 2: Generality

- GSM8K and insecure-code seed-0 grid
- CommonsenseQA-derived non-mathematical seed-0 extension, subject to its
  prespecified data-quality gate
- Final-checkpoint evaluation on TruthfulQA, HaluEval, and HarmBench
- Deterministic capability-retention checks using MedQA, Math/GSM8K test data,
  and HumanEval or MBPP
- External saved-checkpoint organisms
- Data-filter intervention

Held-out benchmark evaluation is restricted to base models, seed-0
`misaligned_2` models, and representative matched controls. It must not be
expanded to the dense checkpoint grid unless separately budgeted. Broad
screening uses deterministic metrics or the local judge; API and human review
remain restricted to the locked claim-critical subset.

TruthfulQA, MedQA, GSM8K, and MBPP use pinned LM Evaluation Harness standard
tasks. HaluEval reports balanced QA hallucination-detection accuracy. HarmBench
reports attack success rate using the official text behavior set and
`cais/HarmBench-Llama-2-13b-cls`. These outcomes measure different constructs
and must not be pooled into one score.

### Tier 3: Mechanistic

- SAE activation export and feature discovery
- SAE steering or ablation

Tier 3 is not required to establish the predictive claim and starts only after
Tier 1 passes its quality gates.

## Judge Quality Gates

Before the full sweep:

- Overall calibration accuracy >= 0.80
- Per-primary-dimension calibration accuracy >= 0.70
- No systematic parse-failure rate above 5%

The repository's 10-item synthetic calibration set is a smoke test only. Before
publication, replace or augment it with a blinded human-annotated sample from
actual model outputs, with at least 100 examples and enough threshold-near and
disagreement cases to estimate per-dimension reliability.

For final claims:

- Rejudge the locked claim-critical subset with an API judge
- Manually annotate a stratified sample containing positives, negatives,
  threshold-near cases, judge-disagreement cases, and all reported exemplars
- Report judge correlation, mean absolute difference, and disagreement rate

Use at least two blinded annotators, report quadratic-weighted kappa, and
adjudicate score differences of 20 points or more.

The local judge is a screening instrument. API or human validation is required
for every result used in the abstract, main figures, and conclusion.

## Leakage Controls

- Never define a warning threshold from the same run used for test reporting.
- Never use future checkpoints as detector features.
- Never tune the prediction horizon after viewing held-out performance.
- Preserve failed runs and excluded samples in an audit log with reasons.
- Treat repeated generations from one question as clustered observations.
- Bootstrap confidence intervals by run or question, not by individual response.

## Adaptation-Method Robustness

The primary grid remains QLoRA/LoRA for compute feasibility. A prespecified
Tier-2 ablation compares rank-32 RSLoRA with rank-16 LoRA and rank-32 DoRA on
Qwen medical misaligned-1 seed 0. This tests PEFT sensitivity but does not
replace full-parameter fine-tuning; retain that limitation unless a separately
budgeted full-parameter run is added.

## Required Reporting

- Per-model and per-domain plots in addition to pooled plots
- Number of runs, checkpoints, questions, generations, and valid judge scores
- Missing-run and parse-failure accounting
- Negative and null results
- Exact checkpoint cadence and conversion from checkpoint lead time to optimizer
  steps and training examples
- Compute, evaluator, and intervention budgets
- Explicit separation between internal persona evaluations, held-out benchmark
  evaluations, and external saved-checkpoint organisms
