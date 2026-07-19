# Project Philosophy

> **Purpose of this document.** Explain why HormoBench-PCOS exists, why it was
> designed this way, and what future researchers should — and should not — use
> it for.

## Origin Story: Meet Sarah

Sarah is 34. For years she has struggled with fatigue, irregular cycles, migraines,
and brain fog — seeing multiple physicians without clear answers. Each appointment
captures only a snapshot, yet hormones shift continuously with sleep, stress,
nutrition, and age.

Sarah's story, adapted from the Hack-Nation Challenge 05 briefing, is not a
single anecdote. Women represent more than 50% of the global population, yet
female physiology remains one of the least represented domains in AI and
biomedical research. PCOS is diagnosed after an average of two years and three
physicians. This is a data-infrastructure problem as much as a clinical one.

## Why a Benchmark, Not Another Model

The temptation in a 24-hour AI hackathon is to ship "yet another PCOS predictor"
— a Streamlit app on top of a tuned XGBoost. That approach optimises for
demo-day theatre and, by the challenge briefing's own definition, produces a
**weak submission**: an isolated application with no reusable dataset,
benchmark, or model contribution.

HormoBench-PCOS is deliberately positioned differently. Every design decision
answers the question: *what would a researcher three months from now need in
order to extend, replicate, or challenge this work?*

The contribution is not the model. The contribution is the **infrastructure**:
a cleaned dataset, a documented split, a reproducible evaluation pipeline, a
model card, and a set of decision records that make each methodological choice
inspectable.

## Design Principles

### 1. Transparency over maximum accuracy

We could have trained a gradient-boosted tree ensemble and reported an
ROC-AUC north of 0.96. We chose L2-regularised Logistic Regression instead.
The DeLong test on the held-out test set confirms this is not a costly
choice — the tree-free alternatives are not significantly better
(see [`docs/adr/ADR-001-logistic-regression-baseline.md`](adr/ADR-001-logistic-regression-baseline.md)).
What Logistic Regression *does* give us is native compatibility with the
exact SHAP LinearExplainer, human-readable coefficients, and a decision
surface that a clinician can inspect line by line.

### 2. Reproducibility as a first-class citizen

Every number in this repository can be regenerated with one command:
`python scripts/benchmark.py`. A fixed random seed (42) governs the split,
the CV folds, the bootstrap resampling, and every model's initialisation.
Split indices are serialised to `data/splits/splits.json` so that even
under a different Python version the same 82 test cases are evaluated.
This discipline is the difference between a benchmark and a demo.

### 3. Clinical grounding of every technical decision

The decision threshold is not the sklearn default of 0.5 — that is a
mathematical convention, not a clinical decision. It is 0.21, selected
on the validation set to prioritise sensitivity, because missed PCOS
diagnoses carry documented harms (sub-fertility, endometrial hyperplasia,
2–4× baseline type-2 diabetes risk) while false positives cost only a
follow-up visit. Every such choice has a paragraph and a citation.

### 4. Honest limitations

The dataset comes from ten hospitals in Kerala, India. It is 541 patients
strong. It uses Rotterdam-era labels. None of these are ideal. Rather than
suppress those facts, [`docs/LIMITATIONS.md`](LIMITATIONS.md) makes them
explicit and calls out the specific extensions that would remedy each.

## Who This Repository Is For

- **Researchers** who want a working baseline they can extend with new models,
  new datasets, or new evaluation strategies without rebuilding infrastructure.
- **Students** learning medical-AI methodology — anti-leakage architecture,
  calibration, threshold optimisation, SHAP interpretation — with a fully
  worked example under 20 seconds of run time.
- **Clinicians and epidemiologists** who want to inspect what the model has
  actually learned before trusting any deployment claim.
- **Hack-Nation judges** who want to verify, in a few minutes, that the
  submission meets the "Data & Benchmark Infrastructure" and "AI Model
  Infrastructure" contribution layers.

## What This Repository Is Not

- **Not a diagnostic tool.** PCOS diagnosis requires clinical evaluation
  applying the Rotterdam criteria; a probabilistic score from a 541-patient
  cohort is not a substitute.
- **Not a production system.** No API, no monitoring, no drift detection.
  Those belong to a deployment project built *on top of* this benchmark.
- **Not a claim of state-of-the-art.** The point is reproducibility and
  extensibility, not leaderboard position.

## Related Documents

- [`METHODOLOGY.md`](METHODOLOGY.md) — technical deep-dive on rigor decisions
- [`LIMITATIONS.md`](LIMITATIONS.md) — explicit failure modes and scope
- [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) — formal input/output contract
- [`adr/`](adr/) — five decision records with alternatives considered
