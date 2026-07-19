# Judge Checklist

> **Purpose of this document.** Map every Hack-Nation Challenge 05 criterion
> to specific evidence inside this repository. Judges can verify each
> requirement in seconds; no hunting required.

## How to read this table

- **Requirement** — quoted or paraphrased from the challenge briefing
- **Evidence** — the concrete artefact in this repository
- **Location** — file path
- **Line/section** — where inside the file, when relevant

---

## Success Criteria (challenge briefing § "Success Criteria")

### 01. Women's Health Impact
*"How significantly could this work improve women's health, and how many
women could ultimately benefit?"*

| Sub-question | Evidence | Location |
|--------------|----------|----------|
| Who benefits? | Target audience section | [`PROJECT_PHILOSOPHY.md`](PROJECT_PHILOSOPHY.md) § "Who This Repository Is For" |
| What clinical outcome improves? | Recall 0.963 (95 % CI 0.867–1.000) — 89 % reduction in missed diagnoses | [`evaluation_section.md`](evaluation_section.md) § 3 + `results/figures/threshold_analysis.png` |
| How is impact framed? | Sarah scenario + trillion-dollar economic framing | [`PROJECT_PHILOSOPHY.md`](PROJECT_PHILOSOPHY.md) § "Origin Story" |
| Extensibility to more women | Extension points in benchmark spec | [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) § 7 |

### 02. Technical Excellence
*"How innovative, rigorous, reproducible, and scalable is the proposed
dataset, benchmark, model, or application?"*

| Sub-question | Evidence | Location |
|--------------|----------|----------|
| Innovation over prior PCOS AI | Calibration + DeLong + bootstrap CIs (rare in prior work) | [`METHODOLOGY.md`](METHODOLOGY.md) §§ 5, 7 |
| Rigor: anti-leakage | Six explicit invariants documented | [`METHODOLOGY.md`](METHODOLOGY.md) § 1 |
| Rigor: uncertainty quantification | Bootstrap 95 % CI for every metric | [`evaluation_section.md`](evaluation_section.md) § 3, [ADR-005](adr/ADR-005-bootstrap-confidence-intervals.md) |
| Rigor: statistical comparison | DeLong pairwise across all four models | [`evaluation_section.md`](evaluation_section.md) § 2, `results/tables/delong_pairwise.md` |
| Reproducibility: one command | `python scripts/benchmark.py` — 20 seconds | [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) § "TL;DR" |
| Reproducibility: seed policy | Single fixed seed, split serialisation | [`METHODOLOGY.md`](METHODOLOGY.md) § 8 |
| Scalability: modular architecture | `src/hormobench/` package structure | Repo root, README § "Repository Structure" |
| Scalability: extension points | Documented for new models, metrics, datasets, explainers | [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) § 7 |

### 03. Foundation Value
*"Does the project leave behind reusable infrastructure that accelerates
future research?"*

| Sub-question | Evidence | Location |
|--------------|----------|----------|
| Reusable dataset | Cleaned CSV + fixed splits published | `data/processed/pcos_clean.csv`, `data/splits/splits.json` |
| Reusable benchmark | Formal specification independent of implementation | [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) |
| Reusable model | Trained model + preprocessor serialised | `results/models/*.joblib` |
| Reusable evaluation pipeline | `scripts/run_rigor.py` + `src/hormobench/rigor/` | Repo root |
| Open license | MIT (code); CC BY-NC-SA 4.0 (data) with attribution | `LICENSE`, [`DATASET_CARD.md`](DATASET_CARD.md) |
| Documented reproduction contract | See § "Reproduction Contract" | [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) § 5 |
| Extensibility documented | See § 7 | [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) § 7 |

---

## Strong Submission Checklist (challenge briefing § "What Makes a Strong Submission")

| Requirement | Evidence | Location |
|-------------|----------|----------|
| "Publish reusable datasets, benchmarks, model checkpoints, and evaluation pipelines under an open license" | MIT-licensed code; data attributed per CC BY-NC-SA; checkpoints in `results/models/` | `LICENSE`, [`DATASET_CARD.md`](DATASET_CARD.md), `results/models/` |
| "Solve one clearly defined prediction or infrastructure problem exceptionally well, with transparent methods and evidence" | Single problem (PCOS risk stratification); 5 ADRs + full methodology doc | [`adr/`](adr/), [`METHODOLOGY.md`](METHODOLOGY.md) |
| "Share reproducible code and documentation so the research community can extend the work" | One-command reproduction, extension points, 10+ docs | [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) § 7 |

---

## Weak Submission Traps (challenge briefing § same)

Explicit evidence of **not** falling into each trap:

| Trap | Why we avoid it | Where it is addressed |
|------|-----------------|-----------------------|
| "Build an isolated application with no reusable dataset, benchmark, or model contribution" | Positioned as benchmark first; explicit non-goal of app-building | [`PROJECT_PHILOSOPHY.md`](PROJECT_PHILOSOPHY.md) § "Why a Benchmark, Not Another Model" |
| "Depend on undocumented proprietary data or hide core assumptions" | Public Kaggle dataset; every preprocessing step and hyperparameter documented | [`DATASET_CARD.md`](DATASET_CARD.md), [`METHODOLOGY.md`](METHODOLOGY.md) |
| "Make unsupported medical or diagnostic claims" | Explicit non-diagnostic scoping + 10-section limitations document | [`LIMITATIONS.md`](LIMITATIONS.md), [`MODEL_CARD.md`](MODEL_CARD.md) § "Out-of-Scope Use" |
| "Present a polished interface without scientific validation" | No polished interface at all; scientific rigor is the deliverable | The entire `docs/` folder |

---

## Open Science Requirement (challenge briefing § same)

*"Whenever possible, publish your datasets, benchmarks, source code, model
checkpoints, documentation, and evaluation pipelines."*

| Artefact | Published as | Location |
|----------|--------------|----------|
| Dataset | Cleaned CSV | `data/processed/pcos_clean.csv` |
| Splits | Serialised JSON | `data/splits/splits.json` |
| Source code | Full repository | Repo root |
| Model checkpoint | joblib serialised | `results/models/best_model.joblib` |
| Preprocessor checkpoint | joblib serialised | `results/models/preprocessor.joblib` |
| Documentation | 10+ documents | `docs/` |
| Evaluation pipeline | `scripts/benchmark.py` + `src/hormobench/rigor/` | Repo root |
| License | MIT (permissive) + CC BY-NC-SA note | `LICENSE` |

---

## Deliverables (challenge briefing § "Deliverables")

*"Working prototype and source code, technical documentation and dataset
description, benchmark methodology and a short demonstration video."*

| Deliverable | Evidence | Location |
|-------------|----------|----------|
| Working prototype | Full pipeline runs in 20 seconds | `scripts/benchmark.py` |
| Source code | Modular Python package | `src/hormobench/`, `scripts/` |
| Technical documentation | Methodology, ADRs, model card | `docs/` |
| Dataset description | HF-style dataset card | [`DATASET_CARD.md`](DATASET_CARD.md) |
| Benchmark methodology | Formal spec + practical methodology | [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md), [`METHODOLOGY.md`](METHODOLOGY.md) |
| Demonstration video | See README § "Demo" | *(link in README when recorded)* |

---

## Measure the Impact (challenge briefing § same)

| Dimension | HormoBench-PCOS answer |
|-----------|------------------------|
| **Reach** — how many women could ultimately benefit? | Any research group evaluating PCOS classifiers on tabular clinical data now has a fixed baseline, a documented split, and an evaluation pipeline they can extend. Downstream: any tool that uses this benchmark to justify its methodology inherits its reproducibility. |
| **Quality of life** — how meaningfully could the solution improve health outcomes, access, understanding, or daily well-being? | Directly, none — this is not a deployed tool. Indirectly, by lowering the methodological floor for future PCOS-AI research, HormoBench-PCOS shortens the path from research to deployed screening. |

---

## Self-Verification Before Submission

- [ ] `python scripts/benchmark.py` exits 0 in <60 seconds
- [ ] Headline numbers match [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) § "Verification"
- [ ] Every doc referenced in this checklist exists at its stated path
- [ ] README ≤ 400 lines with a working documentation index
- [ ] No `__pycache__/` or `.ipynb_checkpoints/` in the repository
- [ ] License and citation files parseable by GitHub
- [ ] Repository description and topics set on GitHub

## Related Documents

- [`PROJECT_PHILOSOPHY.md`](PROJECT_PHILOSOPHY.md) — the "why"
- [`METHODOLOGY.md`](METHODOLOGY.md) — the "how"
- [`LIMITATIONS.md`](LIMITATIONS.md) — the "what this doesn't do"
