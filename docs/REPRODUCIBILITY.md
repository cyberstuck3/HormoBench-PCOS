# Reproducibility

> **Purpose of this document.** Practical instructions for reproducing every
> number, table, and figure in this repository, plus a checklist for
> verifying that the reproduction succeeded.

## TL;DR

```bash
git clone <repository-url>
cd hormobench-pcos
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/benchmark.py
```

Total wall time: **≈ 20 seconds** on a modern CPU. No GPU required.

## Environment

- **Python**: 3.12 (the reference run used 3.12; 3.11 also works)
- **OS**: Linux x86-64 tested; macOS should work; Windows untested
- **Dependencies**: pinned in `requirements.txt`
- **RAM**: <500 MB peak
- **Disk**: <100 MB for repository + generated results

## Stage-by-Stage Reproduction

`scripts/benchmark.py` runs the four stages in sequence. Each stage can
also be executed independently for debugging:

```bash
python scripts/clean_data.py       # Stage 1 → data/processed/pcos_clean.csv
python scripts/train_baseline.py   # Stage 2 → results/models/*, results/results.json
python scripts/explain.py          # Stage 3 → results/figures/shap_*.png
python scripts/run_rigor.py        # Stage 4 → results/rigor_all_results.json + docs/evaluation_section.md
```

Stages depend on their predecessors — running them out of order will
fail with a `FileNotFoundError` on the missing artefact.

## Verification: Did the Reproduction Succeed?

After `python scripts/benchmark.py` exits with code 0, check:

### 1. Files present

```bash
ls results/rigor_all_results.json     # → exists
ls results/tables/*.md                # → 4 files
ls results/figures/*.png              # → 7 files
ls docs/evaluation_section.md         # → exists
```

### 2. Headline numbers match

```bash
python - <<'PY'
import json
r = json.load(open("results/rigor_all_results.json"))
m = r["test_metrics"]
assert abs(m["roc_auc"]["point"] - 0.951) < 1e-3, m["roc_auc"]
assert abs(m["recall"]["point"]  - 0.963) < 1e-3, m["recall"]
assert abs(m["pr_auc"]["point"]  - 0.932) < 1e-3, m["pr_auc"]
print("OK — reproduction matches expected headline numbers.")
PY
```

### 3. Deterministic split

```bash
python - <<'PY'
import json
s = json.load(open("data/splits/splits.json"))
assert s["seed"] == 42
assert len(s["train_idx"]) == 377
assert len(s["val_idx"])   == 82
assert len(s["test_idx"])  == 82
print("OK — split is deterministic and correctly stratified.")
PY
```

## Determinism Guarantees

See [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md) § "Determinism Guarantees" for
a per-artefact table.

Summary:

- **JSON dumps and markdown tables**: bit-identical run-to-run.
- **PNG figures**: identical numeric content; potentially different
  pixel bytes across matplotlib backends.
- **`.joblib` model files**: identical model behaviour; potentially
  different bytes across joblib versions.

## What Can Break Reproduction (and how to fix)

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: hormobench` | src/ not on `sys.path` | Run scripts from repo root, not from `scripts/` |
| Numbers off by 1e-4 | Different scikit-learn version | Pin to `1.8.0` per `requirements.txt` |
| `FileNotFoundError` on splits | Ran Stage 3 or 4 before Stage 2 | Run `scripts/benchmark.py` end-to-end |
| SHAP crashes | shap ≠ 0.52.0 | Pin per `requirements.txt` |
| Different LR winner | Different sklearn LR default solver | Pin sklearn version |

## Docker (not provided)

A Dockerfile is intentionally omitted for this hackathon submission —
the pinned `requirements.txt` and short reproduction steps make it
unnecessary. Adding one is a straightforward future contribution.

## Extending, Then Reproducing

If you fork and modify the benchmark, keep the reproduction contract
by:

1. Not changing `SEED`, `DECISION_THRESHOLD`, or `N_BOOTSTRAP` in
   `src/hormobench/__init__.py` unless bumping the version.
2. Documenting any hyperparameter grid changes in a new ADR under
   `docs/adr/`.
3. Rerunning `python scripts/benchmark.py` and committing the
   updated `results/` and `docs/evaluation_section.md`.

## Related Documents

- [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) — formal spec
- [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md) — what to expect in `results/`
- [`METHODOLOGY.md`](METHODOLOGY.md) — why each stage is designed this way
