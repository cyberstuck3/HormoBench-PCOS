"""
scripts/benchmark.py — one-command reproduction of every HormoBench-PCOS
number, figure, and table.

Usage
-----
    python scripts/benchmark.py

Stages
------
    1. clean_data       → data/processed/pcos_clean.csv
    2. train_baseline   → data/splits/splits.json, results/models/*, results/results.json
    3. explain          → results/figures/shap_*.png, results/shap_importance.csv
    4. run_rigor        → results/{figures,tables,narratives}/*, docs/evaluation_section.md,
                          results/rigor_all_results.json

Exit codes
----------
    0  All four stages succeeded.
    1  A stage failed (details in stderr and the traceback printed above).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib.util


def _load_stage(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    scripts_dir = Path(__file__).resolve().parent
    stages = [
        ("Stage 1 — Clean data",            scripts_dir / "clean_data.py"),
        ("Stage 2 — Train baseline models", scripts_dir / "train_baseline.py"),
        ("Stage 3 — SHAP explainability",   scripts_dir / "explain.py"),
        ("Stage 4 — Scientific rigor",      scripts_dir / "run_rigor.py"),
    ]

    print("=" * 72)
    print("HormoBench-PCOS  ·  one-command benchmark")
    print("=" * 72)

    t0 = time.time()
    for idx, (label, path) in enumerate(stages, 1):
        print(f"\n{'─' * 72}\n▶ {label}\n{'─' * 72}")
        stage_t = time.time()
        try:
            mod = _load_stage(f"stage_{idx}", path)
            mod.main()
        except Exception as exc:  # noqa: BLE001
            print(f"\n✗ {label} FAILED: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        print(f"\n✓ {label} completed in {time.time() - stage_t:.1f}s")

    total = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"✓ All four stages completed in {total:.1f}s")
    print(f"{'=' * 72}")
    print("\nSee:")
    print("  results/rigor_all_results.json  — machine-readable metrics")
    print("  docs/evaluation_section.md      — paper-style write-up")
    print("  results/figures/                — publication-quality figures")


if __name__ == "__main__":
    main()
