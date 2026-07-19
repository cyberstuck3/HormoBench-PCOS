"""
Benchmark output loader — read-only access to every artefact the
Research Assistant needs.

Every function here reads a file and returns a Python object.
Nothing is written.  Nothing is modified.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hormobench import (
    BASELINE_JSON, RIGOR_JSON, SHAP_CSV, DOCS,
)


@dataclass
class BenchmarkOutputs:
    """Container for all benchmark artefacts consumed by the assistant."""
    baseline:         dict[str, Any] = field(default_factory=dict)
    rigor:            dict[str, Any] = field(default_factory=dict)
    shap_features:    list[dict[str, Any]] = field(default_factory=list)
    evaluation_md:    str = ""


def load_baseline() -> dict[str, Any]:
    """Load ``results/results.json``."""
    with open(BASELINE_JSON) as f:
        return json.load(f)


def load_rigor() -> dict[str, Any]:
    """Load ``results/rigor_all_results.json``."""
    with open(RIGOR_JSON) as f:
        return json.load(f)


def load_shap() -> list[dict[str, Any]]:
    """Load ``results/shap_importance.csv`` as a list of dicts."""
    rows: list[dict[str, Any]] = []
    with open(SHAP_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "feature": row["feature"],
                "mean_abs_shap": float(row["mean_abs_shap"]),
            })
    return rows


def load_evaluation_md() -> str:
    """Load ``docs/evaluation_section.md`` as a string."""
    path = DOCS / "evaluation_section.md"
    if path.exists():
        return path.read_text()
    return ""


def load_all_outputs() -> BenchmarkOutputs:
    """Load every benchmark artefact into a single container."""
    return BenchmarkOutputs(
        baseline=load_baseline(),
        rigor=load_rigor(),
        shap_features=load_shap(),
        evaluation_md=load_evaluation_md(),
    )


# ── Formatting helpers (for prompt injection) ────────────────────────────

def format_test_metrics(rigor: dict) -> str:
    """Pretty-print the test-set metrics block for prompt injection."""
    metrics = rigor.get("test_metrics", {})
    lines = ["Metric | Point | 95% CI", "-------|------:|-------:"]
    for name, vals in metrics.items():
        lo, hi = vals.get("lo", "?"), vals.get("hi", "?")
        lines.append(f"{name} | {vals['point']:.3f} | [{lo:.3f}, {hi:.3f}]")
    return "\n".join(lines)


def format_shap_table(shap_features: list[dict], top_n: int = 15) -> str:
    """Markdown table of top-N SHAP features."""
    lines = ["Rank | Feature | Mean |SHAP|",
             "----:|---------|-----:"]
    for i, row in enumerate(shap_features[:top_n], 1):
        lines.append(f"{i} | {row['feature']} | {row['mean_abs_shap']:.4f}")
    return "\n".join(lines)


def format_model_comparison(rigor: dict) -> str:
    """CV results as a compact table."""
    cv = rigor.get("cv_results", [])
    lines = ["Model | Metric | Mean ± SD",
             "------|--------|----------"]
    for row in cv:
        lines.append(
            f"{row['model']} | {row['metric']} | "
            f"{row['mean']:.3f} ± {row['std']:.3f}"
        )
    return "\n".join(lines)
