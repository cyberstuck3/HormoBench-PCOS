#!/usr/bin/env python3
"""
Generate the AI-assisted research report from benchmark outputs.

Usage
-----
    export OPENAI_API_KEY="sk-..."
    python scripts/generate_ai_report.py

Produces
--------
    docs/AI_REPORT.md

The script reads benchmark outputs (read-only) and sends them to the
OpenAI API for scientific interpretation.  It never modifies any
benchmark artefact.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hormobench.assistant.report import generate_report


def main():
    print("=" * 64)
    print("  HormoBench-PCOS — AI Report Generator")
    print("=" * 64)
    print()
    generate_report(verbose=True)


if __name__ == "__main__":
    main()
