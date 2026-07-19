#!/usr/bin/env python3
"""
HormoBench-PCOS Research Assistant — interactive terminal copilot.

Usage
-----
    export OPENAI_API_KEY="sk-..."
    python scripts/copilot.py

Examples
--------
    You > Why did precision decrease?
    You > Why was Logistic Regression selected?
    You > Explain SHAP.
    You > What does MCC mean?
    You > quit

The assistant answers using benchmark outputs plus GPT reasoning.
If benchmark evidence is unavailable, it says so explicitly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hormobench.assistant.qa import interactive_qa


def main():
    interactive_qa()


if __name__ == "__main__":
    main()
