"""
HormoBench Research Assistant
=============================
A read-only subsystem that uses the OpenAI API to help researchers
understand benchmark outputs produced by the HormoBench-PCOS pipeline.

This package sits ON TOP of the benchmark — it never modifies results,
never retrains models, and never alters splits or evaluation metrics.
Every answer it produces is grounded in the artefacts under ``results/``
and ``docs/``.
"""

from hormobench.assistant.client import get_client, chat
from hormobench.assistant.loader import load_all_outputs
from hormobench.assistant.report import generate_report
from hormobench.assistant.qa import interactive_qa

__all__ = ["get_client", "chat", "load_all_outputs",
           "generate_report", "interactive_qa"]
