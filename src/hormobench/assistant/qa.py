"""
Interactive command-line Research Assistant (``scripts/copilot.py``).

Loads benchmark outputs once at startup, then enters a REPL where the
researcher can ask questions.  Every answer is grounded in the loaded
evidence; the assistant explicitly says when evidence is unavailable.
"""
from __future__ import annotations

from hormobench.assistant.client import chat, get_client
from hormobench.assistant.loader import (
    load_all_outputs,
    format_test_metrics,
    format_shap_table,
    format_model_comparison,
)
from hormobench.assistant.prompts import BENCHMARK_ASSISTANT
from hormobench.assistant.utils import pretty_json, estimate_tokens


def _build_context(outputs) -> str:
    """Assemble a context block from all benchmark outputs."""
    parts = [
        "=== BENCHMARK EVIDENCE (read-only) ===\n",
        f"\n## Winner\n{outputs.baseline.get('best_model', '?')} "
        f"with params {outputs.baseline.get('best_params', {})}\n",
        f"\n## Test Metrics\n{format_test_metrics(outputs.rigor)}\n",
        f"\n## CV Comparison\n{format_model_comparison(outputs.rigor)}\n",
        f"\n## SHAP Top 15\n{format_shap_table(outputs.shap_features)}\n",
        f"\n## Calibration\n{pretty_json(outputs.rigor.get('calibration', {}))}\n",
        f"\n## DeLong Pairwise\n{pretty_json(outputs.rigor.get('delong', {}))}\n",
        f"\n## Confusion Matrices\n{pretty_json(outputs.rigor.get('confusion_matrices', {}))}\n",
        f"\n## Config\n{pretty_json(outputs.rigor.get('config', {}))}\n",
    ]
    return "\n".join(parts)


def interactive_qa() -> None:
    """Run the interactive REPL.

    Type a question and press Enter.  Type ``quit``, ``exit``, or Ctrl-C
    to leave.
    """
    client = get_client()
    outputs = load_all_outputs()
    context = _build_context(outputs)

    print("=" * 64)
    print("  HormoBench-PCOS Research Assistant")
    print("  Ask questions about the benchmark outputs.")
    print("  Type 'quit' or 'exit' to leave.")
    print("=" * 64)
    print(f"  (benchmark context: ~{estimate_tokens(context)} tokens)\n")

    while True:
        try:
            question = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        user_msg = (
            f"{context}\n\n"
            f"=== RESEARCHER QUESTION ===\n{question}\n"
        )

        try:
            answer = chat(
                system=BENCHMARK_ASSISTANT,
                user=user_msg,
                client=client,
            )
        except Exception as exc:
            print(f"\n[Error] {exc}\n")
            continue

        print(f"\nAssistant > {answer}\n")
