"""
Deterministic tests for the HormoBench Research Assistant.

Every test mocks OpenAI API calls — no internet connection or API key
required.  Tests verify that:
  • the loader reads benchmark artefacts correctly,
  • the client raises friendly errors when the API key is missing,
  • prompts contain required responsible-AI constraints,
  • utils produce correct output,
  • report generation calls the API for each section,
  • QA builds context from benchmark outputs.
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ─────────────────────────────────────────────────────────────────────────────
# Loader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoader:
    def test_load_baseline(self):
        from hormobench.assistant.loader import load_baseline
        data = load_baseline()
        assert "best_model" in data
        assert "all_models" in data
        assert isinstance(data["best_model"], str)

    def test_load_rigor(self):
        from hormobench.assistant.loader import load_rigor
        data = load_rigor()
        assert "test_metrics" in data
        assert "roc_auc" in data["test_metrics"]
        assert "point" in data["test_metrics"]["roc_auc"]

    def test_load_shap(self):
        from hormobench.assistant.loader import load_shap
        rows = load_shap()
        assert len(rows) > 0
        assert "feature" in rows[0]
        assert "mean_abs_shap" in rows[0]
        assert isinstance(rows[0]["mean_abs_shap"], float)

    def test_load_evaluation_md(self):
        from hormobench.assistant.loader import load_evaluation_md
        md = load_evaluation_md()
        assert "HormoBench-PCOS" in md

    def test_load_all_outputs(self):
        from hormobench.assistant.loader import load_all_outputs
        o = load_all_outputs()
        assert o.baseline
        assert o.rigor
        assert o.shap_features
        assert o.evaluation_md

    def test_format_test_metrics(self):
        from hormobench.assistant.loader import load_rigor, format_test_metrics
        rigor = load_rigor()
        table = format_test_metrics(rigor)
        assert "roc_auc" in table
        assert "|" in table

    def test_format_shap_table(self):
        from hormobench.assistant.loader import load_shap, format_shap_table
        shap = load_shap()
        table = format_shap_table(shap, top_n=5)
        lines = table.strip().split("\n")
        assert len(lines) == 7  # header + separator + 5 rows

    def test_format_model_comparison(self):
        from hormobench.assistant.loader import load_rigor, format_model_comparison
        rigor = load_rigor()
        table = format_model_comparison(rigor)
        assert "Model" in table


# ─────────────────────────────────────────────────────────────────────────────
# Client tests
# ─────────────────────────────────────────────────────────────────────────────

class TestClient:
    def test_missing_api_key_raises(self):
        from hormobench.assistant.client import get_client, APIKeyMissing
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(APIKeyMissing, match="OPENAI_API_KEY"):
                get_client()

    def test_chat_calls_api(self):
        """Mock the OpenAI client and verify chat() returns the response."""
        from hormobench.assistant.client import chat

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "  Test response  "
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        result = chat("system prompt", "user prompt", client=mock_client)
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    def test_model_override(self):
        from hormobench.assistant.client import chat

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        chat("sys", "usr", client=mock_client, model="gpt-5")
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-5"


# ─────────────────────────────────────────────────────────────────────────────
# Prompts tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPrompts:
    def test_all_prompts_contain_responsible_ai(self):
        from hormobench.assistant import prompts
        prompt_names = [
            "SCIENTIFIC_REVIEWER", "CLINICAL_EXPLAINER", "BENCHMARK_ASSISTANT",
            "FUTURE_WORK_GENERATOR", "PAPER_ABSTRACT_GENERATOR",
            "DATASET_LIMITATIONS",
            "REPORT_EXECUTIVE_SUMMARY", "REPORT_MODEL_SELECTION",
            "REPORT_THRESHOLD_ANALYSIS", "REPORT_SHAP_INTERPRETATION",
            "REPORT_CALIBRATION", "REPORT_CLINICAL_INTERPRETATION",
            "REPORT_LIMITATIONS", "REPORT_FUTURE_WORK",
            "REPORT_RESPONSIBLE_AI",
        ]
        for name in prompt_names:
            prompt = getattr(prompts, name)
            assert "Never" in prompt or "never" in prompt or "NEVER" in prompt, \
                f"{name} missing responsible-AI constraint"
            assert "diagnos" in prompt.lower() or "medical" in prompt.lower(), \
                f"{name} missing medical disclaimer reference"


# ─────────────────────────────────────────────────────────────────────────────
# Utils tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_pretty_json(self):
        from hormobench.assistant.utils import pretty_json
        result = pretty_json({"a": 1})
        assert '"a": 1' in result

    def test_compact_json(self):
        from hormobench.assistant.utils import compact_json
        result = compact_json({"a": 1})
        assert result == '{"a":1}'

    def test_md_heading(self):
        from hormobench.assistant.utils import md_heading
        assert md_heading("Test", 3) == "### Test\n"

    def test_md_disclaimer(self):
        from hormobench.assistant.utils import md_disclaimer
        d = md_disclaimer()
        assert "Disclaimer" in d
        assert "clinical" in d.lower()

    def test_estimate_tokens(self):
        from hormobench.assistant.utils import estimate_tokens
        assert estimate_tokens("hello world") >= 1
        assert estimate_tokens("a" * 400) == 100

    def test_safe_read_missing(self, tmp_path):
        from hormobench.assistant.utils import safe_read
        assert safe_read(tmp_path / "nope.txt") == ""

    def test_safe_write(self, tmp_path):
        from hormobench.assistant.utils import safe_write
        p = safe_write(tmp_path / "sub" / "f.txt", "hello")
        assert p.read_text() == "hello"


# ─────────────────────────────────────────────────────────────────────────────
# Report integration test (mocked API)
# ─────────────────────────────────────────────────────────────────────────────

class TestReport:
    def test_generate_report_calls_api_for_each_section(self, tmp_path):
        from hormobench.assistant import report

        call_count = {"n": 0}

        def fake_chat(system, user, *, client=None, max_tokens=2048):
            call_count["n"] += 1
            return f"Section {call_count['n']} content."

        with patch.object(report, "chat", side_effect=fake_chat), \
             patch.object(report, "get_client", return_value=MagicMock()), \
             patch.object(report, "safe_write", side_effect=lambda p, c: p):
            result = report.generate_report(verbose=False)

        # 9 sections defined in _SECTIONS
        assert call_count["n"] == 9
        assert "Executive Summary" in result
        assert "Responsible AI" in result
        assert "Disclaimer" in result


# ─────────────────────────────────────────────────────────────────────────────
# QA context builder test
# ─────────────────────────────────────────────────────────────────────────────

class TestQA:
    def test_build_context_contains_evidence(self):
        from hormobench.assistant.qa import _build_context
        from hormobench.assistant.loader import load_all_outputs

        outputs = load_all_outputs()
        ctx = _build_context(outputs)
        assert "BENCHMARK EVIDENCE" in ctx
        assert "roc_auc" in ctx
        assert "SHAP" in ctx
        assert "Winner" in ctx
