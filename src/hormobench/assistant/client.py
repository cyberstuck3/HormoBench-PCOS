"""
OpenAI client — centralised API access for the Research Assistant.

All LLM calls in the assistant subsystem go through :func:`chat` so that
model selection, error handling, and token usage are in one place.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

# Default model; overridable via HORMOBENCH_MODEL env var.
DEFAULT_MODEL = "gpt-4o"


class APIKeyMissing(RuntimeError):
    """Raised when OPENAI_API_KEY is not set."""


def get_client() -> "OpenAI":
    """Return an ``openai.OpenAI`` instance.

    Reads ``OPENAI_API_KEY`` from the environment.  Raises a human-friendly
    :class:`APIKeyMissing` error instead of a cryptic SDK traceback.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise APIKeyMissing(
            "OPENAI_API_KEY is not set.\n\n"
            "  export OPENAI_API_KEY='sk-...'\n\n"
            "You can obtain a key at https://platform.openai.com/api-keys"
        )
    from openai import OpenAI  # deferred so the package is importable without openai installed
    return OpenAI(api_key=api_key)


def _model_name() -> str:
    """Return the model name to use, checking env override first."""
    return os.environ.get("HORMOBENCH_MODEL", DEFAULT_MODEL)


def chat(
    system: str,
    user: str,
    *,
    client: "OpenAI | None" = None,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """Send a single system+user message pair and return the assistant text.

    Parameters
    ----------
    system : str
        The system prompt (persona / instructions).
    user : str
        The user-facing content (benchmark data + question).
    client : OpenAI, optional
        A pre-built client; if *None* a fresh one is created via
        :func:`get_client`.
    model : str, optional
        Override the model name for this call.
    temperature : float
        Sampling temperature.  Low values keep the output factual.
    max_tokens : int
        Maximum tokens in the completion.

    Returns
    -------
    str
        The assistant's reply text.
    """
    if client is None:
        client = get_client()
    response = client.chat.completions.create(
        model=model or _model_name(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
