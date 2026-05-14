import importlib
from types import SimpleNamespace

import pytest


def test_default_constants_reflect_llm_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "vendor/some-model")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_OUTPUT_RETRIES", "5")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")

    from services import llm_client

    reloaded = importlib.reload(llm_client)

    assert reloaded.DEFAULT_LLM_MODEL == "vendor/some-model"
    assert reloaded.DEFAULT_LLM_API_KEY == "test-llm-key"
    assert reloaded.DEFAULT_LLM_BASE_URL == "https://example.test/v1"
    assert reloaded.DEFAULT_LLM_OUTPUT_RETRIES == 5
    assert reloaded.DEFAULT_LLM_REASONING_EFFORT == "high"


def test_default_model_falls_back_to_openrouter_endpoint(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_OUTPUT_RETRIES", raising=False)
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)

    from services import llm_client

    reloaded = importlib.reload(llm_client)

    assert reloaded.DEFAULT_LLM_MODEL == "google/gemini-3-flash-preview"
    assert reloaded.DEFAULT_LLM_BASE_URL == "https://openrouter.ai/api/v1"
    assert reloaded.DEFAULT_LLM_OUTPUT_RETRIES == 3
    assert reloaded.DEFAULT_LLM_REASONING_EFFORT == "low"


def test_base_url_may_be_empty(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "")

    from services import llm_client

    reloaded = importlib.reload(llm_client)

    assert reloaded.DEFAULT_LLM_BASE_URL == ""


@pytest.mark.parametrize(
    "effort", ["minimal", "low", "medium", "high", "xhigh"]
)
def test_valid_reasoning_effort_accepted(monkeypatch, effort):
    monkeypatch.setenv("LLM_REASONING_EFFORT", effort)

    from services import llm_client

    reloaded = importlib.reload(llm_client)

    assert reloaded.DEFAULT_LLM_REASONING_EFFORT == effort


def test_invalid_reasoning_effort_raises(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "ultra")

    from services import llm_client

    with pytest.raises(ValueError, match="Invalid LLM_REASONING_EFFORT"):
        importlib.reload(llm_client)


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_invalid_output_retries_raises(monkeypatch, value):
    monkeypatch.setenv("LLM_OUTPUT_RETRIES", value)

    from services import llm_client

    with pytest.raises(ValueError, match="Invalid LLM_OUTPUT_RETRIES"):
        importlib.reload(llm_client)


def test_create_passes_output_retries_to_agent(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("LLM_OUTPUT_RETRIES", "4")

    from services import llm_client

    reloaded = importlib.reload(llm_client)
    captured: dict[str, object] = {}

    class FakeProvider:
        def __init__(self, **kwargs):
            captured["provider_kwargs"] = kwargs

    class FakeModel:
        def __init__(self, model, provider):
            captured["model"] = model
            captured["provider"] = provider

    class FakeRunResult:
        output = "ok"

        def usage(self):
            return SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0)

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["agent_kwargs"] = kwargs

        def run_sync(self, user_prompt, message_history, model_settings):
            captured["run_sync"] = {
                "user_prompt": user_prompt,
                "message_history": message_history,
                "model_settings": model_settings,
            }
            return FakeRunResult()

    monkeypatch.setattr(reloaded, "OpenAIProvider", FakeProvider)
    monkeypatch.setattr(reloaded, "OpenAIChatModel", FakeModel)
    monkeypatch.setattr(reloaded, "Agent", FakeAgent)

    result = reloaded.create("hello")

    assert result.text == "ok"
    assert captured["provider_kwargs"] == {
        "api_key": "test-llm-key",
        "base_url": "https://openrouter.ai/api/v1",
    }
    assert captured["agent_kwargs"]["output_retries"] == 4
    assert captured["agent_kwargs"]["output_type"] is str


def test_missing_api_key_raises_on_create(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")

    from services import llm_client

    reloaded = importlib.reload(llm_client)

    with pytest.raises(ValueError, match="LLM_API_KEY is required"):
        reloaded.create("hello")
