import importlib

import pytest


def test_default_constants_reflect_llm_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("LLM_MODEL", "vendor/some-model")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "medium")

    from services import llm_client

    reloaded = importlib.reload(llm_client)

    assert reloaded.DEFAULT_LLM_BASE_URL == "https://example.test/v1"
    assert reloaded.DEFAULT_LLM_API_KEY == "test-llm-key"
    assert reloaded.DEFAULT_LLM_MODEL == "vendor/some-model"
    assert reloaded.DEFAULT_LLM_REASONING_EFFORT == "medium"


def test_invalid_reasoning_effort_raises(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "bogus")

    from services import llm_client

    with pytest.raises(ValueError, match="Invalid LLM_REASONING_EFFORT"):
        importlib.reload(llm_client)


def test_default_base_url_falls_back_to_openrouter(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)

    from services import llm_client

    reloaded = importlib.reload(llm_client)

    assert reloaded.DEFAULT_LLM_BASE_URL == "https://openrouter.ai/api/v1"
    assert reloaded.DEFAULT_LLM_MODEL == "google/gemini-3-flash-preview"
    assert reloaded.DEFAULT_LLM_REASONING_EFFORT == "low"


def test_missing_api_key_raises_on_create(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")

    from services import llm_client

    reloaded = importlib.reload(llm_client)

    with pytest.raises(ValueError, match="LLM_API_KEY is required"):
        reloaded.create("hello")
