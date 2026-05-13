import importlib

import pytest


def test_default_constants_reflect_llm_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "vendor/some-model")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")

    from services import llm_client

    reloaded = importlib.reload(llm_client)

    assert reloaded.DEFAULT_LLM_MODEL == "vendor/some-model"
    assert reloaded.DEFAULT_LLM_API_KEY == "test-llm-key"
    assert reloaded.DEFAULT_LLM_BASE_URL == "https://example.test/v1"


def test_default_model_falls_back_to_openrouter_prefix(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    from services import llm_client

    reloaded = importlib.reload(llm_client)

    assert reloaded.DEFAULT_LLM_MODEL == "openrouter/google/gemini-3-flash-preview"
    assert reloaded.DEFAULT_LLM_BASE_URL == ""


def test_base_url_may_be_empty(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "")

    from services import llm_client

    reloaded = importlib.reload(llm_client)

    assert reloaded.DEFAULT_LLM_BASE_URL == ""


def test_missing_api_key_raises_on_create(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")

    from services import llm_client

    reloaded = importlib.reload(llm_client)

    with pytest.raises(ValueError, match="LLM_API_KEY is required"):
        reloaded.create("hello")
