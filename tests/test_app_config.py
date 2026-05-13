from core.app_config import load_app_config


def test_load_app_config_parses_tavily_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAT_DB_PATH", str(tmp_path / "chat_db"))
    monkeypatch.setenv(
        "LANGGRAPH_SQLITE_PATH",
        str(tmp_path / "langgraph" / "checkpoints.sqlite"),
    )
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("TAVILY_TEXT_MAX_RESULTS", "7")
    monkeypatch.setenv("TAVILY_SEARCH_DEPTH", "advanced")
    monkeypatch.setenv("TAVILY_INCLUDE_RAW_CONTENT", "yes")
    monkeypatch.setenv("TAVILY_IMAGE_FETCH_LIMIT", "8")
    monkeypatch.setenv("TAVILY_IMAGE_LIMIT", "4")
    monkeypatch.setenv("TAVILY_INCLUDE_IMAGE_DESCRIPTIONS", "off")

    config = load_app_config()

    assert config.tavily_api_key == "test-tavily-key"
    assert config.tavily_text_max_results == 7
    assert config.tavily_search_depth == "advanced"
    assert config.tavily_include_raw_content is True
    assert config.tavily_image_fetch_limit == 8
    assert config.tavily_image_limit == 4
    assert config.tavily_include_image_descriptions is False
