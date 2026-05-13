from core.models import WebSource
from services.web_search import (
    dedupe_sources_by_url,
    normalize_tavily_text_results,
    sources_to_prompt_context,
)


def test_normalize_tavily_payload_into_web_sources_and_dedupe_urls():
    payload = {
        "results": [
            {
                "title": "  First source  ",
                "url": "HTTPS://Example.com/style/article/?utm_source=news#section",
                "content": " First snippet\nwith extra spacing. ",
                "published_date": "2026-05-01",
                "score": "0.91",
            },
            {
                "title": "Duplicate source",
                "url": "https://example.com/style/article?ref=feed",
                "content": "Duplicate should be dropped.",
            },
            {
                "title": "Second source",
                "url": "https://example.com/other?keep=1&utm_campaign=spring",
                "content": "Second snippet.",
            },
        ]
    }

    sources = normalize_tavily_text_results(payload, query_kind="canonical")

    assert [source.source_id for source in sources] == ["C01", "C02"]
    assert [source.query_kind for source in sources] == ["canonical", "canonical"]
    assert sources[0].title == "First source"
    assert sources[0].url == "https://example.com/style/article"
    assert sources[0].snippet == "First snippet with extra spacing."
    assert sources[0].published_at == "2026-05-01"
    assert sources[0].score == 0.91
    assert sources[1].url == "https://example.com/other?keep=1"


def test_normalize_tavily_payload_assigns_emerging_source_ids():
    sources = normalize_tavily_text_results(
        {
            "results": [
                {
                    "title": "Emerging source",
                    "url": "https://example.com/emerging",
                    "content": "Emerging snippet.",
                }
            ]
        },
        query_kind="emerging",
    )

    assert [source.source_id for source in sources] == ["E01"]
    assert [source.query_kind for source in sources] == ["emerging"]


def test_dedupe_sources_by_url_reassigns_lane_source_ids():
    canonical, emerging = dedupe_sources_by_url(
        canonical_sources=[
            WebSource(
                source_id="old",
                query_kind="emerging",
                title="Canonical",
                url="https://example.com/a?utm_source=x",
                snippet="Canonical snippet",
            )
        ],
        emerging_sources=[
            WebSource(
                source_id="old",
                query_kind="canonical",
                title="Duplicate",
                url="https://example.com/a/",
                snippet="Duplicate snippet",
            ),
            WebSource(
                source_id="old",
                query_kind="canonical",
                title="Emerging",
                url="https://example.com/b?ref=feed",
                snippet="Emerging snippet",
            ),
        ],
    )

    assert [(source.source_id, source.query_kind) for source in canonical] == [
        ("C01", "canonical")
    ]
    assert [(source.source_id, source.query_kind) for source in emerging] == [
        ("E01", "emerging")
    ]
    assert emerging[0].url == "https://example.com/b"


def test_malformed_tavily_payload_returns_empty_list():
    assert normalize_tavily_text_results(None, query_kind="canonical") == []
    assert normalize_tavily_text_results({}, query_kind="canonical") == []
    assert (
        normalize_tavily_text_results(
            {"results": "not a list"},
            query_kind="canonical",
        )
        == []
    )


def test_sources_to_prompt_context_renders_normalized_fields_only():
    sources = normalize_tavily_text_results(
        {
            "results": [
                {
                    "title": "Runway report",
                    "url": "https://example.com/report",
                    "content": "Visible snippet.",
                    "raw_content": "provider-only raw content",
                    "provider_payload": {"secret": "do not render"},
                }
            ]
        },
        query_kind="canonical",
        include_raw_content=False,
    )

    context = sources_to_prompt_context(sources, label="Canonical")

    assert "[C01]" in context
    assert "- lane=Canonical" in context
    assert "- title=Runway report" in context
    assert "- url=https://example.com/report" in context
    assert "- snippet=Visible snippet." in context
    assert "provider-only raw content" not in context
    assert "provider_payload" not in context
    assert "do not render" not in context
