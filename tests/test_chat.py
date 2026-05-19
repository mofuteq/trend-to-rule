from core.models import Claim, SearchQuery, StructuredClaims
from services import chat
from services.prompt_service import (
    TEMPLATE_SEARCH_QUERY,
    TEMPLATE_STRUCTURED_CLAIMS,
    TEMPLATE_STRUCTURED_DRAFT,
)


class _FakeDate:
    @classmethod
    def today(cls):
        return cls()

    def isoformat(self):
        return "2026-05-15"


def test_generate_search_query_sends_stable_date_context(monkeypatch):
    captured: dict[str, object] = {}
    expected = SearchQuery(
        canonical_query="classic tailoring social signals",
        emerging_query="2026 casual suiting shift signals",
    )

    def fake_create(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(chat, "create", fake_create)
    monkeypatch.setattr(chat, "date", _FakeDate)

    result = chat.generate_search_query(
        user_prompt="What suit signals are changing?",
        request_goal="compare tailoring signals",
        last_request_goal=None,
    )

    assert result is expected
    assert captured["response_model"] is SearchQuery
    assert "Current Date:" in captured["system_prompt"]
    assert "2026-05-15" in captured["system_prompt"]
    assert "Now:" not in captured["system_prompt"]


def test_generate_search_query_prompt_renders_with_current_date():
    prompt = TEMPLATE_SEARCH_QUERY.module.system(
        request_goal="compare tailoring signals",
        last_request_goal=None,
        current_date="2026-05-15",
    )

    assert "Current Date:" in prompt
    assert "2026-05-15" in prompt
    assert "e.g., 2026, recent, last year" in prompt
    assert "Now:" not in prompt


def test_extract_claims_sends_instructions_as_system_and_context_as_user(
    monkeypatch,
):
    captured: dict[str, object] = {}
    expected = StructuredClaims(
        canonical_claims=[
            Claim(
                claim="Tailoring signals formality.",
                claim_type="signal",
                source_id="C01",
            )
        ],
        emerging_claims=[],
    )

    def fake_create(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(chat, "create", fake_create)

    result = chat.extract_claims(
        canonical_context="[C01] Tailoring remains a formal baseline.",
        emerging_context="[E01] Casual suiting appears in street styling.",
        request_goal="compare tailoring signals",
    )

    assert result is expected
    assert captured["response_model"] is StructuredClaims
    assert "You are an information extraction system." in captured["system_prompt"]
    assert "Current Request Goal:" in captured["system_prompt"]
    assert "# Canonical Context:" in captured["user_prompt"]
    assert "# Emerging Context:" in captured["user_prompt"]
    assert "Tailoring remains a formal baseline" in captured["user_prompt"]


def test_extract_claims_prompt_preserves_source_ids_via_schema_fields():
    prompt = TEMPLATE_STRUCTURED_CLAIMS.module.system(
        request_goal="compare tailoring signals",
    )

    assert "lists of claim objects, not lists of strings" in prompt
    assert "`claim`" in prompt
    assert "`claim_type`" in prompt
    assert "`source_id`" in prompt
    assert "without a bracketed source ID" in prompt
    assert "preserve the ID in `source_id` instead of appending it to `claim`" in prompt
    assert "Do not append source IDs inside `claim`" in prompt
    assert "Now:" not in prompt


def test_extract_structured_draft_prompt_renders_without_runtime_time():
    prompt = TEMPLATE_STRUCTURED_DRAFT.module.system(
        request_goal="compare tailoring signals",
    )

    assert "first-pass synthesis engine" in prompt
    assert "Current Request Goal:" in prompt
    assert "Now:" not in prompt
