from core.models import Claim, StructuredClaims
from services import chat


def test_extract_claims_sends_instructions_as_system_and_context_as_user(
    monkeypatch,
):
    captured: dict[str, object] = {}
    expected = StructuredClaims(
        canonical_claims=[
            Claim(
                claim="Tailoring signals formality. [C01]",
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
