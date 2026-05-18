from services.api_models import PersistedTurnArtifacts
from services.chat_workflow import RetrievalBundle
from services.image_search import ImageSearchResult
from ui import app_chat


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err

    def __setattr__(self, name, value):
        self[name] = value


class Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeStreamlit:
    def __init__(self):
        self.session_state = SessionState()
        self.calls = []

    def chat_message(self, role, avatar=None):
        self.calls.append(("chat_message", role, avatar))
        return Context()

    def markdown(self, content, **kwargs):
        self.calls.append(("markdown", content, kwargs))


def _artifact(chat_turn: int = 1) -> PersistedTurnArtifacts:
    return PersistedTurnArtifacts(
        chat_turn=chat_turn,
        image_results=[
            ImageSearchResult(
                title="Visual reference",
                page_url="https://example.test/page",
                image_url="https://example.test/image.jpg",
                thumbnail_url="https://example.test/thumb.jpg",
                source="fixture",
                engine="fixture",
            )
        ],
        retrieval=RetrievalBundle(
            canonical_context="",
            emerging_context="",
            canonical_rows=[
                {
                    "source_id": "C01",
                    "title": "Canonical source",
                    "url": "https://example.test/canonical",
                    "published_at": "2026-01-01",
                    "provider": "tavily",
                }
            ],
        ),
        image_query="visual query",
        request_goal="compare silhouettes",
        is_in_scope=True,
    )


def test_render_history_renders_persisted_artifacts_for_assistant_turn(
    monkeypatch,
):
    fake_st = FakeStreamlit()
    fake_st.session_state.messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    fake_st.session_state.turn_artifacts = {"1": _artifact(1)}
    rendered_turns = []
    monkeypatch.setattr(app_chat, "st", fake_st)
    monkeypatch.setattr(
        app_chat,
        "render_turn_artifacts",
        lambda artifacts: rendered_turns.append(artifacts.chat_turn),
    )

    app_chat.render_history()

    assert rendered_turns == [1]
    assert ("markdown", "Answer", {}) in fake_st.calls


def test_render_history_does_not_render_artifacts_for_user_messages(monkeypatch):
    fake_st = FakeStreamlit()
    fake_st.session_state.messages = [
        {"role": "user", "content": "Question only"},
    ]
    fake_st.session_state.turn_artifacts = {"1": _artifact(1)}
    rendered_turns = []
    monkeypatch.setattr(app_chat, "st", fake_st)
    monkeypatch.setattr(
        app_chat,
        "render_turn_artifacts",
        lambda artifacts: rendered_turns.append(artifacts.chat_turn),
    )

    app_chat.render_history()

    assert rendered_turns == []
    assert fake_st.calls == [
        ("chat_message", "user", None),
        ("markdown", "Question only", {}),
    ]
