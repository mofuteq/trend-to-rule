"""Shared chat-turn runtime used by UI and API boundaries."""

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage

from core.app_config import AppConfig
from core.text_utils import normalize_text_nfkc
from services import tracing
from services.chat_workflow import AssistantResponseBundle, generate_assistant_response
from services.llm_client import (
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_OPENROUTER_OUTPUT_RETRIES,
    DEFAULT_OPENROUTER_REASONING_EFFORT,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)

WORKFLOW_VERSION = "v1"


class ChatTurnResult(BaseModel):
    """Result of a single shared chat workflow invocation."""

    chat_id: str
    user_id: str
    chat_turn: int
    normalized_prompt: str
    assistant_response: AssistantResponseBundle


def run_chat_turn(
    user_prompt: str,
    *,
    chat_id: str,
    user_id: str,
    config: AppConfig,
    chat_turn: int = 1,
    last_request_goal: str | None = None,
    history: list[ModelMessage] | None = None,
    thread_id: str | None = None,
) -> ChatTurnResult:
    """Run one chat turn through the existing LangGraph workflow.

    This function intentionally stays below the UI/API boundary: callers remain
    responsible for rendering, persistence, and transport-specific concerns.
    """
    normalized_prompt = normalize_text_nfkc(user_prompt)
    resolved_thread_id = thread_id or f"{chat_id}:{chat_turn}"
    tracing.update_current_trace(
        name=f"chat_turn_{chat_turn}",
        user_id=user_id,
        session_id=chat_id,
        input=normalized_prompt,
        tags=[
            *tracing.get_repoa_trace_tags(),
            "chat_turn",
            f"workflow:{WORKFLOW_VERSION}",
        ],
        metadata={
            **tracing.get_repoa_trace_metadata(),
            "chat_id": chat_id,
            "chat_turn": chat_turn,
            "workflow_version": WORKFLOW_VERSION,
        },
    )

    assistant_response = generate_assistant_response(
        user_prompt=normalized_prompt,
        config=config,
        last_request_goal=last_request_goal,
        history=history,
        thread_id=resolved_thread_id,
        langfuse_session_id=chat_id,
        langfuse_user_id=user_id,
    )
    _update_chat_turn_trace_output(
        assistant_response=assistant_response,
        chat_turn=chat_turn,
    )
    return ChatTurnResult(
        chat_id=chat_id,
        user_id=user_id,
        chat_turn=chat_turn,
        normalized_prompt=normalized_prompt,
        assistant_response=assistant_response,
    )


def _update_chat_turn_trace_output(
    *,
    assistant_response: AssistantResponseBundle,
    chat_turn: int,
) -> None:
    """Attach workflow output metadata to the active chat-turn trace."""
    request_analysis = assistant_response.request_analysis
    retrieval = assistant_response.retrieval
    assistant_rule = assistant_response.rule
    tracing.update_current_trace(
        output={
            "rule": assistant_rule,
            "request_goal": request_analysis.request_goal,
            "vertical": request_analysis.vertical,
            "is_in_scope": request_analysis.is_in_scope,
            "candidate_queries": request_analysis.candidate_queries.model_dump(),
            "structured_claims": (
                assistant_response.structured_claims.model_dump()
                if assistant_response.structured_claims is not None
                else None
            ),
            "structured_draft": (
                assistant_response.structured_draft.model_dump()
                if assistant_response.structured_draft is not None
                else None
            ),
            "image_query": assistant_response.image_query,
            "final_answer_reflection": (
                assistant_response.final_answer_reflection.model_dump()
                if assistant_response.final_answer_reflection is not None
                else None
            ),
            "final_answer_reflection_passed": (
                assistant_response.final_answer_reflection_passed
            ),
            "final_answer_reflection_failed_criteria": (
                assistant_response.final_answer_reflection_failed_criteria
            ),
            "final_answer_reflection_attempt_count": (
                assistant_response.final_answer_reflection_attempt_count
            ),
            "final_answer_reflection_failure_reason": (
                assistant_response.final_answer_reflection_failure_reason
            ),
            "image_results": [
                item.model_dump() for item in assistant_response.image_results
            ],
            "retrieval": {
                "canonical_sources": [
                    source.model_dump() for source in retrieval.canonical_sources
                ],
                "emerging_sources": [
                    source.model_dump() for source in retrieval.emerging_sources
                ],
                "error_message": retrieval.error_message,
            },
        },
        tags=[
            *tracing.get_repoa_trace_tags(),
            "chat_turn",
            f"workflow:{WORKFLOW_VERSION}",
            f"vertical:{request_analysis.vertical}",
            f"in_scope:{request_analysis.is_in_scope}",
        ],
        metadata={
            **tracing.get_repoa_trace_metadata(),
            "chat_turn": chat_turn,
            "request_goal": request_analysis.request_goal,
            "vertical": request_analysis.vertical,
            "is_in_scope": request_analysis.is_in_scope,
            "image_query": assistant_response.image_query,
            "image_result_count": len(assistant_response.image_results),
            "final_answer_reflection_passed": (
                assistant_response.final_answer_reflection_passed
            ),
            "final_answer_reflection_failed_criteria": (
                assistant_response.final_answer_reflection_failed_criteria
            ),
            "final_answer_reflection_attempt_count": (
                assistant_response.final_answer_reflection_attempt_count
            ),
            "final_answer_reflection_failure_reason": (
                assistant_response.final_answer_reflection_failure_reason
            ),
            "text_retrieval_backend": "tavily",
            "canonical_source_count": len(retrieval.canonical_sources),
            "emerging_source_count": len(retrieval.emerging_sources),
            "total_source_count": retrieval.total_source_count(),
            "canonical_claim_count": (
                len(assistant_response.structured_claims.canonical_claims)
                if assistant_response.structured_claims is not None
                else 0
            ),
            "emerging_claim_count": (
                len(assistant_response.structured_claims.emerging_claims)
                if assistant_response.structured_claims is not None
                else 0
            ),
            "model_params": {
                "model": DEFAULT_OPENROUTER_MODEL,
                "temperature": DEFAULT_TEMPERATURE,
                "top_p": DEFAULT_TOP_P,
                "seed": DEFAULT_SEED,
                "output_retries": DEFAULT_OPENROUTER_OUTPUT_RETRIES,
                "reasoning_effort": DEFAULT_OPENROUTER_REASONING_EFFORT,
            },
        },
    )

