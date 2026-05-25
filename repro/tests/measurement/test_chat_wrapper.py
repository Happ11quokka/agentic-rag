import uuid
from unittest.mock import MagicMock

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from measurement.chat_wrapper import TraceCallbackHandler, TRACE
from measurement.trace_schema import QueryTrace

def _empty_trace() -> QueryTrace:
    return QueryTrace(
        run_id="t", query_id="q", agent_type="react", fewshot=0,
        iteration_limit=10, sample_idx=0, correct=False,
        final_answer="", expected_answer="", e2e_latency_s=0.0,
        llm_total_ms=0, tool_total_ms=0, overhead_ms=0,
        prefill_total_ms=0, decode_total_ms=0,
        n_llm_calls=0, n_tool_calls=0,
        tokens_input_total=0, tokens_output_total=0, tokens_input_max=0,
        kv_cache_max_tokens=0, kv_cache_mean_tokens=0, n_prompt_tokens_max=0,
        llm_calls=[], tool_calls=[],
    )

def test_llm_lifecycle_records_span():
    h = TraceCallbackHandler()
    qt = _empty_trace()
    token = TRACE.set(qt)
    try:
        run_id = uuid.uuid4()
        h.on_llm_start({"name": "ChatOpenAI"}, ["hello"], run_id=run_id)
        # Fake LLMResult with token usage
        result = MagicMock()
        result.llm_output = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        h.on_llm_end(result, run_id=run_id)
    finally:
        TRACE.reset(token)
    assert len(qt.llm_calls) == 1
    span = qt.llm_calls[0]
    assert span.tokens_in == 100
    assert span.tokens_out == 50
    assert span.t_end > span.t_start

def test_llm_error_records_error_span():
    h = TraceCallbackHandler()
    qt = _empty_trace()
    token = TRACE.set(qt)
    try:
        run_id = uuid.uuid4()
        h.on_llm_start({"name": "ChatOpenAI"}, ["hello"], run_id=run_id)
        h.on_llm_error(RuntimeError("boom"), run_id=run_id)
    finally:
        TRACE.reset(token)
    assert qt.llm_calls[0].error == "boom"

def test_handler_defensive_against_missing_usage():
    h = TraceCallbackHandler()
    qt = _empty_trace()
    token = TRACE.set(qt)
    try:
        run_id = uuid.uuid4()
        h.on_llm_start({"name": "ChatOpenAI"}, ["x"], run_id=run_id)
        result = MagicMock()
        result.llm_output = None  # some routes don't populate
        h.on_llm_end(result, run_id=run_id)
    finally:
        TRACE.reset(token)
    assert qt.llm_calls[0].tokens_in == 0
    assert qt.llm_calls[0].tokens_out == 0

def test_tool_lifecycle_records_span():
    h = TraceCallbackHandler()
    qt = _empty_trace()
    token = TRACE.set(qt)
    try:
        run_id = uuid.uuid4()
        h.on_tool_start({"name": "wikipedia"}, "search query", run_id=run_id)
        h.on_tool_end("result text", run_id=run_id)
    finally:
        TRACE.reset(token)
    assert len(qt.tool_calls) == 1
    assert qt.tool_calls[0].tool_name == "wikipedia"


def test_on_chat_model_start_populates_tokens_by_role():
    """Fig 8: on_chat_model_start classifies BaseMessages into role buckets."""
    h = TraceCallbackHandler()
    qt = _empty_trace()
    token = TRACE.set(qt)
    try:
        run_id = uuid.uuid4()
        messages = [[
            SystemMessage(content="you are a helpful assistant agent"),  # 6 words → system
            HumanMessage(content="What city is the Eiffel Tower in"),     # 7 words → human
            AIMessage(content="I should search Wikipedia for facts"),     # 6 words → ai
            ToolMessage(content="Paris France capital city", tool_call_id="tc1"),  # 4 → tool
        ]]
        h.on_chat_model_start({"name": "ChatOpenAI"}, messages, run_id=run_id)
    finally:
        TRACE.reset(token)

    # Per-role buckets must be populated.
    assert qt.tokens_by_role.get("system", 0) == 6
    assert qt.tokens_by_role.get("human", 0) == 7
    assert qt.tokens_by_role.get("ai", 0) == 6
    assert qt.tokens_by_role.get("tool", 0) == 4
    # A new LLM span was opened, just like on_llm_start.
    assert len(qt.llm_calls) == 1


def test_on_chat_model_start_accumulates_across_calls():
    """Subsequent on_chat_model_start calls add to existing role buckets."""
    h = TraceCallbackHandler()
    qt = _empty_trace()
    token = TRACE.set(qt)
    try:
        # First call: just human content (3 words).
        h.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[HumanMessage(content="first question text")]],
            run_id=uuid.uuid4(),
        )
        # Second call: more human content (2 words).
        h.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[HumanMessage(content="follow up")]],
            run_id=uuid.uuid4(),
        )
    finally:
        TRACE.reset(token)
    assert qt.tokens_by_role.get("human") == 5
    assert len(qt.llm_calls) == 2


def test_on_llm_start_fallback_parses_role_headers():
    """Fallback: legacy on_llm_start with serialized chat prompts."""
    h = TraceCallbackHandler()
    qt = _empty_trace()
    token = TRACE.set(qt)
    try:
        prompt = (
            "System: be helpful and concise\n"
            "Human: how many planets are there\n"
            "AI: there are eight planets\n"
            "Tool: lookup_planets returned eight"
        )
        h.on_llm_start({"name": "ChatOpenAI"}, [prompt], run_id=uuid.uuid4())
    finally:
        TRACE.reset(token)
    # All four roles should be populated (counts depend on whitespace splitting).
    assert qt.tokens_by_role.get("system", 0) > 0
    assert qt.tokens_by_role.get("human", 0) > 0
    assert qt.tokens_by_role.get("ai", 0) > 0
    assert qt.tokens_by_role.get("tool", 0) > 0


def test_tokens_by_role_serializes_roundtrip():
    """tokens_by_role must survive JSON serialization."""
    qt = _empty_trace()
    qt.tokens_by_role = {"system": 10, "human": 30, "ai": 20, "tool": 15}
    s = qt.model_dump_json()
    qt2 = QueryTrace.model_validate_json(s)
    assert qt2.tokens_by_role == {"system": 10, "human": 30, "ai": 20, "tool": 15}
