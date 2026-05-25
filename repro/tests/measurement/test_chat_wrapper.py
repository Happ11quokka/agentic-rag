import uuid
from unittest.mock import MagicMock
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
