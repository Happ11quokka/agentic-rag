from measurement.trace_schema import QueryTrace, LLMCallSpan, ToolCallSpan

def test_query_trace_roundtrip():
    qt = QueryTrace(
        run_id="test", query_id="q1", workload="hotpotqa",
        agent_type="react", fewshot=5, iteration_limit=30, sample_idx=0,
        correct=True, final_answer="X", expected_answer="X",
        e2e_latency_s=12.34, llm_total_ms=8000.0, tool_total_ms=4000.0,
        overhead_ms=340.0, prefill_total_ms=2000.0, decode_total_ms=6000.0,
        n_llm_calls=8, n_tool_calls=4,
        tokens_input_total=8000, tokens_output_total=2000, tokens_input_max=3000,
        kv_cache_max_tokens=500, kv_cache_mean_tokens=200.0, n_prompt_tokens_max=3000,
        llm_calls=[LLMCallSpan(t_start=0.0, t_end=1.0, prefill_ms_estimate=100,
                                decode_ms_estimate=900, tokens_in=500, tokens_out=200)],
        tool_calls=[ToolCallSpan(t_start=1.0, t_end=2.0, tool_name="search")],
    )
    s = qt.model_dump_json()
    qt2 = QueryTrace.model_validate_json(s)
    assert qt2 == qt

def test_query_trace_optional_fields():
    qt = QueryTrace(
        run_id="t", query_id="q", agent_type="react", fewshot=0,
        iteration_limit=10, sample_idx=0, correct=False,
        final_answer="", expected_answer="", e2e_latency_s=1.0,
        llm_total_ms=0, tool_total_ms=0, overhead_ms=0,
        prefill_total_ms=0, decode_total_ms=0,
        n_llm_calls=0, n_tool_calls=0,
        tokens_input_total=0, tokens_output_total=0, tokens_input_max=0,
        kv_cache_max_tokens=0, kv_cache_mean_tokens=0, n_prompt_tokens_max=0,
        llm_calls=[], tool_calls=[],
    )
    assert qt.gpu_avg_watts is None
    assert qt.meta == {}
