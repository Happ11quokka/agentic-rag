import time
from unittest.mock import patch
from measurement.metrics_collector import MetricsCollector
from sweep import agent_runner
from sweep.agent_runner import run_one

# Reuse fake server from Task 17 test
from tests.measurement.test_metrics_collector import _serve

def _fake_dispatch(agent_type, sample, agent_kwargs):
    # Simulate one LLM call + one tool call via the callback handler
    from measurement.chat_wrapper import TRACE
    qt = TRACE.get()
    from measurement.trace_schema import LLMCallSpan, ToolCallSpan
    t = time.perf_counter()
    qt.llm_calls.append(LLMCallSpan(t_start=t, t_end=t+0.5, tokens_in=200, tokens_out=50))
    qt.tool_calls.append(ToolCallSpan(t_start=t+0.5, t_end=t+0.6, tool_name="wiki"))
    time.sleep(0.6)
    return {"answer": "Paris", "raw_messages": []}

def test_run_one_records_full_trace(monkeypatch):
    srv = _serve()
    port = srv.server_address[1]
    coll = MetricsCollector(f"http://127.0.0.1:{port}", interval_s=0.05).start()

    monkeypatch.setattr(agent_runner, "_dispatch", _fake_dispatch)
    monkeypatch.setattr(agent_runner, "load_sample",
                        lambda **_: {"_id": "x", "question": "Q?", "answer": "Paris"})

    qt = run_one(
        agent_type="react", fewshot=5, iteration_limit=30,
        sample_idx=0, collector=coll, sample_seed=42,
    )
    coll.stop()

    assert qt.correct is True
    assert qt.final_answer == "Paris"
    assert qt.n_llm_calls == 1
    assert qt.n_tool_calls == 1
    assert qt.e2e_latency_s > 0.5
    assert qt.tokens_input_total == 200
    assert qt.tokens_output_total == 50

def test_run_one_handles_dispatch_exception(monkeypatch):
    srv = _serve()
    port = srv.server_address[1]
    coll = MetricsCollector(f"http://127.0.0.1:{port}", interval_s=0.05).start()

    def _boom(*a, **kw): raise RuntimeError("agent crashed")
    monkeypatch.setattr(agent_runner, "_dispatch", _boom)
    monkeypatch.setattr(agent_runner, "load_sample",
                        lambda **_: {"_id": "x", "question": "Q", "answer": "Paris"})

    qt = run_one(agent_type="react", fewshot=5, iteration_limit=30,
                 sample_idx=0, collector=coll, sample_seed=42)
    coll.stop()
    assert qt.correct is False
    assert qt.meta.get("error", "").startswith("agent crashed") or qt.final_answer == "<ERROR>"
