"""Dispatch one sample to the appropriate AgentBench run_single_query.

Full `run_one` (with watchdog, trace finalization) added in Task 22.
"""
import os
import re
import sys
from typing import Any

AGENTBENCH_PATH = os.environ.get(
    "AGENTBENCH_PATH",
    "/Users/imdonghyeon/agentic_rag/AgentBench",
)
if AGENTBENCH_PATH not in sys.path:
    sys.path.insert(0, AGENTBENCH_PATH)


def _dispatch(agent_type: str, sample: dict, agent_kwargs: dict) -> dict:
    if agent_type == "react":
        from run_react import run_single_query
    elif agent_type == "reflexion":
        from run_reflexion import run_single_query
    elif agent_type == "lats":
        from run_lats import run_single_query
    elif agent_type == "llmcompiler":
        from run_llmcompiler import run_single_query
    else:
        raise ValueError(f"unknown agent_type: {agent_type}")
    return run_single_query(sample, agent_kwargs)


_FINISH_RE = re.compile(r"Action:\s*Finish\[(.+?)\]", re.DOTALL)


def extract_final_answer(agent_type: str, result: dict) -> str:
    """Extract the final answer string from a run_single_query result.

    ReAct/Reflexion: unwrap Action: Finish[...] if present, else use 'answer' as-is.
    LATS/LLMCompiler: use 'answer' field; just strip whitespace.
    """
    answer = (result.get("answer") or "").strip()
    if agent_type in ("react", "reflexion"):
        m = _FINISH_RE.search(answer)
        if m:
            return m.group(1).strip()
    return answer


def load_sample(*, workload: str, idx: int) -> dict:
    """Load one HotpotQA sample via AgentBench's patched load_dataset.

    AgentBench's load_dataset uses a relative path (dataset/...) so we chdir
    to AGENTBENCH_PATH for the duration of the call.
    """
    from src.utils import load_dataset
    prev_cwd = os.getcwd()
    try:
        os.chdir(AGENTBENCH_PATH)
        data = load_dataset(workload, shuffle=False)
    finally:
        os.chdir(prev_cwd)
    return data[idx]


import time
import uuid
import statistics
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from contextvars import copy_context

from measurement.chat_wrapper import TRACE, TraceCallbackHandler
from measurement.eval import hotpotqa_em
from measurement.metrics_collector import MetricsCollector, PollSample
from measurement.trace_schema import QueryTrace

HANDLER = TraceCallbackHandler()
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dispatch")

# Spec §9.5: force-restart llama-server after this many consecutive timeouts in a row.
_CONSECUTIVE_TIMEOUTS = 0
_FORCE_RESTART_THRESHOLD = 3


def _default_timeout(agent_type: str, iteration_limit: int) -> float:
    # LATS on M3 Pro Q4 needs ~18s/LLM call. Each LATS iteration triggers
    # ~3-5 LLM calls (expansion + evaluation + children).
    # Empirical: default 600s only finished ~33 of paper's 71 calls/query.
    # Budget formula: max(600, iteration_limit * 80) gives ~80s per LATS iter.
    if agent_type == "lats":
        return max(600.0, iteration_limit * 80.0)
    return 600.0


def _force_restart_llama_server() -> None:
    """Spec §9.5: kill and respawn llama-server when three consecutive queries time out.

    Called from run_one's exception path. Assumes start_server.sh is runnable from cwd.
    """
    import subprocess as _sp
    import requests as _rq
    import time as _t
    print("[watchdog] force-restarting llama-server", flush=True)
    _sp.run(["pkill", "-f", "llama-server"], check=False)
    _t.sleep(2)
    _sp.Popen(
        [os.path.join(os.path.dirname(__file__), "..", "setup", "start_server.sh")],
        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
    )
    # Wait for /health to return 200
    for _ in range(60):
        try:
            r = _rq.get("http://127.0.0.1:8000/health", timeout=1)
            if r.status_code == 200:
                return
        except Exception:
            pass
        _t.sleep(1)
    raise RuntimeError("llama-server did not come back within 60s of force-restart")


def _new_trace_skeleton(**fields) -> QueryTrace:
    """Build a QueryTrace with zero/empty defaults, ready to be filled in."""
    return QueryTrace(
        run_id=fields.get("run_id", ""),
        query_id=str(uuid.uuid4()),
        agent_type=fields["agent_type"],
        fewshot=fields["fewshot"],
        iteration_limit=fields["iteration_limit"],
        sample_idx=fields["sample_idx"],
        correct=False, final_answer="", expected_answer="",
        e2e_latency_s=0.0,
        llm_total_ms=0.0, tool_total_ms=0.0, overhead_ms=0.0,
        prefill_total_ms=0.0, decode_total_ms=0.0,
        n_llm_calls=0, n_tool_calls=0,
        tokens_input_total=0, tokens_output_total=0, tokens_input_max=0,
        kv_cache_max_tokens=0, kv_cache_mean_tokens=0.0, n_prompt_tokens_max=0,
        llm_calls=[], tool_calls=[],
    )


def _attribute(span, samples: list[PollSample]) -> tuple[float, float, bool]:
    before = max((s for s in samples if s.t <= span.t_start), key=lambda s: s.t, default=None)
    after = min((s for s in samples if s.t >= span.t_end), key=lambda s: s.t, default=None)
    if before is None or after is None:
        return (0.0, 0.0, True)
    prefill_ms = (after.prefill_s_total - before.prefill_s_total) * 1000.0
    decode_ms = (after.decode_s_total - before.decode_s_total) * 1000.0
    return (max(prefill_ms, 0.0), max(decode_ms, 0.0), False)


def finalize_trace(qt: QueryTrace, samples: list[PollSample]) -> None:
    """Fill in derived fields after the agent run completes.

    Precondition: qt.e2e_latency_s is already set by run_one before this call.
    """
    qt.n_llm_calls = len(qt.llm_calls)
    qt.n_tool_calls = len(qt.tool_calls)

    qt.llm_total_ms = sum((s.t_end - s.t_start) for s in qt.llm_calls) * 1000.0
    qt.tool_total_ms = sum((s.t_end - s.t_start) for s in qt.tool_calls) * 1000.0
    qt.overhead_ms = max(qt.e2e_latency_s * 1000.0 - qt.llm_total_ms - qt.tool_total_ms, 0.0)

    qt.tokens_input_total = sum(s.tokens_in for s in qt.llm_calls)
    qt.tokens_output_total = sum(s.tokens_out for s in qt.llm_calls)
    qt.tokens_input_max = max((s.tokens_in for s in qt.llm_calls), default=0)

    # Phase attribution from collector samples
    prefill_total = 0.0
    decode_total = 0.0
    for span in qt.llm_calls:
        pms, dms, coarse = _attribute(span, samples)
        span.prefill_ms_estimate = pms
        span.decode_ms_estimate = dms
        span.coarse_attribution = coarse
        prefill_total += pms
        decode_total += dms
    qt.prefill_total_ms = prefill_total
    qt.decode_total_ms = decode_total

    # KV cache from polling samples
    if samples:
        qt.kv_cache_max_tokens = max(s.n_prompt_tokens_cache for s in samples)
        qt.kv_cache_mean_tokens = statistics.mean(s.n_prompt_tokens_cache for s in samples)
        qt.n_prompt_tokens_max = max(s.n_prompt_tokens for s in samples)
        # Spec §11: detect KV eviction (non-monotonic cache_tokens during processing)
        prev = None
        evicted = False
        for s in samples:
            if prev is not None and s.is_processing and prev.is_processing:
                if s.n_prompt_tokens_cache < prev.n_prompt_tokens_cache:
                    evicted = True
                    break
            prev = s
        if evicted:
            qt.meta["kv_eviction_detected"] = True


def run_one(
    *,
    agent_type: str,
    fewshot: int,
    iteration_limit: int,
    sample_idx: int,
    collector: MetricsCollector,
    sample_seed: int = 42,
    run_id: str = "",
    extra_kwargs: dict = None,
    timeout_s: float = None,
) -> QueryTrace:
    """Run one HotpotQA query through one agent; return a fully-populated QueryTrace."""
    qt = _new_trace_skeleton(
        run_id=run_id, agent_type=agent_type, fewshot=fewshot,
        iteration_limit=iteration_limit, sample_idx=sample_idx,
    )
    os.environ["REPRO_SAMPLE_SEED"] = str(sample_seed)
    sample = load_sample(workload="hotpotqa", idx=sample_idx)
    qt.expected_answer = sample.get("answer", "")

    agent_kwargs = {
        "fewshot": fewshot,
        "iteration_limit": iteration_limit,
        "callbacks": [HANDLER],
    }
    if extra_kwargs:
        agent_kwargs.update(extra_kwargs)

    timeout_s = timeout_s or _default_timeout(agent_type, iteration_limit)

    global _CONSECUTIVE_TIMEOUTS
    token = TRACE.set(qt)
    t_start = time.perf_counter()
    try:
        ctx = copy_context()
        future = _POOL.submit(ctx.run, _dispatch, agent_type, sample, agent_kwargs)
        try:
            result = future.result(timeout=timeout_s)
            _CONSECUTIVE_TIMEOUTS = 0  # reset on success
        except FuturesTimeout:
            future.cancel()
            qt.meta["timeout"] = True
            qt.meta["timeout_reason"] = "wall_clock"
            result = {"answer": "<TIMEOUT>", "raw_messages": []}
            _CONSECUTIVE_TIMEOUTS += 1
            if _CONSECUTIVE_TIMEOUTS >= _FORCE_RESTART_THRESHOLD:
                # Spec §9.5: force-restart after 3 consecutive timeouts
                _force_restart_llama_server()
                _CONSECUTIVE_TIMEOUTS = 0
        except Exception as e:
            qt.meta["error"] = str(e)[:300]
            result = {"answer": "<ERROR>", "raw_messages": []}
    finally:
        t_end = time.perf_counter()
        TRACE.reset(token)

    qt.e2e_latency_s = t_end - t_start
    qt.final_answer = extract_final_answer(agent_type, result)
    qt.correct = hotpotqa_em(qt.final_answer, qt.expected_answer)

    finalize_trace(qt, collector.slice(t_start, t_end))
    return qt
