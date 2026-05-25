"""LangChain BaseCallbackHandler for uniform LLM/tool span instrumentation.

Works across ChatOpenAI (ReAct, Reflexion), OpenAIChatClient composition (LATS),
get_model() factory (LLMCompiler), and sync/async/stream code paths.

Per-query TRACE is scoped via contextvars.ContextVar -- `run_one` sets/resets it.
"""
import contextvars
import time
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

from .trace_schema import LLMCallSpan, QueryTrace, ToolCallSpan

TRACE: contextvars.ContextVar[QueryTrace] = contextvars.ContextVar("TRACE")


def _current_trace() -> Optional[QueryTrace]:
    try:
        return TRACE.get()
    except LookupError:
        return None


class TraceCallbackHandler(BaseCallbackHandler):
    """Records LLM and tool spans into the current per-query TRACE."""

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id, **kwargs) -> None:
        qt = _current_trace()
        if qt is None:
            return
        qt.llm_calls.append(LLMCallSpan(
            t_start=time.perf_counter(),
            t_end=0.0,
            tokens_in=0, tokens_out=0,
            prefill_ms_estimate=0.0, decode_ms_estimate=0.0,
            run_id=str(run_id),
        ))

    def on_llm_end(self, response: Any, *, run_id, **kwargs) -> None:
        qt = _current_trace()
        if qt is None:
            return
        span = next((s for s in qt.llm_calls if s.run_id == str(run_id)), None)
        if span is None:
            return
        span.t_end = time.perf_counter()
        # Defensive token extraction (per spec section 7.3).
        # Streaming ChatOpenAI puts usage on message.usage_metadata
        # (input_tokens/output_tokens) instead of llm_output['token_usage'].
        # Try llm_output first (non-streaming), then usage_metadata (streaming).
        tokens_in = 0
        tokens_out = 0
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or {}
        if usage:
            tokens_in = int(usage.get("prompt_tokens", 0) or 0)
            tokens_out = int(usage.get("completion_tokens", 0) or 0)
        if tokens_in == 0 and tokens_out == 0:
            generations = getattr(response, "generations", None) or []
            for batch in generations:
                for gen in batch:
                    msg = getattr(gen, "message", None)
                    um = getattr(msg, "usage_metadata", None) if msg else None
                    if not um:
                        continue
                    tokens_in += int(um.get("input_tokens", 0) or 0)
                    tokens_out += int(um.get("output_tokens", 0) or 0)
        span.tokens_in = tokens_in
        span.tokens_out = tokens_out

    def on_llm_error(self, error: BaseException, *, run_id, **kwargs) -> None:
        qt = _current_trace()
        if qt is None:
            return
        span = next((s for s in qt.llm_calls if s.run_id == str(run_id)), None)
        if span is None:
            return
        span.t_end = time.perf_counter()
        span.error = str(error)[:200]

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id, **kwargs) -> None:
        qt = _current_trace()
        if qt is None:
            return
        qt.tool_calls.append(ToolCallSpan(
            t_start=time.perf_counter(),
            t_end=0.0,
            tool_name=str(serialized.get("name", "unknown")),
        ))

    def on_tool_end(self, output: Any, *, run_id, **kwargs) -> None:
        qt = _current_trace()
        if qt is None or not qt.tool_calls:
            return
        # Match by ordering: complete the most recent unfinished tool span
        for span in reversed(qt.tool_calls):
            if span.t_end == 0.0:
                span.t_end = time.perf_counter()
                return

    def on_tool_error(self, error: BaseException, *, run_id, **kwargs) -> None:
        qt = _current_trace()
        if qt is None or not qt.tool_calls:
            return
        for span in reversed(qt.tool_calls):
            if span.t_end == 0.0:
                span.t_end = time.perf_counter()
                span.error = str(error)[:200]
                return
