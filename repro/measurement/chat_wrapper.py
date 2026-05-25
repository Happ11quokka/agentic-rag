"""LangChain BaseCallbackHandler for uniform LLM/tool span instrumentation.

Works across ChatOpenAI (ReAct, Reflexion), OpenAIChatClient composition (LATS),
get_model() factory (LLMCompiler), and sync/async/stream code paths.

Per-query TRACE is scoped via contextvars.ContextVar -- `run_one` sets/resets it.
"""
import contextvars
import re
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


# Mapping from LangChain BaseMessage class name -> role bucket for Fig 8.
_MESSAGE_CLASS_TO_ROLE = {
    "SystemMessage": "system",
    "SystemMessageChunk": "system",
    "HumanMessage": "human",
    "HumanMessageChunk": "human",
    "AIMessage": "ai",
    "AIMessageChunk": "ai",
    "ToolMessage": "tool",
    "ToolMessageChunk": "tool",
    "FunctionMessage": "tool",
    "FunctionMessageChunk": "tool",
    "ChatMessage": "human",  # generic, treat as human-ish input
}

# Fallback regex for parsing serialized chat prompts ("\nHuman: ", "\nAI: ", etc.).
_ROLE_HEADER_RE = re.compile(
    r"(?:^|\n)(System|Human|AI|Tool|Function)\s*:\s*", re.IGNORECASE
)
_HEADER_TO_ROLE = {
    "system": "system",
    "human": "human",
    "ai": "ai",
    "tool": "tool",
    "function": "tool",
}


def _word_count(s: Any) -> int:
    """Cheap token estimate: word count of str(content). Used for the role-bucket
    sum, which only needs to be a stable approximation (per Fig 8 in the paper)."""
    if s is None:
        return 0
    if not isinstance(s, str):
        s = str(s)
    return len(s.split())


def _classify_message(msg: Any) -> str:
    """Map a LangChain BaseMessage to a role bucket. Falls back to 'human'."""
    cls = type(msg).__name__
    if cls in _MESSAGE_CLASS_TO_ROLE:
        return _MESSAGE_CLASS_TO_ROLE[cls]
    # Some libraries expose .type ("system" | "human" | "ai" | "tool" | "function")
    msg_type = getattr(msg, "type", None)
    if isinstance(msg_type, str) and msg_type.lower() in _HEADER_TO_ROLE:
        return _HEADER_TO_ROLE[msg_type.lower()]
    return "human"


def _accumulate_role_tokens(qt: QueryTrace, role: str, n: int) -> None:
    if n <= 0:
        return
    qt.tokens_by_role[role] = qt.tokens_by_role.get(role, 0) + n


def _accumulate_from_prompt_string(qt: QueryTrace, prompt: str) -> None:
    """Fallback: parse a serialized chat prompt string into role-bucket counts.

    Looks for `\\nHuman: `, `\\nAI: `, `\\nSystem: `, `\\nTool: ` / `\\nFunction: `
    headers. If no headers are found, attributes the whole thing to "human".
    This is approximate — used only when on_chat_model_start isn't invoked
    (e.g. plain completion-style call paths).
    """
    matches = list(_ROLE_HEADER_RE.finditer(prompt))
    if not matches:
        _accumulate_role_tokens(qt, "human", _word_count(prompt))
        return
    for i, m in enumerate(matches):
        role = _HEADER_TO_ROLE.get(m.group(1).lower(), "human")
        seg_start = m.end()
        seg_end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        _accumulate_role_tokens(qt, role, _word_count(prompt[seg_start:seg_end]))


class TraceCallbackHandler(BaseCallbackHandler):
    """Records LLM and tool spans into the current per-query TRACE."""

    def on_chat_model_start(
        self, serialized: dict, messages: list, *, run_id, **kwargs
    ) -> None:
        """Chat-model variant: receives BaseMessage objects directly so we can
        classify input tokens by role for Fig 8.

        LangChain calls this (instead of on_llm_start) for ChatOpenAI-style
        models. `messages` is `list[list[BaseMessage]]` (one list per batch
        element); we iterate the outer list defensively.
        """
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
        # `messages` is normally list[list[BaseMessage]]; tolerate flat-list too.
        try:
            iterables = messages if (messages and isinstance(messages[0], list)) else [messages]
            for batch in iterables:
                for msg in batch:
                    role = _classify_message(msg)
                    content = getattr(msg, "content", "") or ""
                    _accumulate_role_tokens(qt, role, _word_count(content))
        except Exception:
            # tokens_by_role is best-effort; never break the wrapped agent.
            pass

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
        # Fallback path: classify by parsing the prompt string. LangChain only
        # invokes on_llm_start (not on_chat_model_start) for legacy completion
        # endpoints — tokens_by_role is still best-effort here.
        try:
            for prompt in prompts or []:
                if isinstance(prompt, str) and prompt:
                    _accumulate_from_prompt_string(qt, prompt)
        except Exception:
            pass

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
