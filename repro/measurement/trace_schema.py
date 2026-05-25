from typing import Literal, Optional
from pydantic import BaseModel, Field


class LLMCallSpan(BaseModel):
    t_start: float
    t_end: float
    prefill_ms_estimate: float = 0.0
    decode_ms_estimate: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    coarse_attribution: bool = False
    run_id: str = ""
    error: Optional[str] = None


class ToolCallSpan(BaseModel):
    t_start: float
    t_end: float
    tool_name: str
    error: Optional[str] = None


class QueryTrace(BaseModel):
    # Identity
    run_id: str
    query_id: str
    workload: Literal["hotpotqa"] = "hotpotqa"

    # Sweep variables
    agent_type: Literal["react", "reflexion", "lats", "llmcompiler"]
    fewshot: int
    iteration_limit: int
    sample_idx: int

    # Outcome
    correct: bool
    final_answer: str
    expected_answer: str

    # End-to-end latency
    e2e_latency_s: float

    # Wall-clock decomposition
    llm_total_ms: float
    tool_total_ms: float
    overhead_ms: float

    # Server-side phase breakdown
    prefill_total_ms: float
    decode_total_ms: float

    # Counters
    n_llm_calls: int
    n_tool_calls: int
    n_reflections: int = 0
    n_tree_expansions: int = 0

    # Tokens
    tokens_input_total: int
    tokens_output_total: int
    tokens_input_max: int

    # Memory / KV cache
    kv_cache_max_tokens: int
    kv_cache_mean_tokens: float
    n_prompt_tokens_max: int

    # Energy (deferred)
    gpu_avg_watts: Optional[float] = None
    gpu_total_wh: Optional[float] = None

    # Diagnostics
    meta: dict = Field(default_factory=dict)

    # Per-call detail
    llm_calls: list[LLMCallSpan]
    tool_calls: list[ToolCallSpan]
