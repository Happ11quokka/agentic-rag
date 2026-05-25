# HotpotQA Reproduction on M3 Pro — Design Spec

**Date**: 2026-05-25
**Status**: Design (awaiting review + implementation plan)
**Owner**: imdonghyeon
**Reference paper**: Kim et al., *The Cost of Dynamic Reasoning* (arXiv 2506.04301v2, HPCA-2026)
**Reference code**: <https://github.com/VIA-Research/AgentBench>

---

## 1. Context

This spec defines how to locally reproduce a subset of the HotpotQA experiments from the KAIST paper *The Cost of Dynamic Reasoning*. The reproduction supports the user's downstream decode-RAG research by providing a measurement harness that captures the same agent-level cost structure the paper analyzes (LLM/tool call counts, Prefill/Decode/Tool/Idle breakdown, KV cache pressure), even though absolute numbers cannot match the paper because the hardware and serving stack are different.

**Why this matters**: The paper's open-source repo (`VIA-Research/AgentBench`) contains the agent logic but **none of the measurement infrastructure** (verified via direct GitHub inspection — `DCGM`, `NVML`, `nvidia-smi`, `pynvml`, `poisson` all return 0 hits in code search). The user must build the measurement harness themselves; the question is whether the local hardware allows enough fidelity to validate the paper's *patterns* (idle %, diminishing returns, etc.) so that decode-RAG improvements can be measured against this baseline later.

---

## 2. Goal

Validate the paper's qualitative findings on HotpotQA using a Mac-local stack, while producing a reusable per-query trace pipeline that the user's later decode-RAG implementation can plug into for A/B comparison.

**Concretely, reproduce the *shape* of:**
- Fig 4 (per-request LLM/tool call counts)
- Fig 7 (95th-percentile latency distribution)
- Fig 13 (accuracy-vs-latency Pareto across 4 agents)
- Fig 14 (iteration-budget sweep — mean vs 95th-percentile latency)
- Fig 15 (few-shot count sweep)

**Not in scope** (justified in §9):
- Llama-3.1-70B (RAM ceiling)
- Serving experiments (Fig 10–12)
- DCGM-equivalent GPU util %
- Other benchmarks (WebShop, MATH, HumanEval)
- vLLM-specific phenomena (LLMCompiler overlap %)
- Replicating paper's absolute numbers (different GPU + backend)

---

## 3. Constraints

### 3.1 Hardware
- Apple M3 Pro, 12 CPU cores (6P+6E), 18 GPU cores
- 36 GB unified memory
- ~28 GB available disk after cleanup (2026-05-25)

### 3.2 Software stack
- **LLM serving**: llama.cpp built from source at a pinned commit (`repro/setup/LLAMACPP_COMMIT` records the tag/SHA). Metal backend. `brew install llama.cpp` is acceptable for Phase 0 smoke only; Phases 2–4 must use the pinned-source build.
- **Model**: Llama-3.1-8B-Instruct, Q4_K_M GGUF (~5 GB). Q8_0 (~9 GB) is a fallback if Phase 3 shows Q4 distorting figure shapes (especially for LATS — see §11).
- **Agent harness**: AgentBench at pinned commit (`repro/setup/AGENTBENCH_COMMIT`). Patches tracked in `repro/patches/*.patch`, applied via `git apply` during setup:
  - `config.patch` — `config.yaml`: model, host, port, `samples: 1`, `shuffle: false`, `temperature: 0.0`
  - `deterministic_select.patch` — `src/utils.py::load_dataset` (the only `random.shuffle` call site for HotpotQA): replace with `random.Random(sample_seed).shuffle(indices)` + explicit index selection
  - `entry_points.patch` — extract per-query entry functions from `run_react.py` / `run_reflexion.py` / `run_lats.py` / `run_llmcompiler.py` so we can call them directly (see §5.1)
  - `tool_retry.patch` (optional, applied if Phase 3 shows Wikipedia 429s) — `src/tools/hotpotqa_tools/wikipedia.py` wrapped in `tenacity` retry
- **Python**: pinned to 3.13.x via `pyenv` (record exact patch in `repro/setup/PYTHON_VERSION`). Python 3.13 not on macOS by default. `random.Random.shuffle()` output stability is only guaranteed *within* a Python version — pinning is required for deterministic sample selection (§8.2).
- **Extra Python deps not in AgentBench's `requirements.txt`**: `tenacity` (retry), `pydantic` (already in LangChain deps), `pandas`, `matplotlib`, `scipy` (Spearman ρ for §12).
- **Key library versions** (from AgentBench `requirements.txt`): LangChain 1.0.5, LangGraph 1.0.3, langchain-openai 1.0.2.

### 3.3 Time budget
- One-time setup: ~1 hour
- Smoke + mini-sweep validation: ~5 hours
- Full sweep (Phase 4): **~30-40 hours total** (across 2 nights). LATS Pareto cell alone is ~7h at 25 samples (paper reports 71 LLM calls/query × ~14s/call on Q4 Metal). See §8 for revised sweep matrix and §9 Phase 4 timing breakdown.

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ M3 Pro (36 GB unified memory)                               │
│                                                             │
│  ┌───────────────┐     ┌───────────────────────────────┐  │
│  │ Sweep Runner  │────▶│ AgentBench (unmodified logic) │  │
│  │ (Python)      │     │ - ReAct/Reflexion/LATS/       │  │
│  │ varies:       │     │   LLMCompiler                 │  │
│  │  - agent      │     │ - HotpotQA workload only      │  │
│  │  - fewshot    │     │ - LangChain → OpenAI API       │  │
│  │  - iter_limit │     └───────────────┬───────────────┘  │
│  └───────┬───────┘                     │ HTTP /v1/chat     │
│          ▲                             ▼                   │
│          │              ┌──────────────────────────────┐  │
│          │              │ llama-server (llama.cpp)     │  │
│          │              │ - Metal acceleration         │  │
│          │              │ - Llama-3.1-8B Q4_K_M        │  │
│          │              │ - --metrics  (Prefill/Decode)│  │
│          │              │ - --slots    (KV cache)      │  │
│          │              │ - --cache-reuse N            │  │
│          │              └──────────────┬───────────────┘  │
│          │                             │ polled @100ms    │
│          │              ┌──────────────▼───────────────┐  │
│          │              │ MetricsCollector             │  │
│          │              │ - diffs cumulative counters  │  │
│          │              │ - tags spans by request_id   │  │
│          │              │ - records KV cache size     │  │
│          │              └──────────────┬───────────────┘  │
│          │                             │                  │
│          │              ┌──────────────▼───────────────┐  │
│          └──────────────│ Per-query JSONL trace        │  │
│                         │ (one row per HotpotQA query) │  │
│                         └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Analysis      │
                  │ - Fig 4/7/13/ │
                  │   14/15 PNGs  │
                  └───────────────┘
```

### 4.1 Key design choices

| Choice | Reason |
|---|---|
| llama.cpp over Ollama/MLX-LM | Only backend exposing per-phase metrics (`/metrics`, `/slots`) without forking the engine |
| OpenAI-compatible endpoint at `/v1` | Drop-in for AgentBench's `langchain-openai` calls (one config patch only) |
| 100 ms polling interval | Fine enough to catch sub-second prefill/decode transitions; coarse enough to not pollute GPU. Edge case: LLM calls shorter than the interval get coarse attribution (`coarse_attribution=True`); see §7.4 |
| Per-query JSONL output (append-only, `fsync` per row) | Crash-resilient, trivially parseable; **enables `--resume` (see §9 Phase 4)** |
| Q4_K_M quantization | Best speed × accuracy tradeoff on Apple Silicon; ~5 GB fits comfortably. Q8_0 fallback for LATS if Q4 noise compounds over 71 LLM calls/query |
| Single slot only (no continuous batching) | `--parallel 1`. Single-request mode matches our experimental scope; serving deferred |
| `--cache-reuse` **OFF for measurement phases** | `prompt_seconds_total` behavior under cache reuse is undocumented in llama.cpp; we disable to guarantee clean per-call attribution. Wall-clock cost accepted. |
| `--seed N` set on llama-server | Metal at `temperature=0` is not bit-exact across runs without explicit seed (reduction-order non-determinism) |
| **`BaseCallbackHandler` for LLM instrumentation, NOT `ChatOpenAI` subclassing** | Two of four agents wrap `ChatOpenAI` in composition (LATS: `OpenAIChatClient`; LLMCompiler: `get_model()`), and ReAct calls `agent.stream()` not `invoke()`. A LangChain callback handler intercepts uniformly across sync/async/stream paths via `on_llm_start`/`on_llm_end` — works for all 4 agents without per-agent patching. (See §7.3.) |
| In-process execution (no subprocess) | Callback handler, polling thread, and trace context-var share memory; subprocess invocation would orphan them (see §5.1 contract) |

---

## 5. Repository layout

```
/Users/imdonghyeon/agentic_rag/
├── 2506.04301v2.pdf                ← existing (paper)
├── paper_analysis.md               ← existing
├── experiment_methodology.md       ← existing
├── docs/superpowers/specs/
│   └── 2026-05-25-hotpotqa-reproduction-design.md   ← THIS FILE
│
├── AgentBench/                     ← cloned, config.yaml only modified
│
└── repro/                          ← new — our reproduction code
    ├── README.md
    ├── models/
    │   └── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    ├── setup/
    │   ├── install_llamacpp.sh
    │   ├── LLAMACPP_COMMIT         ← pinned tagged commit
    │   ├── download_model.sh
    │   └── start_server.sh
    ├── patches/                     ← applied via git apply during setup
    │   ├── config.patch             ← AgentBench config.yaml
    │   └── deterministic_select.patch  ← shuffle: false + index-based select
    ├── measurement/
    │   ├── __init__.py
    │   ├── metrics_collector.py     ← polling thread, owns ring buffer
    │   ├── chat_wrapper.py          ← InstrumentedChat + per-query TRACE context
    │   ├── trace_schema.py          ← Pydantic models
    │   └── powermetrics_logger.py   (deferred, see §10)
    ├── sweep/
    │   ├── __init__.py
    │   ├── sweep_runner.py          ← top-level: parses YAML, owns lifecycle
    │   ├── agent_runner.py          ← one function: run_one(...) -> QueryTrace
    │   └── configs/
    │       ├── fig13_pareto.yaml
    │       ├── fig14_iteration.yaml
    │       └── fig15_fewshot.yaml
    ├── results/
    │   ├── raw/                     ← *.jsonl (append-only per run_id)
    │   └── aggregated/              ← *.csv
    └── analysis/
        ├── plot_fig4.py
        ├── plot_fig7.py
        ├── plot_fig13.py
        ├── plot_fig14.py
        ├── plot_fig15.py
        └── shared.py
```

### 5.1 Component contract (resolves sweep_runner ↔ agent_runner ↔ AgentBench)

**Everything runs in a single Python process.** No subprocess invocation of AgentBench.

#### 5.1.1 Per-agent entry points

AgentBench has **no uniform `.invoke()` interface**. The `entry_points.patch` (§3.2) extracts a per-query function from each `run_*.py`:

```python
# After applying entry_points.patch, these are importable:
from run_react       import run_single_query as react_run
from run_reflexion   import run_single_query as reflexion_run
from run_lats        import run_single_query as lats_run     # synchronous wrapper around arun
from run_llmcompiler import run_single_query as llmcompiler_run  # synchronous wrapper around arun

# Common signature:
#   def run_single_query(question: dict, agent_kwargs: dict) -> dict
# Returns: {"answer": str, "raw_messages": [...], "iterations": int, ...}
```

Each `run_single_query` takes one HotpotQA sample dict and agent-specific kwargs (`fewshot`, `iteration_limit`, etc.) — the patch wraps the existing per-sample loop body so we control sampling externally.

#### 5.1.2 LLM call interception via callback handler

```python
# measurement/chat_wrapper.py
class TraceCallbackHandler(BaseCallbackHandler):
    """LangChain callback. Works for sync/async/stream + all 4 agents."""
    def on_llm_start(self, serialized, prompts, *, run_id, **kw):
        TRACE.get().llm_calls.append(LLMCallSpan(
            t_start=time.perf_counter(), t_end=0.0,
            tokens_in=0, tokens_out=0,
            prefill_ms_estimate=0.0, decode_ms_estimate=0.0,
            run_id=str(run_id),
        ))
    def on_llm_end(self, response, *, run_id, **kw):
        span = next(s for s in TRACE.get().llm_calls if s.run_id == str(run_id))
        span.t_end = time.perf_counter()
        usage = getattr(response, "llm_output", {}) or {}
        token_usage = (usage.get("token_usage") or {})
        span.tokens_in  = token_usage.get("prompt_tokens", 0)
        span.tokens_out = token_usage.get("completion_tokens", 0)
    def on_llm_error(self, error, *, run_id, **kw):
        span = next(s for s in TRACE.get().llm_calls if s.run_id == str(run_id))
        span.t_end = time.perf_counter()
        span.error = str(error)[:200]
```

Registered per-run via `RunnableConfig` (preferred over global handlers in LangChain 1.0):

```python
# Pass via callbacks=[HANDLER] in agent_kwargs at each run_one invocation
# (see _dispatch in §5.1.3 — handler propagates through Runnable callback manager
# to every BaseChatModel and BaseTool dispatched downstream)
HANDLER = TraceCallbackHandler()
```

#### 5.1.3 `run_one` and `sweep_runner.main`

```python
# sweep/agent_runner.py
import contextvars
TRACE: contextvars.ContextVar[QueryTrace] = contextvars.ContextVar("TRACE")

def run_one(
    agent_type: Literal["react","reflexion","lats","llmcompiler"],
    fewshot: int,
    iteration_limit: int,
    sample_idx: int,
    *,
    collector: MetricsCollector,
    sample_seed: int = 42,
    timeout_s: float | None = None,    # 600 default; 1200 for LATS iteration_limit ≥ 50
) -> QueryTrace:
    qt = QueryTrace.bootstrap(agent_type=agent_type, fewshot=fewshot,
                              iteration_limit=iteration_limit, sample_idx=sample_idx)
    sample = load_sample(workload="hotpotqa", seed=sample_seed, idx=sample_idx)

    token = TRACE.set(qt)
    qt.t_start = time.perf_counter()
    try:
        future = THREAD_POOL.submit(_dispatch, agent_type, sample,
                                     {"fewshot": fewshot, "iteration_limit": iteration_limit,
                                      "callbacks": [HANDLER]})
        try:
            result = future.result(timeout=timeout_s or _default_timeout(agent_type, iteration_limit))
        except futures.TimeoutError:
            future.cancel()  # may not actually kill the request; see §9.5
            result = {"answer": "<TIMEOUT>", "timeout_reason": "wall_clock"}
            qt.meta["timeout"] = True
    finally:
        qt.t_end = time.perf_counter()
        TRACE.reset(token)

    qt.final_answer    = extract_final_answer(agent_type, result)
    qt.expected_answer = sample["answer"]
    qt.correct         = hotpotqa_em(qt.final_answer, qt.expected_answer)   # normalize_answer()
    finalize_trace(qt, collector.slice(qt.t_start, qt.t_end))
    return qt


# sweep/sweep_runner.py
def main(config_path: str, *, resume: bool = True):
    cfg = yaml.safe_load(open(config_path))
    out_path = f"results/raw/{cfg['run_id']}.jsonl"
    collector = MetricsCollector("http://localhost:8000").start()
    cells = list(enumerate_cells(cfg))
    if resume:
        done = read_done_tuples(out_path)   # tuple = (agent, sweep_var_val, sample_idx)
        cells = [c for c in cells if c.resume_key() not in done]
    for cell in cells:
        trace = run_one(**cell.as_kwargs(), collector=collector)
        append_jsonl_fsync(out_path, trace)   # f.flush(); os.fsync(f.fileno())
    collector.stop()
```

**Helper functions defined in `agent_runner.py`**:
- `_dispatch(agent_type, sample, kwargs)` — dispatches to `react_run` / `reflexion_run` / `lats_run` / `llmcompiler_run`
- `_default_timeout(agent, iter_limit)` — 600 for ReAct/Reflexion/LLMCompiler; 1200 for LATS at `iter_limit ≥ 50`
- `extract_final_answer(agent_type, result)` — per-agent extraction; see §5.2
- `hotpotqa_em(predicted, expected)` — exact match using HotpotQA's official `normalize_answer` (ported from `hotpot_evaluate_v1.py`):
  ```python
  def normalize_answer(s: str) -> str:
      import re, string
      s = s.lower()
      s = re.sub(r"\b(a|an|the)\b", " ", s)
      s = "".join(ch for ch in s if ch not in set(string.punctuation))
      s = " ".join(s.split())
      return s
  def hotpotqa_em(pred: str, gold: str) -> bool:
      return normalize_answer(pred) == normalize_answer(gold)
  ```
  AgentBench has `src/tools/hotpotqa_tools/hotpot_evaluate.py` — confirm whether their version matches before re-implementing. If matched, import from there.
- `append_jsonl_fsync(path, trace)` — `with open(path, "ab") as f: f.write(json+b"\n"); f.flush(); os.fsync(f.fileno())`
- `read_done_tuples(path)` — line-by-line `json.loads`, on malformed last line truncate at last full `\n`

**AgentBench's own `samples:` is pinned to `1` in `config.patch`** so the outer loop is always driven by `sweep_runner.py`. `enumerate_cells` returns a NamedTuple-like dataclass with explicit fields (`agent_type`, `fewshot`, `iteration_limit`, `sample_idx`, `sweep_var_name`, `sweep_var_val`) so `resume_key()` is well-defined.

### 5.2 Final-answer extraction (per agent)

| Agent | How `final_answer` is extracted |
|---|---|
| **ReAct** | Last `AIMessage` content in the LangGraph stream; if message text matches `Action: Finish[(.+?)]`, capture group 1. Else trailing AI content stripped. |
| **Reflexion** | Same as ReAct — Reflexion's outer loop calls the same `Finish[…]` action; reflection summaries discarded. |
| **LATS** | The `_best_solution()` aggregator in `src/agents/LATS/hotpotqa/` returns a `(answer, score)` tuple; take `answer`. |
| **LLMCompiler** | Return value of `await agent.arun(question)` — this is already the final string. |

These are implemented as `extract_final_answer(agent_type, result_dict)` in `agent_runner.py`.

### 5.3 Canonical llama-server startup

Single source of truth — referenced by §3, §4, §7, §9.5 (force-restart), §11:

```bash
# repro/setup/start_server.sh
llama-server \
    -m ~/agentic_rag/repro/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --host 127.0.0.1 --port 8000 \
    --metrics --slots \
    -c 32768 \
    --n-gpu-layers 999 \
    --parallel 1 \
    --seed 42 \
    --timeout 600
# NOTE: --cache-reuse is intentionally OMITTED for measurement phases 2-4 (see §4.1)
# Phase 1 smoke may add `--cache-reuse 256` for sanity but switch back before Phase 2.
```

Flag rationale:
- `-c 32768` — context window sized for LATS accumulated reasoning trees (~10× ReAct context)
- `--n-gpu-layers 999` — push all transformer layers to Metal
- `--parallel 1` — single slot for single-request mode (no continuous batching)
- `--seed 42` — token-tiebreak determinism at `temperature=0` (Metal reduction order varies otherwise)
- `--timeout 600` — server-side read/write timeout (default already 600; explicit for clarity); complements client-side `timeout=600` on `ChatOpenAI`

---

## 6. Trace schema (per query)

One JSON object per HotpotQA query, written to `results/raw/<run_id>.jsonl`.

```python
class QueryTrace(BaseModel):
    # Identity
    run_id: str                      # e.g., "fig14_iter50_react"
    query_id: str                    # UUID
    workload: Literal["hotpotqa"] = "hotpotqa"

    # Sweep variables
    agent_type: Literal["react", "reflexion", "lats", "llmcompiler"]
    fewshot: int
    iteration_limit: int
    sample_idx: int                  # 0..(n_samples-1)

    # Outcome
    correct: bool
    final_answer: str
    expected_answer: str

    # End-to-end latency
    e2e_latency_s: float

    # Wall-clock decomposition (top-level)
    llm_total_ms: float              # Σ(LLM call durations) from Python wrapper
    tool_total_ms: float             # Σ(tool durations)
    overhead_ms: float               # e2e - llm - tool

    # Server-side phase breakdown (from /metrics diff)
    prefill_total_ms: float
    decode_total_ms: float

    # Counters
    n_llm_calls: int
    n_tool_calls: int
    n_reflections: int = 0           # Reflexion only
    n_tree_expansions: int = 0       # LATS only

    # Tokens
    tokens_input_total: int
    tokens_output_total: int
    tokens_input_max: int            # peak context size in any single LLM call

    # Memory / KV cache (from /slots polling — n_prompt_tokens_cache field)
    kv_cache_max_tokens: int        # peak n_prompt_tokens_cache during query
    kv_cache_mean_tokens: float
    n_prompt_tokens_max: int        # peak n_prompt_tokens (total context, not just cached)

    # Energy (deferred infrastructure — see §10)
    gpu_avg_watts: Optional[float] = None
    gpu_total_wh: Optional[float] = None

    # Diagnostics
    meta: dict = {}                 # e.g. {"timeout": bool, "timeout_reason": str,
                                    #       "kv_eviction_detected": bool, ...}

    # Per-call detail (for Fig 5-style reconstruction)
    llm_calls: List[LLMCallSpan]
    tool_calls: List[ToolCallSpan]


class LLMCallSpan(BaseModel):
    t_start: float                   # perf_counter
    t_end: float
    prefill_ms_estimate: float       # bracket-attributed from /metrics diff
    decode_ms_estimate: float
    tokens_in: int                   # prompt_tokens from llm_output.token_usage
    tokens_out: int                  # completion_tokens
    coarse_attribution: bool = False # True when bracket samples unavailable
    run_id: str = ""                 # LangChain run UUID for correlation
    error: Optional[str] = None      # on_llm_error caught


class ToolCallSpan(BaseModel):
    t_start: float
    t_end: float
    tool_name: str
    error: Optional[str] = None      # tool exceptions captured
```

---

## 7. Measurement pipeline

### 7.1 llama-server data sources

**`/metrics`** (Prometheus text format, `--metrics` flag) — verified against llama.cpp `tools/server/README.md`:
- `llamacpp:prompt_seconds_total` — cumulative prefill time (s)
- `llamacpp:tokens_predicted_seconds_total` — cumulative decode time (s)
- `llamacpp:prompt_tokens_total` — cumulative prefill tokens
- `llamacpp:tokens_predicted_total` — cumulative decode tokens
- `llamacpp:n_decode_total` — cumulative decode iterations (sanity check)
- `llamacpp:n_tokens_max` — high-watermark of any single context's token count
- `llamacpp:requests_processing` — gauge of in-flight requests

All `_total` counters are cumulative since server start → **work in a window = diff between polls**.

> **NOT present**: there is no `llamacpp:kv_cache_tokens` or `llamacpp:kv_cache_used_cells` metric in current llama.cpp builds. **KV occupancy must come from `/slots` only.** (Verified against `tools/server/README.md` on master.)

**Cache-reuse caveat**: For measurement phases (2–4) we disable `--cache-reuse`. The behavior of `prompt_seconds_total` under cache reuse is not documented in the llama.cpp README; rather than rely on undocumented behavior, we disable to guarantee that every LLM call's prefill is fully attributable from the counter diff. Wall-clock cost accepted.

**`/slots`** (JSON, `--slots` flag) — verified against `server-context.cpp` `to_json()` output:

Actual per-slot fields:
- `id` (int) — slot index
- `is_processing` (bool) — true while the slot is generating
- `id_task` (int) — current task id (-1 when idle)
- `n_ctx` (int) — context window size
- `n_prompt_tokens` (int) — total tokens in current prompt
- `n_prompt_tokens_processed` (int) — tokens prefilled so far in current request
- `n_prompt_tokens_cache` (int) — tokens served from KV cache (i.e., NOT re-prefilled)
- `prompt` (str)
- `params` (dict), `next_token` (dict), `generated` (str)

We use `is_processing` for state (no `state` string field exists), `n_prompt_tokens_cache` for KV-reuse tracking, and `n_prompt_tokens_processed` as a sanity counter. `n_ctx` is constant across polls but useful for the trace.

### 7.2 Polling thread

**Lifecycle**: `MetricsCollector.start()` is called **once at sweep start** (by `sweep_runner.py`). It runs a background daemon thread polling both endpoints at 100 ms intervals into an in-memory ring buffer for the entire sweep duration. The deque is capped at ~50 hours of samples (1.8 M entries × ~80 bytes ≈ 144 MB upper bound). `collector.stop()` runs at sweep end. **It is NOT per-query.**

Per-query attribution happens at trace finalization (§7.4) by slicing the ring buffer between `query.t_start` and `query.t_end`.

Each sample:

```python
@dataclass(slots=True, frozen=True)
class PollSample:
    t: float                              # perf_counter
    # from /metrics (all cumulative since server start)
    prefill_s_total: float
    decode_s_total: float
    prefill_tokens_total: int
    decode_tokens_total: int
    n_decode_total: int
    n_tokens_max: int
    requests_processing: int
    # from /slots[0]  (single-slot config)
    is_processing: bool
    n_prompt_tokens: int
    n_prompt_tokens_processed: int
    n_prompt_tokens_cache: int            # KV-reuse tracking
```

`MetricsCollector` exposes `slice(t_start, t_end) -> List[PollSample]` that returns all samples in `[t_start, t_end]` plus the one immediately before `t_start` and the one immediately after `t_end` (boundary samples), to enable correct delta computation when no sample falls inside the window (see §7.4).

### 7.3 LLM call interception

**Use `TraceCallbackHandler` (defined in §5.1.2), NOT `ChatOpenAI` subclassing.**

Rationale (failures of the subclass approach):
- ReAct calls `agent.stream(...)` — overriding `invoke()` records zero calls.
- LATS wraps `ChatOpenAI` in `OpenAIChatClient` (composition) — subclass not seen.
- LLMCompiler uses `get_model()` factory — subclass not seen.
- LLMCompiler uses `await agent.arun(...)` — sync `invoke()` never called.

The callback handler approach (`on_llm_start` / `on_llm_end` / `on_llm_error`) intercepts at the LangChain framework level, **uniformly across sync/async/stream code paths and all 4 agents**, with no per-agent patching beyond passing `callbacks=[HANDLER]` in `agent_kwargs`.

**Defensive coding in `on_llm_end`**:
- `response.llm_output` may be `None` (some routes don't populate it) → use `getattr(..., "llm_output", None) or {}`.
- `token_usage` may be `None` → `(usage.get("token_usage") or {})`.
- Token field names are `prompt_tokens` / `completion_tokens` in the OpenAI raw API response shape (what `llm_output` carries), not `input_tokens` / `output_tokens` (those are `usage_metadata` on `AIMessage`).

**Trace context scoping**: `TRACE` is a `contextvars.ContextVar[QueryTrace]`. `run_one` sets it before dispatch and resets after (`token = TRACE.set(qt); ... TRACE.reset(token)`). Multi-threaded watchdog (§9.5) inherits the same context.

### 7.3.1 Tool call interception

For ReAct/Reflexion (LangGraph): tool nodes register their own callbacks (`on_tool_start`/`on_tool_end`). `TraceCallbackHandler` implements these to append `ToolCallSpan` entries.

For LATS/LLMCompiler: tools are invoked through the same LangChain `BaseTool` interface, so the same callback fires. Verified by reading `src/tools/hotpotqa_tools/wikipedia.py` which uses `BaseTool._run/_arun`.

### 7.4 Trace finalization

At query end, for each `LLMCallSpan` use **bracketing samples** (not in-window samples) for correct delta computation:

```python
def attribute(span: LLMCallSpan, samples: list[PollSample]) -> tuple[float, float, bool]:
    # samples is collector.slice(span.t_start, span.t_end)
    # which includes one boundary sample before t_start and one after t_end
    before = max((s for s in samples if s.t <= span.t_start), key=lambda s: s.t, default=None)
    after  = min((s for s in samples if s.t >= span.t_end),   key=lambda s: s.t, default=None)
    if before is None or after is None:
        return (0.0, 0.0, True)   # coarse_attribution=True
    prefill_ms = (after.prefill_s_total - before.prefill_s_total) * 1000.0
    decode_ms  = (after.decode_s_total  - before.decode_s_total ) * 1000.0
    return (max(prefill_ms, 0.0), max(decode_ms, 0.0), False)

# Caller writes result back to span:
for span in qt.llm_calls:
    pms, dms, coarse = attribute(span, collector.slice(span.t_start, span.t_end))
    span.prefill_ms_estimate = pms
    span.decode_ms_estimate  = dms
    span.coarse_attribution  = coarse
```

Bracketing handles three edge cases:
- **Call shorter than polling interval (< 100 ms)**: zero in-window samples but bracketing samples exist → correct attribution.
- **Polling thread paused (rare)**: `before` or `after` may be missing → `coarse_attribution=True`, both fields set to 0.0, flag carried in `LLMCallSpan.coarse_attribution`.
- **Counter rollover (extremely rare, >2^63 ns)**: `max(..., 0.0)` guards against negative deltas.

At the query level:
```python
e2e_latency_s    = query.t_end - query.t_start
llm_total_ms     = sum(span.t_end - span.t_start for span in llm_calls) * 1000
tool_total_ms    = sum(span.t_end - span.t_start for span in tool_calls) * 1000
overhead_ms      = max(e2e_latency_s * 1000 - llm_total_ms - tool_total_ms, 0.0)
prefill_total_ms = sum(span.prefill_ms_estimate for span in llm_calls)
decode_total_ms  = sum(span.decode_ms_estimate  for span in llm_calls)

kv_cache_max_tokens = max(s.n_prompt_tokens_cache for s in samples_in_query) if samples else 0
kv_cache_mean_tokens = mean(s.n_prompt_tokens_cache for s in samples_in_query) if samples else 0.0
n_prompt_tokens_max = max(s.n_prompt_tokens for s in samples_in_query) if samples else 0
```

**The "GPU idle %" analog** (Fig 6 in the paper) is computable as:
```
idle_ratio = 1 - (prefill_total_ms + decode_total_ms) / (e2e_latency_s * 1000)
```

This measures "fraction of e2e time the LLM was NOT actively computing." It naturally encompasses tool wait, Python orchestration overhead, **and** intra-call decode bubbles (e.g., async I/O during streaming). It is **not** identical to NVIDIA's DCGM-reported SM activity %:

- The paper's DCGM number measures hardware-level SM occupancy. Our `idle_ratio` measures *whether any LLM work happened*, not how saturated the GPU was during LLM work.
- Apple GPU has no DCGM equivalent. For paper-pattern validation (Fig 6 shows idle_ratio > 0.5 on HotpotQA), our proxy suffices because the dominant contributor is tool wait, which our formula captures.

The earlier draft of this spec used `(tool_total_ms + overhead_ms) / e2e` — that formula conflated tool wait with framework overhead and was rejected as incompatible with LLMCompiler's parallel-tool overlap. The corrected formula above derives idle from the inverse of measured LLM work, making it backend-agnostic.

---

## 8. Sweep matrix

| Experiment | Agents | Sweep variable | Values | Samples/cell | Total runs | Est. wall-clock |
|---|---|---|---|---|---|---|
| Fig 13 (Pareto) | ReAct, Reflexion, LLMCompiler | none (default config) | — | 50 | 150 | ~5h |
| Fig 13 (Pareto, LATS) | LATS | none | — | **25** | 25 | ~7h |
| Fig 14 (iteration sweep) | ReAct | `iteration_limit` | [5, 10, 15, 20, 30, 50, 75] | 50 | 350 | ~12h |
| Fig 15 (few-shot sweep) | ReAct | `fewshot` | [0, 1, 2, 3, 4, 5] | 50 | 300 | ~10h |
| | | | | **Total** | **825** | **~34h** |

**LATS sample reduction rationale**: paper reports ~71 LLM calls/query for LATS on HotpotQA. At Q4_K_M on M3 Pro Metal (~30-50 tok/s decode, ~500 output tokens/call), each call ≈ 12-17s. 71 calls × 14s ≈ 17 min/query. 25 samples × 17 min ≈ 7h, vs 14h for 50 samples. Cuts Phase 4 by 7h while keeping LATS visible on Fig 13.

If Phase 3 extrapolation shows the full sweep > 40h, further reduce ReAct sweeps to 25 samples (Fig 14/15 → ~325 runs, ~11h total).

### 8.1 Sweep configuration format

`sweep/configs/fig14_iteration.yaml`:
```yaml
run_id: fig14_iteration_sweep
workload: hotpotqa
agent_type: react
defaults:
  fewshot: 5
  iteration_limit: 30
sweeps:
  iteration_limit: [5, 10, 15, 20, 30, 50, 75]
samples_per_cell: 50    # authoritative — see §5.1 contract
sample_seed: 42         # see §8.2 for how this is actually applied
```

**Precedence**: `samples_per_cell` here is authoritative. AgentBench's own `samples:` key is pinned to `1` in `config.patch`. The outer loop is always driven by `sweep_runner.py`.

### 8.2 Deterministic HotpotQA selection

The naive plan ("set numpy seed and take first N from a shuffled dataset") fails because AgentBench may use Python's `random.shuffle` or HuggingFace `dataset.shuffle` — neither honors `numpy.random.seed()`.

**Concrete approach**:

1. **Patch AgentBench** (`patches/deterministic_select.patch`) to set `shuffle: false` and replace its sample-selection logic with:
   ```python
   import random
   ds = load_hotpotqa()  # whatever AgentBench's loader returns
   rng = random.Random(sample_seed)        # Python stdlib, fully reproducible
   shuffled_indices = list(range(len(ds)))
   rng.shuffle(shuffled_indices)
   selected = [ds[i] for i in shuffled_indices[:samples_per_cell]]
   ```
2. Phase 1 smoke test verifies that two independent runs with `sample_seed=42, samples_per_cell=5` select the **exact same 5 query IDs**.
3. The same sample set is reused across all sweep cells of a given experiment → variance comes from agent behavior, not data drift.

If AgentBench's loader exposes `.shuffle(seed=N)` (HuggingFace `Dataset`), prefer that built-in over manual indexing. The patch documents which path was taken.

---

## 9. Execution phases & verification gates

| Phase | What | Duration | Verification gate |
|---|---|---|---|
| **0. Setup** | Install llama.cpp, download model, clone AgentBench, apply patches | ~1h | `llama-server --version`, model file exists, `python agent_bench.py --help`, `patches/` cleanly applies |
| **1. Smoke** | One ReAct query end-to-end + deterministic-select verification | ~30m | `/metrics` and `/slots` respond; one trace JSONL row well-formed; `correct` ∈ {true, false}; two `--seed 42 --samples 5` runs select identical query IDs (§8.2) |
| **2. Pipeline validation** | One run with measurement layer + idle_ratio sanity | ~1-2h | All schema fields populated; `n_llm_calls ≥ 1`; `tokens_output_total > 0`; `kv_cache_max_tokens > 0`; `idle_ratio ∈ [0, 1]`. (`prefill_total_ms = 0` is **acceptable** if all calls were < polling interval — `coarse_attribution` flag captures this) |
| **3. Mini sweep** | All sweep configs at 5 samples/cell + extrapolate Phase 4 timing | ~3-5h | Every cell completes within timeout; row count = expected; plot scripts produce non-empty figures; resume from a deliberately-killed run skips done tuples; **measured per-agent latency × 50 samples extrapolates to Phase 4 budget (target ≤ 40h)** |
| **4. Full sweep** | Bump to 50 samples/cell (25 for LATS — see §8), overnight × 2 nights, `--resume` enabled | **~30-40h total** | 825 unique tuples in JSONL (see §8 revised); zero unhandled exceptions in log; ≤ 5% of queries marked timeout |
| **5. Analysis** | Generate Fig 4/7/13/14/15 PNGs | ~1-2h | All numeric criteria in §12 met |

### 9.4 Phase 4: Resume & checkpointing

- `--resume` (default ON) reads existing `results/raw/<run_id>.jsonl` files and skips any `(agent_type, sweep_var_value, sample_idx)` tuple already present.
- JSONL is append-only and `fsync`'d after each row → safe across crashes, machine sleeps, manual restarts.
- No separate checkpoint file. The JSONL itself is the checkpoint.
- `--no-resume` requires explicit confirmation (CLI prompt) before truncating existing JSONL.
- The polling thread is restarted from scratch on each invocation; the ring buffer holds only in-memory samples for the current process lifetime. This is correct because per-query attribution always uses *current-run* poll samples.

### 9.5 Diagnostics: hanging queries

A "hanging query" is operationally defined as any of:
1. **Wall-clock timeout**: `run_one` exceeds **600 seconds** (1200s for LATS at `iteration_limit ≥ 50`). Enforced by `concurrent.futures.ThreadPoolExecutor.submit(...).result(timeout=N)`.
2. **No decode progress** (detector in collector thread): `decode_tokens_total` does not increase across **30 consecutive polls** (3.0 s) while `/slots[0].is_processing == True`.
3. **No state change**: slot stays `is_processing=True` for **60 s** with no token-counter movement and no prefill progress.

**Cancellation mechanism** (langchain has no "cancellation token" — corrected from v1):
- LLM call timeout is set at the `ChatOpenAI` constructor: `request_timeout=600` (propagates to OpenAI `httpx` client which kills the socket on timeout).
- `concurrent.futures.Future.cancel()` is called on the outer dispatch future, but **may not abort** an in-flight LLM call until the socket times out. We accept this — at worst the query consumes its wall-clock budget then the watchdog reaps it.
- After three consecutive query timeouts in the same cell, the watchdog **force-restarts `llama-server`** via `subprocess.run(["pkill", "llama-server"]); subprocess.Popen([...startup args...])` and waits for `/health` to return 200. This handles the case where a slot stays wedged.

On detection:
- Write a `QueryTrace` with `correct=False`, `final_answer="<TIMEOUT>"`, `e2e_latency_s=elapsed`, and `meta.timeout=True`, `meta.timeout_reason` set to one of the three causes above.
- Capture the last 100 poll samples and the most recent `/slots[0]` snapshot into `meta.timeout_context` for offline diagnosis.
- Continue the sweep; do not abort the whole run.

These traces are counted in the timeout budget (Gate 4: ≤ 5% timeout rate). Wikipedia tool failures that exhaust the tenacity retry budget (§11) are also counted in this 5% — there is a single combined budget.

---

## 10. Out of scope (deferred)

- **Llama-3.1-70B**: 36 GB RAM cannot host 70B even Q4 (~40 GB)
- **Serving experiments** (Fig 10, 11, 12): need Poisson driver + concurrent slot serving; defer until single-request results validated
- **WebShop / MATH / HumanEval**: scope explicitly narrowed to HotpotQA
- **DCGM-equivalent GPU util %**: Apple GPU has no DCGM analog; the `idle_ratio` from §7.4 is the closest proxy
- **vLLM LLMCompiler overlap visualization** (Fig 5 pink bars): llama.cpp single-slot does not produce equivalent overlap; the breakdown collapses to LLM/Tool/Overhead
- **Energy measurement**: `powermetrics` measures whole-system power, not GPU-isolated. `gpu_avg_watts` / `gpu_total_wh` columns remain in the schema (§6) so the harness can be extended later, but `powermetrics_logger.py` is **deferred infrastructure** — Phase 4 ships with these columns nullable, and no figure in §12 depends on them.
- **CoT agent**: AgentBench code has no separate CoT module; paper's CoT row is reproducible only by stripping ReAct's tool node, which is deferred

---

## 11. Known risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Wikipedia API rate-limits HotpotQA tool | Medium | High (LATS makes ~70 calls/query) | Cache responses per query; add retry-with-backoff in `wikipedia.py` |
| **Wikipedia network blips / outage during 14h run** | Medium | High | Each tool call wrapped in `tenacity` retry (5 tries, exponential backoff up to 60 s); on hard fail, marked `tool_error=True` and trace continues; Phase 4 budget treats these as timeouts in the 5% cap |
| **KV cache eviction silently affects `n_prompt_tokens_cache`** (LATS > 8k ctx) | Medium | Medium | Server started with `-c 32768`; collector watches for non-monotonic decreases of `/slots[0].n_prompt_tokens_cache` while `is_processing=True`; finalize marks `meta.kv_eviction_detected=True` if observed |
| **Q4_K_M numerical instability on long contexts** (LATS repetition loops, compounded across 71 LLM calls/query) | High | High | Per-call decode-token cap (`max_tokens=2048`) on every `ChatOpenAI` constructor; diagnostic in §9.5 catches no-progress; **§12 Fig 13 LATS criterion is intentionally relaxed** (`accuracy[lats] ≥ 0.9 × accuracy[react]`) to absorb Q4 noise; Q8_0 fallback for LATS only if criterion still fails on Phase 3 mini-sweep |
| **llama-server slot OOM at high `n_ctx`** | Low | High | `-c 32768` sized to fit 8B Q4 + KV in 36 GB RAM; pre-Phase-3 stress test runs LATS with `iteration_limit=128` on a single sample to probe |
| LATS 50-sample run exceeds budget at default `iteration_limit: 20` | Medium | Medium | Phase 3 (mini sweep) reveals timing; reduce `iteration_limit` or sample count if needed |
| Polling thread misses sub-100ms phase transitions | Low | Low | Aggregated counters are still accurate; only fine-grained span attribution suffers |
| `Q4_K_M` accuracy degradation vs FP16 distorts Fig 13 shape | Medium | Medium | Q8_0 (~9 GB) fallback available if Q4 patterns look wrong in Phase 3 |
| llama.cpp `/metrics` endpoint format changes across versions | Low | Medium | Build from source at pinned commit (`repro/setup/LLAMACPP_COMMIT`); brew bottle is smoke-only |

---

## 12. Verification criteria (definition of done)

Phase 5 (analysis) produces 5 PNG figures. Each figure must satisfy:

| Figure | Acceptance criterion (numeric, testable) |
|---|---|
| **Fig 4** | `mean(n_llm_calls[lats]) / mean(n_llm_calls[react]) ≥ 5.0` (paper reports 71/8 ≈ 9; we accept ≥5 to allow Q4 variance) |
| **Fig 7** | `p95(e2e_latency_s[react]) / p50(e2e_latency_s[react]) ≥ 2.0` on HotpotQA |
| **Fig 13** | Spearman ρ between `mean_e2e_latency` and `accuracy` ≥ 0.6 across the 4 agents; **`accuracy[lats] ≥ 0.9 × accuracy[react]`** (Q4 noise compounding over 71 calls/query expected) AND `mean_e2e_latency[lats] > mean_e2e_latency[react]` (LATS at upper-right of Pareto) |
| **Fig 14** | `mean_e2e_latency` curve plateaus: max increase between consecutive iteration_limit values in the upper half of the sweep is < 25% of the increase in the lower half. `p95_e2e_latency` increases monotonically (Spearman ρ with iteration_limit ≥ 0.8) |
| **Fig 15** | Accuracy is **not monotone increasing** over fewshot ∈ [0..5]: either `accuracy[5] < max(accuracy[0..5])` OR the curve plateaus (slope < 0.5%-pt per added shot in upper half) |

If a criterion fails, that is **not** an immediate project failure — see §11. Most likely cause is one of: Q4 quantization noise (try Q8_0), insufficient samples (re-run with samples_per_cell=100 on the failing figure), or HotpotQA Wikipedia degradation. The criteria are designed to catch *qualitative regressions* (e.g., ReAct dominating LATS on accuracy), not minor numeric deviation.

Quantitative matching to paper numbers is **explicitly not required** (see §2, §10).

---

## 13. References

- Paper: `/Users/imdonghyeon/agentic_rag/2506.04301v2.pdf`
- Korean walkthrough: `/Users/imdonghyeon/agentic_rag/paper_analysis.md`
- Methodology comparison: `/Users/imdonghyeon/agentic_rag/experiment_methodology.md`
- AgentBench: <https://github.com/VIA-Research/AgentBench>
- llama.cpp server docs: <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- Llama-3.1-8B GGUF: <https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF>
