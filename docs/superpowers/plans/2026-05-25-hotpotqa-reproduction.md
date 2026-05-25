# HotpotQA Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce 5 figures (Fig 4/7/13/14/15) from arXiv 2506.04301v2 (KAIST AgentBench paper) on HotpotQA locally with M3 Pro + llama.cpp, producing a per-query trace pipeline reusable by the user's later decode-RAG research.

**Architecture:** llama.cpp serves Llama-3.1-8B-Instruct Q4_K_M via OpenAI-compatible HTTP. A single Python process runs the sweep, using a LangChain `BaseCallbackHandler` (uniform across all 4 agents + sync/async/stream) to attribute LLM/tool spans, plus a 100ms polling thread that reads llama.cpp's `/metrics` and `/slots` endpoints for prefill/decode/KV breakdown. Each query becomes one JSONL row; the sweep is resumable via append-only JSONL.

**Tech Stack:** llama.cpp (Metal), Python 3.13 (pyenv), LangChain 1.0.5 / LangGraph 1.0.3 / langchain-openai 1.0.2, AgentBench (patched), pydantic, pandas, matplotlib, scipy, tenacity.

**Spec reference:** `/Users/imdonghyeon/agentic_rag/docs/superpowers/specs/2026-05-25-hotpotqa-reproduction-design.md`

**Working directory:** `/Users/imdonghyeon/agentic_rag/`

---

## File structure

After this plan, the repo looks like:

```
/Users/imdonghyeon/agentic_rag/
├── .git/                              ← Task 1 (git init)
├── 2506.04301v2.pdf                   (existing)
├── paper_analysis.md                  (existing)
├── experiment_methodology.md          (existing)
├── docs/superpowers/{specs,plans}/    (existing)
├── AgentBench/                        ← Task 4 (cloned at pinned commit)
└── repro/                             ← all new code below
    ├── README.md                      ← Task 33
    ├── pyproject.toml                 ← Task 7 (deps + tooling)
    ├── models/
    │   └── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  ← Task 3
    ├── setup/
    │   ├── LLAMACPP_COMMIT            ← Task 2
    │   ├── AGENTBENCH_COMMIT          ← Task 4
    │   ├── PYTHON_VERSION             ← Task 5
    │   ├── install_llamacpp.sh        ← Task 2
    │   ├── download_model.sh          ← Task 3
    │   └── start_server.sh            ← Task 6
    ├── patches/
    │   ├── config.patch               ← Task 8
    │   ├── deterministic_select.patch ← Task 9
    │   ├── entry_points.patch         ← Tasks 10-13 (one per agent file)
    │   └── tool_retry.patch           ← Task 26 Step 4 (conditional on Phase 3 results)
    ├── measurement/
    │   ├── __init__.py
    │   ├── trace_schema.py            ← Task 14 (Pydantic models)
    │   ├── eval.py                    ← Task 15 (normalize_answer)
    │   ├── metrics_collector.py       ← Tasks 16-17 (polling + parsing)
    │   └── chat_wrapper.py            ← Task 18 (TraceCallbackHandler)
    ├── sweep/
    │   ├── __init__.py
    │   ├── cells.py                   ← Task 19 (enumerate_cells, ResumeKey)
    │   ├── agent_runner.py            ← Tasks 20-22 (run_one + dispatch + extract)
    │   ├── sweep_runner.py            ← Task 23 (main loop + resume + fsync)
    │   └── configs/
    │       ├── fig13_pareto.yaml      ← Task 24
    │       ├── fig14_iteration.yaml   ← Task 24
    │       └── fig15_fewshot.yaml     ← Task 24
    ├── results/
    │   ├── raw/                       ← *.jsonl
    │   └── aggregated/                ← *.csv
    └── analysis/
        ├── shared.py                  ← Task 28
        ├── plot_fig4.py               ← Task 29
        ├── plot_fig7.py               ← Task 30
        ├── plot_fig13.py              ← Task 31
        ├── plot_fig14.py              ← Task 32
        └── plot_fig15.py              ← Task 32
```

Test files under `repro/tests/` mirror the structure with `test_*.py`.

---

## Phase 0 — Setup (Tasks 1–7)

### Task 1: Initialize git repo and create reproduction directory

**Files:**
- Create: `/Users/imdonghyeon/agentic_rag/.gitignore`
- Create: `/Users/imdonghyeon/agentic_rag/repro/` (empty dirs)

- [ ] **Step 1: Init git in agentic_rag**

```bash
cd /Users/imdonghyeon/agentic_rag
git init
git add 2506.04301v2.pdf paper_analysis.md paper_analysis.pdf experiment_methodology.md docs/
git commit -m "chore: import paper analysis and reproduction spec"
```

- [ ] **Step 2: Create `.gitignore`**

Write `/Users/imdonghyeon/agentic_rag/.gitignore`:
```
# Reproduction outputs
repro/models/*.gguf
repro/results/raw/*.jsonl
repro/results/aggregated/*.csv
repro/results/figures/*.png

# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/

# AgentBench clone
AgentBench/

# Misc
.DS_Store
*.log
trace.txt
```

- [ ] **Step 3: Create empty repro skeleton**

```bash
cd /Users/imdonghyeon/agentic_rag
mkdir -p repro/{setup,patches,measurement,sweep/configs,results/{raw,aggregated,figures},analysis,tests/{measurement,sweep,analysis,integration},models}
touch repro/measurement/__init__.py repro/sweep/__init__.py repro/analysis/__init__.py
# Test packages need __init__.py for cross-test imports (e.g., Task 22 reuses _serve from Task 17 test)
touch repro/tests/__init__.py
touch repro/tests/measurement/__init__.py
touch repro/tests/sweep/__init__.py
touch repro/tests/integration/__init__.py
touch repro/tests/analysis/__init__.py
```

- [ ] **Step 4: Commit skeleton**

```bash
git add .gitignore repro/measurement/__init__.py repro/sweep/__init__.py
git commit -m "chore: initialize repro/ skeleton"
```

---

### Task 2: Install llama.cpp from source at pinned commit

**Files:**
- Create: `repro/setup/LLAMACPP_COMMIT`
- Create: `repro/setup/install_llamacpp.sh`

- [ ] **Step 1: Resolve current stable llama.cpp tag**

Run:
```bash
gh api repos/ggml-org/llama.cpp/releases/latest --jq '.tag_name'
```
Expected: a tag like `b4321` or similar. Record this exact string.

- [ ] **Step 2: Write `repro/setup/LLAMACPP_COMMIT`**

Write the tag name resolved above to the file (one line, no trailing whitespace).

- [ ] **Step 3: Write `repro/setup/install_llamacpp.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="$(cat "$SCRIPT_DIR/LLAMACPP_COMMIT")"
SRC_DIR="$SCRIPT_DIR/llama.cpp.src"
INSTALL_DIR="$SCRIPT_DIR/llama.cpp.build"

if [[ ! -d "$SRC_DIR" ]]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$SRC_DIR"
fi
cd "$SRC_DIR"
git fetch --tags origin
git checkout "$TAG"

cmake -B "$INSTALL_DIR" -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$INSTALL_DIR" --config Release -j

echo "Built. Binary at: $INSTALL_DIR/bin/llama-server"
"$INSTALL_DIR/bin/llama-server" --version
```

- [ ] **Step 4: Run installer and verify**

```bash
chmod +x repro/setup/install_llamacpp.sh
./repro/setup/install_llamacpp.sh
```
Expected: builds without error; final line prints version string.

- [ ] **Step 5: Commit**

```bash
git add repro/setup/install_llamacpp.sh repro/setup/LLAMACPP_COMMIT
git commit -m "feat: pin and build llama.cpp from source"
```

---

### Task 3: Download Llama-3.1-8B-Instruct Q4_K_M GGUF

**Files:**
- Create: `repro/setup/download_model.sh`
- Output: `repro/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (~4.9 GB)

- [ ] **Step 1: Write `repro/setup/download_model.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
FILE="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

if [[ -f "$MODELS_DIR/$FILE" ]]; then
    echo "Already downloaded: $MODELS_DIR/$FILE"
    exit 0
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "Installing huggingface-hub CLI..."
    pip install --user 'huggingface_hub[cli]'
fi

huggingface-cli download \
    bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    "$FILE" \
    --local-dir "$MODELS_DIR"

ls -lh "$MODELS_DIR/$FILE"
```

- [ ] **Step 2: Run downloader**

```bash
chmod +x repro/setup/download_model.sh
./repro/setup/download_model.sh
```
Expected: ~4.9 GB file at `repro/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`.

- [ ] **Step 3: Sanity-load via llama-server**

```bash
./repro/setup/llama.cpp.build/bin/llama-server \
    -m repro/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -c 2048 --port 8000 &
SERVER_PID=$!
sleep 8
curl -s http://localhost:8000/v1/models | python3 -m json.tool
kill $SERVER_PID 2>/dev/null || true
```
Expected: JSON with `data[0].id` matching the model file name.

- [ ] **Step 4: Commit**

```bash
git add repro/setup/download_model.sh
git commit -m "feat: model download script"
```

---

### Task 4: Clone AgentBench at pinned commit

**Files:**
- Create: `repro/setup/AGENTBENCH_COMMIT`
- Output: `/Users/imdonghyeon/agentic_rag/AgentBench/` (cloned)

- [ ] **Step 1: Resolve current AgentBench HEAD SHA**

```bash
gh api repos/VIA-Research/AgentBench/commits/main --jq '.sha[:12]'
```
Expected: 12-char hex string. Record it.

- [ ] **Step 2: Write `repro/setup/AGENTBENCH_COMMIT`**

Write the SHA from Step 1 (one line).

- [ ] **Step 3: Clone and checkout pinned commit**

```bash
cd /Users/imdonghyeon/agentic_rag
git clone https://github.com/VIA-Research/AgentBench.git
cd AgentBench
git checkout "$(cat ../repro/setup/AGENTBENCH_COMMIT)"
git rev-parse HEAD
```
Expected: HEAD hash matches AGENTBENCH_COMMIT (12-prefix).

- [ ] **Step 4: Confirm key files**

```bash
ls AgentBench/run_react.py AgentBench/run_reflexion.py AgentBench/run_lats.py AgentBench/run_llmcompiler.py
ls AgentBench/config.yaml AgentBench/src/utils.py
ls AgentBench/src/agents/{ReAct,Reflexion,LATS,LLMCompiler}
ls AgentBench/dataset/hotpot_dev_fullwiki_v1.json
```
Expected: all files exist (no errors).

- [ ] **Step 5: Commit pinned-commit reference**

```bash
cd /Users/imdonghyeon/agentic_rag
git add repro/setup/AGENTBENCH_COMMIT
git commit -m "chore: pin AgentBench commit"
```

(AgentBench/ itself is gitignored — only the SHA is tracked.)

---

### Task 5: Set up Python 3.13 via pyenv

**Files:**
- Create: `repro/setup/PYTHON_VERSION`
- Create: `repro/.python-version`

- [ ] **Step 1: Check pyenv is installed**

```bash
which pyenv || brew install pyenv
pyenv --version
```
Expected: pyenv version string.

- [ ] **Step 2: Install Python 3.13.x**

```bash
pyenv install --skip-existing 3.13.0
pyenv versions
```
Expected: 3.13.0 in the list.

- [ ] **Step 3: Pin in `repro/`**

```bash
cd /Users/imdonghyeon/agentic_rag/repro
pyenv local 3.13.0
echo "3.13.0" > setup/PYTHON_VERSION
python3 --version
```
Expected: `Python 3.13.0`. The `.python-version` file is created automatically by `pyenv local`.

- [ ] **Step 4: Commit**

```bash
cd /Users/imdonghyeon/agentic_rag
git add repro/setup/PYTHON_VERSION repro/.python-version
git commit -m "chore: pin Python 3.13 for repro/"
```

---

### Task 6: Write canonical `start_server.sh`

**Files:**
- Create: `repro/setup/start_server.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_BIN="$SCRIPT_DIR/llama.cpp.build/bin/llama-server"
MODEL="$SCRIPT_DIR/../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

if [[ ! -x "$LLAMA_BIN" ]]; then
    echo "llama-server not found at $LLAMA_BIN — run install_llamacpp.sh first" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "Model not found at $MODEL — run download_model.sh first" >&2
    exit 1
fi

# Canonical command — referenced from spec §5.3
# NO --cache-reuse during measurement phases 2-4
exec "$LLAMA_BIN" \
    -m "$MODEL" \
    --host 127.0.0.1 --port 8000 \
    --metrics --slots \
    -c 32768 \
    --n-gpu-layers 999 \
    --parallel 1 \
    --seed 42 \
    --timeout 600
```

- [ ] **Step 2: Smoke-test the script**

```bash
chmod +x repro/setup/start_server.sh
./repro/setup/start_server.sh &
SERVER_PID=$!
sleep 10
curl -s http://localhost:8000/v1/models | head -50
curl -s http://localhost:8000/metrics | head -20
curl -s http://localhost:8000/slots | head -50
kill $SERVER_PID 2>/dev/null || true
```
Expected: `/v1/models` returns JSON; `/metrics` contains `llamacpp:prompt_seconds_total`; `/slots` returns a JSON array with one element having `is_processing` field.

- [ ] **Step 3: Commit**

```bash
git add repro/setup/start_server.sh
git commit -m "feat: canonical llama-server startup script"
```

---

### Task 7: Initialize Python environment + dependencies

**Files:**
- Create: `repro/pyproject.toml`
- Create: `repro/.venv/` (gitignored)

- [ ] **Step 1: Write `repro/pyproject.toml`**

```toml
[project]
name = "agentic-rag-repro"
version = "0.1.0"
description = "Local reproduction of HotpotQA experiments from arXiv 2506.04301v2"
requires-python = "==3.13.*"

dependencies = [
    # Match AgentBench's requirements.txt pin
    "langchain==1.0.5",
    "langchain-core==1.0.4",
    "langchain-openai==1.0.2",
    "langgraph==1.0.3",
    "langgraph-prebuilt==1.0.2",
    "openai>=1.40.0",

    # Measurement
    "pydantic>=2.7",
    "requests>=2.31",
    "tenacity>=8.0",

    # Analysis
    "pandas>=2.2",
    "matplotlib>=3.8",
    "scipy>=1.13",
    "pyyaml>=6.0",
    "numpy>=2.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-timeout>=2.3",   # for --timeout=... on long integration tests
    "httpx>=0.27",   # for mock server tests
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Install in venv**

```bash
cd /Users/imdonghyeon/agentic_rag/repro
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
python -c "import langchain, langgraph, pydantic; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 3: Also install AgentBench requirements (in same venv)**

```bash
pip install -r /Users/imdonghyeon/agentic_rag/AgentBench/requirements.txt
```
Expected: no version conflicts (pyproject.toml deps were chosen to match).

- [ ] **Step 4: Commit**

```bash
cd /Users/imdonghyeon/agentic_rag
git add repro/pyproject.toml
git commit -m "chore: pyproject.toml with pinned dependencies"
```

---

## Phase 1 — Patches (Tasks 8–13)

### Task 8: Write `config.patch`

**Files:**
- Create: `repro/patches/config.patch`

The patch sets AgentBench's `config.yaml` to point at our llama-server with `samples: 1, shuffle: false`.

- [ ] **Step 1: View original `config.yaml` to baseline the diff**

```bash
cat /Users/imdonghyeon/agentic_rag/AgentBench/config.yaml | head -15
```

- [ ] **Step 2: Edit `config.yaml` to target our setup**

Modify `/Users/imdonghyeon/agentic_rag/AgentBench/config.yaml` so the `global:` block reads:
```yaml
global:
  model: "Meta-Llama-3.1-8B-Instruct-Q4_K_M"
  host: 127.0.0.1
  port: 8000
  temperature: 0.0
  samples: 1                # OUR sweep_runner drives the outer loop
  shuffle: false            # See deterministic_select.patch (Task 9)
  save_trace: true
  trace_path: "./trace.txt"
  webshop_url: "http://localhost:3000"
```

- [ ] **Step 3: Generate the patch**

```bash
cd /Users/imdonghyeon/agentic_rag/AgentBench
git diff config.yaml > /Users/imdonghyeon/agentic_rag/repro/patches/config.patch
git checkout -- config.yaml   # revert; we'll re-apply via patch
cat /Users/imdonghyeon/agentic_rag/repro/patches/config.patch
```
Expected: a `diff --git a/config.yaml b/config.yaml` block with the changes above.

- [ ] **Step 4: Verify patch applies cleanly**

```bash
cd /Users/imdonghyeon/agentic_rag/AgentBench
git apply --check /Users/imdonghyeon/agentic_rag/repro/patches/config.patch
echo "Apply check: $?"
```
Expected: exit 0 (clean apply).

- [ ] **Step 5: Commit**

```bash
cd /Users/imdonghyeon/agentic_rag
git add repro/patches/config.patch
git commit -m "feat: config.patch — point AgentBench at our llama-server"
```

---

### Task 9: Write `deterministic_select.patch`

**Files:**
- Modify: `AgentBench/src/utils.py` (single `random.shuffle` call site)
- Create: `repro/patches/deterministic_select.patch`
- Create: `repro/tests/test_deterministic_select.py`

- [ ] **Step 1: Locate the shuffle call**

```bash
grep -n "random.shuffle\|shuffle" /Users/imdonghyeon/agentic_rag/AgentBench/src/utils.py
```
Expected: a line like `random.shuffle(data)` inside `load_dataset()`.

- [ ] **Step 2: Edit `src/utils.py` `load_dataset` to use seeded `Random` and explicit index selection**

In `/Users/imdonghyeon/agentic_rag/AgentBench/src/utils.py`, locate the body of `load_dataset(workload, shuffle=...)`. Modify so that when `shuffle=False`, the dataset is returned in deterministic order based on a fixed seed (read from env var `REPRO_SAMPLE_SEED`, default 42):

Replace the original shuffle block with:
```python
import os
import random

# After loading raw data list `data`:
seed = int(os.environ.get("REPRO_SAMPLE_SEED", "42"))
if shuffle:
    random.shuffle(data)
else:
    # Deterministic: shuffle a list of indices with seeded RNG, then reorder
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    data = [data[i] for i in indices]
```

(Adjust to match the actual variable name `data` in the file.)

- [ ] **Step 3: Generate the patch**

```bash
cd /Users/imdonghyeon/agentic_rag/AgentBench
git diff src/utils.py > /Users/imdonghyeon/agentic_rag/repro/patches/deterministic_select.patch
git checkout -- src/utils.py
```

- [ ] **Step 4: Write determinism test**

Create `/Users/imdonghyeon/agentic_rag/repro/tests/test_deterministic_select.py`:
```python
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/imdonghyeon/agentic_rag")
AB = REPO / "AgentBench"

def _apply_patches():
    # Reset only the specific file this patch touches (preserve other in-progress patches)
    subprocess.run(["git", "checkout", "--", "src/utils.py"],
                   cwd=AB, check=True, capture_output=True)
    subprocess.run(["git", "apply", str(REPO / "repro/patches/deterministic_select.patch")],
                    cwd=AB, check=True)

def _load_first_n(n: int, seed: int = 42):
    """Use AgentBench's load_dataset to grab first n samples deterministically."""
    sys.path.insert(0, str(AB))
    if "src.utils" in sys.modules:
        del sys.modules["src.utils"]
    os.environ["REPRO_SAMPLE_SEED"] = str(seed)
    from src.utils import load_dataset
    data = load_dataset("hotpotqa", shuffle=False)
    sys.path.pop(0)
    return [d["_id"] for d in data[:n]]

def test_two_loads_same_seed_same_ids():
    _apply_patches()
    first = _load_first_n(5)
    second = _load_first_n(5)
    assert first == second, f"Non-deterministic selection: {first} != {second}"

def test_different_seeds_different_ids():
    _apply_patches()
    s42 = _load_first_n(5, seed=42)
    s43 = _load_first_n(5, seed=43)
    assert s42 != s43, "Same selection across seeds — RNG not respected"
```

- [ ] **Step 5: Run test, expect FAIL initially**

```bash
cd /Users/imdonghyeon/agentic_rag/repro
source .venv/bin/activate
pytest tests/test_deterministic_select.py -v
```
Expected: tests **fail** if patch hasn't been applied cleanly (first run after apply should pass; this is the smoke).

If pass: proceed. If fail: read the failure, fix the patch, regenerate, re-test.

- [ ] **Step 6: Commit**

```bash
cd /Users/imdonghyeon/agentic_rag
git add repro/patches/deterministic_select.patch repro/tests/test_deterministic_select.py
git commit -m "feat: deterministic HotpotQA sample selection via seeded RNG"
```

---

### Task 10: Write `entry_points.patch` — ReAct

**Files:**
- Modify: `AgentBench/run_react.py`
- Output (after all 4 patches): `repro/patches/entry_points.patch`

- [ ] **Step 1: Read `run_react.py` end-to-end AND record real helper names**

```bash
wc -l /Users/imdonghyeon/agentic_rag/AgentBench/run_react.py
cat /Users/imdonghyeon/agentic_rag/AgentBench/run_react.py
grep -nE 'def |ChatOpenAI|create_react_agent|load_dataset' /Users/imdonghyeon/agentic_rag/AgentBench/run_react.py
```

Identify:
- The sample loop (`for sample in samples: ...`)
- The body processing one sample (LLM construction, agent build, stream/invoke, answer extraction)
- **Real names** of helpers — write them down before refactoring:
  - Config loader: `_load_global_config` (this plan's name) → REAL: `<fill in>`
  - Tool builder: `_build_hotpotqa_tools` → REAL: `<fill in>`
  - Message builder: `_build_react_messages` → REAL: `<fill in>`

The plan's placeholder names below MUST be replaced with the real ones from this Step.

- [ ] **Step 2: Refactor — extract a `run_single_query` function**

Edit `/Users/imdonghyeon/agentic_rag/AgentBench/run_react.py`. Add a top-level function above `main()`:

```python
def run_single_query(sample: dict, agent_kwargs: dict) -> dict:
    """
    One-shot per-sample entry point used by repro/sweep/agent_runner.py.

    Args:
        sample: HotpotQA dataset entry. Must contain 'question' and 'answer'.
        agent_kwargs: {
            'fewshot': int,
            'iteration_limit': int,
            'callbacks': list[BaseCallbackHandler],
            'config': dict (optional global config; defaults to load from config.yaml),
        }
    Returns:
        {
            'answer': str,            # final agent answer
            'raw_messages': list,     # full message stream for debugging
            'iterations': int,        # actual iterations consumed
        }
    """
    config = agent_kwargs.get("config") or _load_global_config()
    fewshot = agent_kwargs["fewshot"]
    iteration_limit = agent_kwargs["iteration_limit"]
    callbacks = agent_kwargs.get("callbacks", [])

    # [Reuse the existing per-sample LLM construction, agent build, and stream loop
    #  from main(). Wrap the original logic so it accepts callbacks via
    #  RunnableConfig: agent.stream(state, config={"callbacks": callbacks}).
    #  Extract the final AI message content (or Action: Finish[X] capture group).]

    llm = ChatOpenAI(
        model=config["model"],
        base_url=f"http://{config['host']}:{config['port']}/v1",
        api_key="EMPTY",
        temperature=config["temperature"],
        stream_usage=True,
        timeout=600,
        max_tokens=2048,  # safety cap; see spec §11
    )
    tools = _build_hotpotqa_tools()
    agent = create_react_agent(llm, tools)

    messages = _build_react_messages(sample, fewshot=fewshot)
    raw = []
    answer = ""
    iterations = 0
    for chunk in agent.stream(
        {"messages": messages},
        stream_mode="values",
        config={"callbacks": callbacks, "recursion_limit": iteration_limit * 3},
    ):
        raw.append(chunk)
        iterations += 1

    # Extract final answer
    last_msg = raw[-1]["messages"][-1]
    text = getattr(last_msg, "content", "") or ""
    import re
    m = re.search(r"Action:\s*Finish\[(.+?)\]", text, re.DOTALL)
    answer = m.group(1).strip() if m else text.strip()

    return {"answer": answer, "raw_messages": raw, "iterations": iterations}
```

(Names like `_load_global_config`, `_build_hotpotqa_tools`, `_build_react_messages` may already exist in `run_react.py` with different names — adapt to use the real ones. The point is to wrap the *existing* per-sample body, not rewrite it.)

- [ ] **Step 3: Verify import works**

```bash
cd /Users/imdonghyeon/agentic_rag/repro && source .venv/bin/activate
PYTHONPATH=/Users/imdonghyeon/agentic_rag/AgentBench python -c "from run_react import run_single_query; print(run_single_query.__doc__[:80])"
```
Expected: prints the docstring start.

- [ ] **Step 4: Skip patch generation until all 4 entry_points are done**

(We'll generate one combined `entry_points.patch` after Task 13.)

---

### Task 11: Write `entry_points.patch` — Reflexion

**Files:**
- Modify: `AgentBench/run_reflexion.py`

- [ ] **Step 1: Read `run_reflexion.py` AND identify exact helper names**

```bash
cat /Users/imdonghyeon/agentic_rag/AgentBench/run_reflexion.py
# Identify the actual helper names used (LLM construction, agent build, sample loop body).
# Record them as a comment in the patch — DO NOT guess.
grep -nE 'def |ChatOpenAI|main\(' /Users/imdonghyeon/agentic_rag/AgentBench/run_reflexion.py
```

Write down the exact function names you find at the top of your `run_single_query` as a comment, e.g.:
```python
# Helpers in this file used by run_single_query:
#   - <real_name_1>
#   - <real_name_2>
```

- [ ] **Step 2: Extract `run_single_query`**

Same pattern as Task 10, **but using the real helper names from Step 1**. All `ChatOpenAI` and reflection-LLM constructions must include `max_tokens=2048` (Q4 numerical-instability cap per spec §11) and `timeout=600`. The signature stays:
```python
def run_single_query(sample: dict, agent_kwargs: dict) -> dict
```
Reflexion's outer loop has trials with reflections between them. `agent_kwargs` includes `reflection_limit` (in addition to `fewshot`, `iteration_limit`). Callbacks are passed to **both** the actor LLM and the reflection LLM via `RunnableConfig`.

Extract the existing per-sample body — typically a loop:
```
trial 1: ReAct-style steps → final answer (maybe correct, maybe wrong)
if wrong: reflection LLM → reflection string added
trial 2: ReAct-style steps with reflections in context
...
```

Final answer = last trial's `Action: Finish[X]` capture.

**Counter increment**: increment `TRACE.get().n_reflections` each time a reflection LLM call is made. The callback handler in Task 18 records LLM call spans, but reflection-specific counting is the responsibility of the entry-point wrapper (the callback handler can't distinguish actor LLM from reflection LLM). Add:
```python
from measurement.chat_wrapper import TRACE
# At the point where a reflection step is invoked:
trace = TRACE.get()
trace.n_reflections += 1
```

- [ ] **Step 3: Verify import**

```bash
PYTHONPATH=/Users/imdonghyeon/agentic_rag/AgentBench python -c "from run_reflexion import run_single_query; print('ok')"
```
Expected: `ok`.

---

### Task 12: Write `entry_points.patch` — LATS

**Files:**
- Modify: `AgentBench/run_lats.py`
- Note: LATS uses `src/agents/LATS/model_client.py::OpenAIChatClient` (composition wrapper, not direct ChatOpenAI). Callbacks must propagate.

- [ ] **Step 1: Read `run_lats.py` and `src/agents/LATS/model_client.py` AND find the real LATS search function**

```bash
cat /Users/imdonghyeon/agentic_rag/AgentBench/run_lats.py
ls /Users/imdonghyeon/agentic_rag/AgentBench/src/agents/LATS/hotpotqa/
grep -rn "def " /Users/imdonghyeon/agentic_rag/AgentBench/src/agents/LATS/hotpotqa/ | grep -i "search\|tree\|main\|run"
sed -n '80,160p' /Users/imdonghyeon/agentic_rag/AgentBench/src/agents/LATS/model_client.py
```
Identify (a) the exact function in `src/agents/LATS/hotpotqa/` that executes one full LATS tree search for one sample, and (b) confirm `OpenAIChatClient.__init__` wraps `ChatOpenAI(...)` — callbacks must be injected via `client.llm = client.llm.with_config({"callbacks": callbacks})`.

- [ ] **Step 2: Extract `run_single_query`**

Use the **actual** LATS search function name discovered in Step 1 (called `_LATS_SEARCH_FN` in the template below — replace before writing code):

```python
def run_single_query(sample: dict, agent_kwargs: dict) -> dict:
    config = agent_kwargs.get("config") or _load_global_config()
    callbacks = agent_kwargs.get("callbacks", [])
    fewshot = agent_kwargs["fewshot"]
    iteration_limit = agent_kwargs["iteration_limit"]
    max_depth = agent_kwargs.get("max_depth", 7)
    n_generate = agent_kwargs.get("n_generate_sample", 5)

    # Build the client; inject callbacks via the inner ChatOpenAI's config
    client = OpenAIChatClient(
        model=config["model"],
        base_url=f"http://{config['host']}:{config['port']}/v1",
        temperature=1.0,  # LATS uses sampling for child diversification
    )
    # Cap output tokens to mitigate Q4 numerical instability over many calls (spec §11)
    # and propagate callbacks to the inner ChatOpenAI:
    client.llm = client.llm.bind(max_tokens=2048).with_config({"callbacks": callbacks})

    # Reuse the existing LATS hotpotqa search function (REPLACE _LATS_SEARCH_FN with real name)
    from src.agents.LATS.hotpotqa.<MODULE> import <ACTUAL_FN> as _LATS_SEARCH_FN
    best_node = _LATS_SEARCH_FN(
        sample=sample, client=client, fewshot=fewshot,
        iteration_limit=iteration_limit, max_depth=max_depth,
        n_generate_sample=n_generate,
    )

    # Count tree expansions for spec schema field n_tree_expansions
    from measurement.chat_wrapper import TRACE
    trace = TRACE.get()
    trace.n_tree_expansions = getattr(best_node, "n_expansions", iteration_limit)

    answer = best_node.answer if best_node else ""
    return {
        "answer": answer,
        "raw_messages": getattr(best_node, "trajectory", []),
        "iterations": getattr(best_node, "iterations_used", 0),
    }
```

Step 1's `grep` output tells you what to substitute for `<MODULE>` and `<ACTUAL_FN>`. Do NOT proceed to Step 3 until the import resolves cleanly.

- [ ] **Step 3: Verify import**

```bash
PYTHONPATH=/Users/imdonghyeon/agentic_rag/AgentBench python -c "from run_lats import run_single_query; print('ok')"
```
Expected: `ok`.

---

### Task 13: Write `entry_points.patch` — LLMCompiler + generate combined patch

**Files:**
- Modify: `AgentBench/run_llmcompiler.py`
- Create: `repro/patches/entry_points.patch`

- [ ] **Step 1: Read `run_llmcompiler.py`**

```bash
cat /Users/imdonghyeon/agentic_rag/AgentBench/run_llmcompiler.py
```
Note: this is **async**. The function uses `asyncio.run(agent.arun(question))`.

- [ ] **Step 2: Extract sync wrapper `run_single_query`**

```python
def run_single_query(sample: dict, agent_kwargs: dict) -> dict:
    import asyncio
    config = agent_kwargs.get("config") or _load_global_config()
    callbacks = agent_kwargs.get("callbacks", [])
    fewshot = agent_kwargs["fewshot"]
    max_replan = agent_kwargs.get("max_replan", 20)

    async def _run():
        # Use AgentBench's existing get_model() but inject callbacks
        from src.agents.LLMCompiler.utils.model_utils import get_model
        llm = get_model(
            model=config["model"],
            base_url=f"http://{config['host']}:{config['port']}/v1",
            temperature=config["temperature"],
        )
        # Cap output tokens (Q4 instability mitigation) + propagate callbacks
        llm = llm.bind(max_tokens=2048).with_config({"callbacks": callbacks})
        agent = _build_llmcompiler_agent(llm, fewshot=fewshot, max_replan=max_replan)
        result = await agent.arun(sample["question"])
        return result

    answer = asyncio.run(_run())
    return {"answer": str(answer).strip(), "raw_messages": [], "iterations": -1}
```

(`iterations: -1` for now — LLMCompiler doesn't expose this cleanly. Update later if needed.)

- [ ] **Step 3: Verify all 4 imports**

```bash
PYTHONPATH=/Users/imdonghyeon/agentic_rag/AgentBench python -c "
from run_react import run_single_query as r1
from run_reflexion import run_single_query as r2
from run_lats import run_single_query as r3
from run_llmcompiler import run_single_query as r4
print('all 4 ok')
"
```
Expected: `all 4 ok`.

- [ ] **Step 4: Generate combined patch**

```bash
cd /Users/imdonghyeon/agentic_rag/AgentBench
git diff run_react.py run_reflexion.py run_lats.py run_llmcompiler.py \
    > /Users/imdonghyeon/agentic_rag/repro/patches/entry_points.patch
wc -l /Users/imdonghyeon/agentic_rag/repro/patches/entry_points.patch
git checkout -- run_react.py run_reflexion.py run_lats.py run_llmcompiler.py
```
Expected: patch is non-empty (>50 lines).

- [ ] **Step 5: Verify clean apply on fresh checkout**

```bash
git apply --check /Users/imdonghyeon/agentic_rag/repro/patches/entry_points.patch
echo "Apply check: $?"
```
Expected: exit 0.

- [ ] **Step 6: Apply all three patches and commit**

```bash
cd /Users/imdonghyeon/agentic_rag/AgentBench
git apply /Users/imdonghyeon/agentic_rag/repro/patches/config.patch
git apply /Users/imdonghyeon/agentic_rag/repro/patches/deterministic_select.patch
git apply /Users/imdonghyeon/agentic_rag/repro/patches/entry_points.patch

cd /Users/imdonghyeon/agentic_rag
git add repro/patches/entry_points.patch
git commit -m "feat: entry_points.patch — extract run_single_query from all 4 agents"
```

---

## Phase 2 — Measurement Infrastructure (Tasks 14–18)

### Task 14: `trace_schema.py` — Pydantic models

**Files:**
- Create: `repro/measurement/trace_schema.py`
- Create: `repro/tests/measurement/test_trace_schema.py`

- [ ] **Step 1: Write failing test**

Create `/Users/imdonghyeon/agentic_rag/repro/tests/measurement/test_trace_schema.py`:
```python
import json
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
```

- [ ] **Step 2: Run test, verify FAIL**

```bash
cd /Users/imdonghyeon/agentic_rag/repro
pytest tests/measurement/test_trace_schema.py -v
```
Expected: `ModuleNotFoundError: No module named 'measurement.trace_schema'`

- [ ] **Step 3: Implement `trace_schema.py`**

Create `/Users/imdonghyeon/agentic_rag/repro/measurement/trace_schema.py`:
```python
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
```

- [ ] **Step 4: Run test, verify PASS**

```bash
pytest tests/measurement/test_trace_schema.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/imdonghyeon/agentic_rag
git add repro/measurement/trace_schema.py repro/tests/measurement/test_trace_schema.py
git commit -m "feat: Pydantic trace schema (QueryTrace, LLMCallSpan, ToolCallSpan)"
```

---

### Task 15: `eval.py` — `normalize_answer` + `hotpotqa_em`

**Files:**
- Create: `repro/measurement/eval.py`
- Create: `repro/tests/measurement/test_eval.py`

- [ ] **Step 1: Write failing test**

Create `/Users/imdonghyeon/agentic_rag/repro/tests/measurement/test_eval.py`:
```python
from measurement.eval import normalize_answer, hotpotqa_em

def test_normalize_strips_articles():
    assert normalize_answer("The Eiffel Tower") == "eiffel tower"
    assert normalize_answer("a cat") == "cat"
    assert normalize_answer("an apple") == "apple"

def test_normalize_strips_punctuation():
    assert normalize_answer("Paris, France.") == "paris france"

def test_normalize_lowercases():
    assert normalize_answer("PARIS") == "paris"

def test_em_passes_normalized_equal():
    assert hotpotqa_em("The Eiffel Tower", "eiffel tower") is True
    assert hotpotqa_em("Paris.", "paris") is True

def test_em_fails_substring():
    assert hotpotqa_em("Paris, France", "Paris") is False

def test_em_handles_empty():
    assert hotpotqa_em("", "") is True
    assert hotpotqa_em("answer", "") is False
```

- [ ] **Step 2: Run test, verify FAIL**

```bash
pytest tests/measurement/test_eval.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `eval.py`**

Create `/Users/imdonghyeon/agentic_rag/repro/measurement/eval.py`:
```python
"""HotpotQA exact-match scoring, ported from hotpot_evaluate_v1.py."""
import re
import string


def normalize_answer(s: str) -> str:
    """Lowercase, remove articles, strip punctuation, normalize whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def hotpotqa_em(predicted: str, expected: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(expected)
```

- [ ] **Step 4: Run test, verify PASS**

```bash
pytest tests/measurement/test_eval.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Cross-check against AgentBench's `hotpot_evaluate.py`**

```bash
cat /Users/imdonghyeon/agentic_rag/AgentBench/src/tools/hotpotqa_tools/hotpot_evaluate.py | head -40
```
Compare `normalize_answer`. If identical → ok. If differs → update ours to match (canonical paper version takes precedence).

- [ ] **Step 6: Commit**

```bash
git add repro/measurement/eval.py repro/tests/measurement/test_eval.py
git commit -m "feat: HotpotQA normalize_answer + hotpotqa_em"
```

---

### Task 16: `metrics_collector.py` — parse `/metrics` and `/slots`

**Files:**
- Create: `repro/measurement/metrics_collector.py` (parsing only — polling thread in Task 17)
- Create: `repro/tests/measurement/test_metrics_parsing.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/imdonghyeon/agentic_rag/repro/tests/measurement/test_metrics_parsing.py`:
```python
from measurement.metrics_collector import parse_metrics, parse_slots, PollSample

METRICS_SAMPLE = """\
# HELP llamacpp:prompt_seconds_total Prefill time
# TYPE llamacpp:prompt_seconds_total counter
llamacpp:prompt_seconds_total 12.34
# TYPE llamacpp:tokens_predicted_seconds_total counter
llamacpp:tokens_predicted_seconds_total 45.67
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 1500
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 3200
# TYPE llamacpp:n_decode_total counter
llamacpp:n_decode_total 3200
# TYPE llamacpp:n_tokens_max gauge
llamacpp:n_tokens_max 4096
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
"""

SLOTS_SAMPLE = [
    {
        "id": 0,
        "is_processing": True,
        "id_task": 12,
        "n_ctx": 32768,
        "n_prompt_tokens": 2048,
        "n_prompt_tokens_processed": 2048,
        "n_prompt_tokens_cache": 512,
        "prompt": "...",
    }
]

def test_parse_metrics():
    d = parse_metrics(METRICS_SAMPLE)
    assert d["llamacpp:prompt_seconds_total"] == 12.34
    assert d["llamacpp:tokens_predicted_total"] == 3200
    assert d["llamacpp:requests_processing"] == 1

def test_parse_slots():
    d = parse_slots(SLOTS_SAMPLE)
    assert d["is_processing"] is True
    assert d["n_prompt_tokens"] == 2048
    assert d["n_prompt_tokens_cache"] == 512

def test_poll_sample_construction():
    m = parse_metrics(METRICS_SAMPLE)
    s = parse_slots(SLOTS_SAMPLE)
    sample = PollSample.from_endpoints(t=1.23, metrics=m, slots=s)
    assert sample.prefill_s_total == 12.34
    assert sample.is_processing is True
    assert sample.n_prompt_tokens_cache == 512
```

- [ ] **Step 2: Run, verify FAIL**

```bash
pytest tests/measurement/test_metrics_parsing.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement parser + PollSample**

Create `/Users/imdonghyeon/agentic_rag/repro/measurement/metrics_collector.py`:
```python
"""llama.cpp metrics polling and parsing.

Polling thread implementation in Task 17; this file is parser-only for now.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class PollSample:
    t: float
    prefill_s_total: float
    decode_s_total: float
    prefill_tokens_total: int
    decode_tokens_total: int
    n_decode_total: int
    n_tokens_max: int
    requests_processing: int
    is_processing: bool
    n_prompt_tokens: int
    n_prompt_tokens_processed: int
    n_prompt_tokens_cache: int

    @classmethod
    def from_endpoints(cls, *, t: float, metrics: dict, slots: dict) -> "PollSample":
        return cls(
            t=t,
            prefill_s_total=metrics.get("llamacpp:prompt_seconds_total", 0.0),
            decode_s_total=metrics.get("llamacpp:tokens_predicted_seconds_total", 0.0),
            prefill_tokens_total=int(metrics.get("llamacpp:prompt_tokens_total", 0)),
            decode_tokens_total=int(metrics.get("llamacpp:tokens_predicted_total", 0)),
            n_decode_total=int(metrics.get("llamacpp:n_decode_total", 0)),
            n_tokens_max=int(metrics.get("llamacpp:n_tokens_max", 0)),
            requests_processing=int(metrics.get("llamacpp:requests_processing", 0)),
            is_processing=slots.get("is_processing", False),
            n_prompt_tokens=slots.get("n_prompt_tokens", 0),
            n_prompt_tokens_processed=slots.get("n_prompt_tokens_processed", 0),
            n_prompt_tokens_cache=slots.get("n_prompt_tokens_cache", 0),
        )


def parse_metrics(text: str) -> dict[str, float]:
    """Parse Prometheus text format. Skips comments and HELP/TYPE lines."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name, value = parts[0], parts[-1]
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


def parse_slots(slots_json: list[dict]) -> dict[str, Any]:
    """Take the first slot (we use --parallel 1). Returns flat dict."""
    if not slots_json:
        return {
            "is_processing": False,
            "n_prompt_tokens": 0,
            "n_prompt_tokens_processed": 0,
            "n_prompt_tokens_cache": 0,
        }
    s = slots_json[0]
    return {
        "is_processing": bool(s.get("is_processing", False)),
        "n_prompt_tokens": int(s.get("n_prompt_tokens", 0)),
        "n_prompt_tokens_processed": int(s.get("n_prompt_tokens_processed", 0)),
        "n_prompt_tokens_cache": int(s.get("n_prompt_tokens_cache", 0)),
    }
```

- [ ] **Step 4: Run, verify PASS**

```bash
pytest tests/measurement/test_metrics_parsing.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add repro/measurement/metrics_collector.py repro/tests/measurement/test_metrics_parsing.py
git commit -m "feat: /metrics and /slots parser + PollSample dataclass"
```

---

### Task 17: `metrics_collector.py` — polling thread + ring buffer + slice

**Files:**
- Modify: `repro/measurement/metrics_collector.py` (add `MetricsCollector` class)
- Create: `repro/tests/measurement/test_metrics_collector.py`

- [ ] **Step 1: Write failing test using a fake HTTP server**

Create `/Users/imdonghyeon/agentic_rag/repro/tests/measurement/test_metrics_collector.py`:
```python
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from measurement.metrics_collector import MetricsCollector

METRICS_TEXT = """\
llamacpp:prompt_seconds_total {prefill}
llamacpp:tokens_predicted_seconds_total {decode}
llamacpp:prompt_tokens_total 0
llamacpp:tokens_predicted_total 0
llamacpp:n_decode_total 0
llamacpp:n_tokens_max 0
llamacpp:requests_processing 0
"""

class _FakeHandler(BaseHTTPRequestHandler):
    counter = [0]
    def do_GET(self):
        if self.path == "/metrics":
            i = self.counter[0]; self.counter[0] += 1
            body = METRICS_TEXT.format(prefill=i*0.1, decode=i*0.2).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        elif self.path == "/slots":
            body = b'[{"id":0,"is_processing":false,"n_prompt_tokens":0,"n_prompt_tokens_processed":0,"n_prompt_tokens_cache":0}]'
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a): pass

def _serve():
    srv = HTTPServer(("127.0.0.1", 0), _FakeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv

def test_collector_polls_and_slices():
    srv = _serve()
    port = srv.server_address[1]
    coll = MetricsCollector(f"http://127.0.0.1:{port}", interval_s=0.05)
    coll.start()
    t0 = time.perf_counter()
    time.sleep(0.3)
    t1 = time.perf_counter()
    coll.stop()

    samples = coll.slice(t0, t1)
    assert len(samples) >= 3
    # Cumulative counters must be monotone non-decreasing
    for a, b in zip(samples, samples[1:]):
        assert b.prefill_s_total >= a.prefill_s_total
        assert b.decode_s_total >= a.decode_s_total

def test_slice_includes_boundary_samples():
    srv = _serve()
    port = srv.server_address[1]
    coll = MetricsCollector(f"http://127.0.0.1:{port}", interval_s=0.05)
    coll.start()
    time.sleep(0.4)
    # Take a window that's entirely between two ticks
    mid = time.perf_counter() - 0.025
    samples = coll.slice(mid, mid + 0.001)
    coll.stop()
    # Even a 1 ms window must include bracket samples (one before, one after)
    assert len(samples) >= 2
```

- [ ] **Step 2: Run, verify FAIL**

```bash
pytest tests/measurement/test_metrics_collector.py -v
```
Expected: AttributeError (no MetricsCollector).

- [ ] **Step 3: Implement `MetricsCollector`**

Append to `/Users/imdonghyeon/agentic_rag/repro/measurement/metrics_collector.py`:
```python
import threading
import time
from collections import deque
from typing import Optional
import requests


class MetricsCollector:
    """Background polling thread for llama-server /metrics and /slots.

    Lifecycle:
        c = MetricsCollector("http://localhost:8000").start()
        # ... run sweep ...
        samples = c.slice(t_start, t_end)
        c.stop()

    `slice(a, b)` returns all PollSample within [a, b] PLUS the boundary samples
    (one before a, one after b) when available — required for correct delta
    computation when the window is shorter than the polling interval (see spec §7.4).
    """

    def __init__(self, base_url: str, *, interval_s: float = 0.1, maxlen: int = 1_800_000):
        self.base_url = base_url.rstrip("/")
        self.interval_s = interval_s
        self._buf: deque[PollSample] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._session = requests.Session()

    def start(self) -> "MetricsCollector":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="MetricsCollector")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                m_text = self._session.get(f"{self.base_url}/metrics", timeout=2).text
                s_json = self._session.get(f"{self.base_url}/slots", timeout=2).json()
                sample = PollSample.from_endpoints(
                    t=t,
                    metrics=parse_metrics(m_text),
                    slots=parse_slots(s_json),
                )
                with self._lock:
                    self._buf.append(sample)
            except Exception:
                # transient network errors — skip this tick
                pass
            self._stop.wait(self.interval_s)

    def slice(self, t_start: float, t_end: float) -> list[PollSample]:
        """Return samples in [t_start, t_end] plus bracket samples (one before, one after)."""
        with self._lock:
            buf = list(self._buf)
        if not buf:
            return []
        in_window = [s for s in buf if t_start <= s.t <= t_end]
        before = [s for s in buf if s.t < t_start]
        after = [s for s in buf if s.t > t_end]
        result: list[PollSample] = []
        if before:
            result.append(before[-1])
        result.extend(in_window)
        if after:
            result.append(after[0])
        return result

    def detect_kv_eviction(self, samples: list[PollSample]) -> bool:
        """Spec §11: return True if n_prompt_tokens_cache decreased while is_processing=True.

        Such a decrease signals llama.cpp evicted cached prefix tokens to fit a longer
        prompt — relevant for LATS at long contexts.
        """
        prev = None
        for s in samples:
            if prev is not None and s.is_processing and prev.is_processing:
                if s.n_prompt_tokens_cache < prev.n_prompt_tokens_cache:
                    return True
            prev = s
        return False

    def detect_no_decode_progress(
        self, *, window_samples: int = 30
    ) -> bool:
        """Spec §9.5 #2: return True if decode_tokens_total did not increase
        across the last `window_samples` polls while is_processing=True.

        Caller checks this from the run_one watchdog every poll interval.
        """
        with self._lock:
            buf = list(self._buf)
        if len(buf) < window_samples:
            return False
        recent = buf[-window_samples:]
        if not all(s.is_processing for s in recent):
            return False
        return recent[-1].decode_tokens_total == recent[0].decode_tokens_total
```

- [ ] **Step 4: Run, verify PASS**

```bash
pytest tests/measurement/test_metrics_collector.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add repro/measurement/metrics_collector.py repro/tests/measurement/test_metrics_collector.py
git commit -m "feat: MetricsCollector polling thread with bracket-aware slice"
```

---

### Task 18: `chat_wrapper.py` — `TraceCallbackHandler`

**Files:**
- Create: `repro/measurement/chat_wrapper.py`
- Create: `repro/tests/measurement/test_chat_wrapper.py`

- [ ] **Step 1: Write failing test**

Create `/Users/imdonghyeon/agentic_rag/repro/tests/measurement/test_chat_wrapper.py`:
```python
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
```

- [ ] **Step 2: Run, verify FAIL**

```bash
pytest tests/measurement/test_chat_wrapper.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `chat_wrapper.py`**

Create `/Users/imdonghyeon/agentic_rag/repro/measurement/chat_wrapper.py`:
```python
"""LangChain BaseCallbackHandler for uniform LLM/tool span instrumentation.

Works across ChatOpenAI (ReAct, Reflexion), OpenAIChatClient composition (LATS),
get_model() factory (LLMCompiler), and sync/async/stream code paths.

Per-query TRACE is scoped via contextvars.ContextVar — `run_one` sets/resets it.
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
        # Defensive token extraction (per spec §7.3)
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or {}
        span.tokens_in = int(usage.get("prompt_tokens", 0) or 0)
        span.tokens_out = int(usage.get("completion_tokens", 0) or 0)

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
```

- [ ] **Step 4: Run, verify PASS**

```bash
pytest tests/measurement/test_chat_wrapper.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add repro/measurement/chat_wrapper.py repro/tests/measurement/test_chat_wrapper.py
git commit -m "feat: TraceCallbackHandler for uniform LLM/tool span capture"
```

---

## Phase 3 — Sweep Infrastructure (Tasks 19–23)

### Task 19: `cells.py` — enumerate_cells + ResumeKey

**Files:**
- Create: `repro/sweep/cells.py`
- Create: `repro/tests/sweep/test_cells.py`

- [ ] **Step 1: Write failing test**

Create `/Users/imdonghyeon/agentic_rag/repro/tests/sweep/test_cells.py`:
```python
from sweep.cells import enumerate_cells, ResumeKey, Cell

CFG_FIG14 = {
    "run_id": "fig14_iteration_sweep",
    "workload": "hotpotqa",
    "agent_type": "react",
    "defaults": {"fewshot": 5, "iteration_limit": 30},
    "sweeps": {"iteration_limit": [5, 10, 20]},
    "samples_per_cell": 3,
    "sample_seed": 42,
}

def test_enumerate_cells_fig14():
    cells = list(enumerate_cells(CFG_FIG14))
    # 3 iteration values × 3 samples = 9 cells
    assert len(cells) == 9
    first = cells[0]
    assert first.agent_type == "react"
    assert first.fewshot == 5
    assert first.iteration_limit == 5
    assert first.sample_idx == 0
    assert first.sweep_var_name == "iteration_limit"
    assert first.sweep_var_val == 5

def test_resume_key_uniqueness():
    cells = list(enumerate_cells(CFG_FIG14))
    keys = [c.resume_key() for c in cells]
    assert len(keys) == len(set(keys))

def test_resume_key_format():
    c = Cell(agent_type="react", fewshot=5, iteration_limit=10,
             sample_idx=2, sweep_var_name="iteration_limit", sweep_var_val=10)
    # ResumeKey = (agent_type, (fewshot, iteration_limit), sample_idx)
    assert c.resume_key() == ("react", (5, 10), 2)

CFG_FIG13 = {
    "run_id": "fig13_pareto",
    "workload": "hotpotqa",
    "agent_types": ["react", "reflexion", "lats", "llmcompiler"],
    "samples_per_agent": {"react": 3, "reflexion": 3, "lats": 2, "llmcompiler": 3},
    "defaults": {"fewshot": 5, "iteration_limit": 30, "max_replan": 20},
    "sample_seed": 42,
}

def test_enumerate_cells_fig13_variable_samples_per_agent():
    cells = list(enumerate_cells(CFG_FIG13))
    assert len(cells) == 3 + 3 + 2 + 3
    by_agent = {}
    for c in cells:
        by_agent.setdefault(c.agent_type, []).append(c)
    assert len(by_agent["lats"]) == 2
```

- [ ] **Step 2: Run, FAIL**

```bash
pytest tests/sweep/test_cells.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

Create `/Users/imdonghyeon/agentic_rag/repro/sweep/cells.py`:
```python
"""Enumerate sweep cells and define resume keys."""
from dataclasses import dataclass, asdict
from typing import Any, Iterator


# Resume key = (agent_type, signature, sample_idx) where signature is a tuple of
# all agent-level kwargs that distinguish cells. This shape is what sweep_runner
# uses to compare against rows reconstructed from JSONL — see sweep_runner.py
# `_resume_key_from_row` for the symmetric reconstruction.
ResumeKey = tuple[str, tuple, int]


def _signature(fewshot: int, iteration_limit: int) -> tuple:
    """Canonical signature of agent-level kwargs. Used by both Cell and JSONL reader."""
    return (fewshot, iteration_limit)


@dataclass
class Cell:
    agent_type: str
    fewshot: int
    iteration_limit: int
    sample_idx: int
    sweep_var_name: str       # "fewshot" | "iteration_limit" | "_default"
    sweep_var_val: Any
    # Extra agent-specific kwargs (e.g., max_replan, max_depth) pass through
    extra_kwargs: dict = None

    def __post_init__(self):
        if self.extra_kwargs is None:
            self.extra_kwargs = {}

    def resume_key(self) -> ResumeKey:
        return (self.agent_type, _signature(self.fewshot, self.iteration_limit), self.sample_idx)

    def as_run_one_kwargs(self) -> dict:
        return {
            "agent_type": self.agent_type,
            "fewshot": self.fewshot,
            "iteration_limit": self.iteration_limit,
            "sample_idx": self.sample_idx,
        }


def enumerate_cells(cfg: dict) -> Iterator[Cell]:
    """Generate cells from a sweep config.

    Two config shapes supported:
      - Single-agent sweep (Fig 14, Fig 15):
          {agent_type, defaults, sweeps: {var: [values]}, samples_per_cell}
      - Multi-agent Pareto (Fig 13):
          {agent_types: [...], defaults, samples_per_agent: {agent: n}}
    """
    defaults = cfg.get("defaults", {})

    if "agent_types" in cfg:
        # Pareto: one cell per (agent, sample) at default config
        samples_per_agent = cfg["samples_per_agent"]
        for agent in cfg["agent_types"]:
            n = samples_per_agent[agent]
            for i in range(n):
                yield Cell(
                    agent_type=agent,
                    fewshot=defaults.get("fewshot", 5),
                    iteration_limit=defaults.get("iteration_limit", 30),
                    sample_idx=i,
                    sweep_var_name="_default",
                    sweep_var_val=agent,    # disambiguates Pareto cells in resume key
                    extra_kwargs={k: v for k, v in defaults.items()
                                  if k not in ("fewshot", "iteration_limit")},
                )
    else:
        # Single-agent sweep
        agent = cfg["agent_type"]
        sweeps = cfg["sweeps"]
        if len(sweeps) != 1:
            raise ValueError(f"Single-agent config must sweep exactly one variable, got {list(sweeps)}")
        var_name, values = next(iter(sweeps.items()))
        n_samples = cfg["samples_per_cell"]
        for value in values:
            for i in range(n_samples):
                kwargs = dict(defaults)
                kwargs[var_name] = value
                yield Cell(
                    agent_type=agent,
                    fewshot=kwargs.get("fewshot", 5),
                    iteration_limit=kwargs.get("iteration_limit", 30),
                    sample_idx=i,
                    sweep_var_name=var_name,
                    sweep_var_val=value,
                    extra_kwargs={k: v for k, v in kwargs.items()
                                  if k not in ("fewshot", "iteration_limit")},
                )
```

- [ ] **Step 4: Run, PASS**

```bash
pytest tests/sweep/test_cells.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add repro/sweep/cells.py repro/tests/sweep/test_cells.py
git commit -m "feat: sweep cell enumeration with resume keys"
```

---

### Task 20: `agent_runner.py` — `_dispatch` + `extract_final_answer`

**Files:**
- Create: `repro/sweep/agent_runner.py` (stub — dispatch + extract only; run_one in Task 22)
- Create: `repro/tests/sweep/test_dispatch_and_extract.py`

- [ ] **Step 1: Write failing test**

Create `/Users/imdonghyeon/agentic_rag/repro/tests/sweep/test_dispatch_and_extract.py`:
```python
import pytest
from sweep.agent_runner import extract_final_answer

def test_extract_react_finish_action():
    result = {"raw_messages": [], "answer": "Action: Finish[Paris]"}
    # extract_final_answer should accept either the raw {answer: ...} or unwrap Finish[]
    assert extract_final_answer("react", result) == "Paris"

def test_extract_react_plain_text():
    result = {"answer": "Paris", "raw_messages": []}
    assert extract_final_answer("react", result) == "Paris"

def test_extract_lats_passthrough():
    result = {"answer": "Eiffel Tower", "raw_messages": []}
    assert extract_final_answer("lats", result) == "Eiffel Tower"

def test_extract_llmcompiler_strips_whitespace():
    result = {"answer": "  Paris\n", "raw_messages": []}
    assert extract_final_answer("llmcompiler", result) == "Paris"
```

- [ ] **Step 2: Run, FAIL**

```bash
pytest tests/sweep/test_dispatch_and_extract.py -v
```

- [ ] **Step 3: Implement minimal stub**

Create `/Users/imdonghyeon/agentic_rag/repro/sweep/agent_runner.py`:
```python
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
```

- [ ] **Step 4: Run, PASS**

```bash
pytest tests/sweep/test_dispatch_and_extract.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add repro/sweep/agent_runner.py repro/tests/sweep/test_dispatch_and_extract.py
git commit -m "feat: _dispatch + extract_final_answer stub"
```

---

### Task 21: HotpotQA sample loader

**Files:**
- Modify: `repro/sweep/agent_runner.py` (add `load_sample`)
- Create: `repro/tests/sweep/test_load_sample.py`

- [ ] **Step 1: Write failing test**

Create `/Users/imdonghyeon/agentic_rag/repro/tests/sweep/test_load_sample.py`:
```python
import os
import subprocess
import pytest
from pathlib import Path
from sweep.agent_runner import load_sample

@pytest.fixture(autouse=True, scope="module")
def _ensure_patches_applied():
    """Precondition: deterministic_select.patch + entry_points.patch must be applied,
    otherwise load_dataset behavior is unpredictable."""
    AB = Path("/Users/imdonghyeon/agentic_rag/AgentBench")
    # Quick check: deterministic_select.patch should make src/utils.py reference REPRO_SAMPLE_SEED
    content = (AB / "src/utils.py").read_text()
    if "REPRO_SAMPLE_SEED" not in content:
        pytest.skip("deterministic_select.patch not applied — run "
                    "`git apply repro/patches/deterministic_select.patch` first")
    yield

def test_load_sample_deterministic():
    os.environ["REPRO_SAMPLE_SEED"] = "42"
    s_a = load_sample(workload="hotpotqa", idx=0)
    s_b = load_sample(workload="hotpotqa", idx=0)
    assert s_a["_id"] == s_b["_id"]
    assert "question" in s_a and "answer" in s_a

def test_load_sample_indices_distinct():
    s0 = load_sample(workload="hotpotqa", idx=0)
    s1 = load_sample(workload="hotpotqa", idx=1)
    assert s0["_id"] != s1["_id"]
```

- [ ] **Step 2: Run, FAIL**

```bash
pytest tests/sweep/test_load_sample.py -v
```

- [ ] **Step 3: Add `load_sample` to `agent_runner.py`**

Append to `/Users/imdonghyeon/agentic_rag/repro/sweep/agent_runner.py`:
```python
def load_sample(*, workload: str, idx: int) -> dict:
    """Load one HotpotQA sample via AgentBench's patched load_dataset."""
    from src.utils import load_dataset
    data = load_dataset(workload, shuffle=False)
    return data[idx]
```

- [ ] **Step 4: Run, PASS**

```bash
pytest tests/sweep/test_load_sample.py -v
```
Expected: 2 passed.

(Note: This requires the patches from Tasks 8-13 to be applied. If they aren't, re-apply per Task 13 Step 6 before running this test.)

- [ ] **Step 5: Commit**

```bash
git add repro/sweep/agent_runner.py repro/tests/sweep/test_load_sample.py
git commit -m "feat: load_sample via patched AgentBench load_dataset"
```

---

### Task 22: `agent_runner.py` — full `run_one` with watchdog + finalization

**Files:**
- Modify: `repro/sweep/agent_runner.py` (add `run_one`, `finalize_trace`, `_default_timeout`)
- Create: `repro/tests/sweep/test_run_one_stub.py` (with mock dispatch)

- [ ] **Step 1: Write failing integration test (with mock dispatch)**

Create `/Users/imdonghyeon/agentic_rag/repro/tests/sweep/test_run_one_stub.py`:
```python
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
```

- [ ] **Step 2: Run, FAIL**

```bash
pytest tests/sweep/test_run_one_stub.py -v
```

- [ ] **Step 3: Implement `run_one` + `finalize_trace` + `_default_timeout`**

Append to `/Users/imdonghyeon/agentic_rag/repro/sweep/agent_runner.py`:
```python
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
    if agent_type == "lats" and iteration_limit >= 50:
        return 1200.0
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
```

(The `e2e_latency_s` field is set in `run_one` before calling `finalize_trace`; `finalize_trace` only derives quantities that depend on the spans + collector samples.)

- [ ] **Step 4: Run, PASS**

```bash
pytest tests/sweep/test_run_one_stub.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add repro/sweep/agent_runner.py repro/tests/sweep/test_run_one_stub.py
git commit -m "feat: run_one with watchdog + finalize_trace"
```

---

### Task 23: `sweep_runner.py` — main loop + resume + fsync

**Files:**
- Create: `repro/sweep/sweep_runner.py`
- Create: `repro/tests/sweep/test_sweep_runner.py`

- [ ] **Step 1: Write failing test (with mock run_one)**

Create `/Users/imdonghyeon/agentic_rag/repro/tests/sweep/test_sweep_runner.py`:
```python
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from measurement.trace_schema import QueryTrace
from sweep import sweep_runner

CFG = {
    "run_id": "test_run",
    "workload": "hotpotqa",
    "agent_type": "react",
    "defaults": {"fewshot": 5, "iteration_limit": 30},
    "sweeps": {"iteration_limit": [10, 20]},
    "samples_per_cell": 2,
    "sample_seed": 42,
}

def _fake_qt(cell):
    return QueryTrace(
        run_id="test_run", query_id=f"q-{cell.iteration_limit}-{cell.sample_idx}",
        agent_type=cell.agent_type, fewshot=cell.fewshot,
        iteration_limit=cell.iteration_limit, sample_idx=cell.sample_idx,
        correct=True, final_answer="x", expected_answer="x",
        e2e_latency_s=1.0, llm_total_ms=500, tool_total_ms=300, overhead_ms=200,
        prefill_total_ms=100, decode_total_ms=400,
        n_llm_calls=2, n_tool_calls=1,
        tokens_input_total=1000, tokens_output_total=200, tokens_input_max=500,
        kv_cache_max_tokens=100, kv_cache_mean_tokens=50.0, n_prompt_tokens_max=500,
        llm_calls=[], tool_calls=[],
    )

def test_full_run_writes_all_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_runner, "run_one_for_cell", lambda cell, _col: _fake_qt(cell))
    monkeypatch.setattr(sweep_runner, "_start_collector",
                        lambda: type("C", (), {"start": lambda s: s, "stop": lambda s: None})())
    out = tmp_path / "out.jsonl"
    sweep_runner.run_sweep(CFG, out_path=str(out), resume=False)
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 4   # 2 sweep values × 2 samples

def test_resume_skips_done_cells(tmp_path, monkeypatch):
    from sweep.cells import Cell
    out = tmp_path / "out.jsonl"
    # Pre-seed two done cells (iteration_limit=10, samples 0 and 1)
    done1 = _fake_qt(Cell(agent_type="react", fewshot=5, iteration_limit=10,
                          sample_idx=0, sweep_var_name="iteration_limit", sweep_var_val=10))
    done2 = _fake_qt(Cell(agent_type="react", fewshot=5, iteration_limit=10,
                          sample_idx=1, sweep_var_name="iteration_limit", sweep_var_val=10))
    out.write_text(done1.model_dump_json() + "\n" + done2.model_dump_json() + "\n")

    calls = []
    def _fake_run(cell, col):
        calls.append(cell.resume_key())
        return _fake_qt(cell)
    monkeypatch.setattr(sweep_runner, "run_one_for_cell", _fake_run)
    monkeypatch.setattr(sweep_runner, "_start_collector",
                        lambda: type("C", (), {"start": lambda s: s, "stop": lambda s: None})())
    sweep_runner.run_sweep(CFG, out_path=str(out), resume=True)
    # Should have only run the remaining 2 cells (iteration_limit=20, samples 0 and 1)
    assert len(calls) == 2
    # Each remaining call has signature (fewshot=5, iteration_limit=20)
    assert all(c[1] == (5, 20) for c in calls)
```

- [ ] **Step 2: Run, FAIL**

```bash
pytest tests/sweep/test_sweep_runner.py -v
```

- [ ] **Step 3: Implement `sweep_runner.py`**

Create `/Users/imdonghyeon/agentic_rag/repro/sweep/sweep_runner.py`:
```python
"""Top-level sweep runner: parses config, owns MetricsCollector lifecycle,
iterates cells, supports --resume via append-only JSONL with fsync."""
import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Iterable

from measurement.metrics_collector import MetricsCollector
from measurement.trace_schema import QueryTrace
from sweep.cells import Cell, enumerate_cells, ResumeKey
from sweep.agent_runner import run_one


def _start_collector(base_url: str = "http://localhost:8000") -> MetricsCollector:
    return MetricsCollector(base_url).start()


def run_one_for_cell(cell: Cell, collector: MetricsCollector) -> QueryTrace:
    """Thin wrapper around run_one that passes the cell's extra_kwargs through."""
    return run_one(
        agent_type=cell.agent_type,
        fewshot=cell.fewshot,
        iteration_limit=cell.iteration_limit,
        sample_idx=cell.sample_idx,
        collector=collector,
        extra_kwargs=cell.extra_kwargs,
    )


def _resume_key_from_row(row: dict) -> ResumeKey:
    """Reconstruct the canonical ResumeKey from a JSONL row.

    Must match the shape returned by Cell.resume_key() in sweep/cells.py:
    (agent_type, (fewshot, iteration_limit), sample_idx)
    """
    from sweep.cells import _signature
    return (
        row["agent_type"],
        _signature(row.get("fewshot", 0), row.get("iteration_limit", 0)),
        row["sample_idx"],
    )


def read_done_tuples(path: str) -> set[ResumeKey]:
    """Read existing JSONL and return the set of completed resume keys.
    Tolerates a malformed trailing line from a crash by skipping it.
    """
    if not os.path.exists(path):
        return set()
    done: set[ResumeKey] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # malformed line — likely truncated last row from a crash; ignore
                continue
            done.add(_resume_key_from_row(row))
    return done


def append_jsonl_fsync(path: str, trace: QueryTrace) -> None:
    line = trace.model_dump_json() + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def run_sweep(cfg: dict, *, out_path: str, resume: bool = True,
              base_url: str = "http://localhost:8000") -> None:
    """Execute one sweep config end-to-end."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cells = list(enumerate_cells(cfg))

    if resume:
        done = read_done_tuples(out_path)
        before = len(cells)
        cells = [c for c in cells if c.resume_key() not in done]
        skipped = before - len(cells)
        print(f"[resume] skipping {skipped}/{before} cells already done", file=sys.stderr)

    collector = _start_collector(base_url=base_url)
    try:
        for i, cell in enumerate(cells):
            print(f"[{i+1}/{len(cells)}] {cell.agent_type} "
                  f"{cell.sweep_var_name}={cell.sweep_var_val} sample={cell.sample_idx}",
                  file=sys.stderr, flush=True)
            trace = run_one_for_cell(cell, collector)
            trace.run_id = cfg.get("run_id", "")
            append_jsonl_fsync(out_path, trace)
    finally:
        collector.stop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to sweep YAML")
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--base-url", default="http://localhost:8000")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_sweep(cfg, out_path=args.out, resume=not args.no_resume, base_url=args.base_url)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run, PASS**

```bash
pytest tests/sweep/test_sweep_runner.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add repro/sweep/sweep_runner.py repro/tests/sweep/test_sweep_runner.py
git commit -m "feat: sweep_runner main loop with resume and fsync"
```

---

## Phase 4 — Sweep Configurations + Mini Sweep (Tasks 24–27)

### Task 24: Write sweep YAML configs

**Files:**
- Create: `repro/sweep/configs/fig13_pareto.yaml`
- Create: `repro/sweep/configs/fig14_iteration.yaml`
- Create: `repro/sweep/configs/fig15_fewshot.yaml`

- [ ] **Step 1: Write `fig13_pareto.yaml`**

```yaml
run_id: fig13_pareto
workload: hotpotqa
agent_types: [react, reflexion, lats, llmcompiler]
samples_per_agent:
  react: 50
  reflexion: 50
  lats: 25       # spec §8 — reduced due to ~17 min/query timing
  llmcompiler: 50
defaults:
  fewshot: 5
  iteration_limit: 30
  # Agent-specific defaults consumed via extra_kwargs:
  reflection_limit: 3       # Reflexion
  max_depth: 7              # LATS
  n_generate_sample: 5      # LATS
  max_replan: 20            # LLMCompiler
sample_seed: 42
```

- [ ] **Step 2: Write `fig14_iteration.yaml`**

```yaml
run_id: fig14_iteration_sweep
workload: hotpotqa
agent_type: react
defaults:
  fewshot: 5
  iteration_limit: 30
sweeps:
  iteration_limit: [5, 10, 15, 20, 30, 50, 75]
samples_per_cell: 50
sample_seed: 42
```

- [ ] **Step 3: Write `fig15_fewshot.yaml`**

```yaml
run_id: fig15_fewshot_sweep
workload: hotpotqa
agent_type: react
defaults:
  fewshot: 5
  iteration_limit: 30
sweeps:
  fewshot: [0, 1, 2, 3, 4, 5]
samples_per_cell: 50
sample_seed: 42
```

- [ ] **Step 4: Sanity-validate YAML loads + cell counts**

```bash
cd /Users/imdonghyeon/agentic_rag/repro
source .venv/bin/activate
python -c "
import yaml
from sweep.cells import enumerate_cells
for f in ['fig13_pareto', 'fig14_iteration', 'fig15_fewshot']:
    cfg = yaml.safe_load(open(f'sweep/configs/{f}.yaml'))
    n = sum(1 for _ in enumerate_cells(cfg))
    print(f'{f}: {n} cells')
"
```
Expected output:
```
fig13_pareto: 175 cells   (50+50+25+50)
fig14_iteration: 350 cells
fig15_fewshot: 300 cells
```
Total = 825 (matches spec §8).

- [ ] **Step 5: Commit**

```bash
git add repro/sweep/configs/
git commit -m "feat: sweep YAML configs for Fig 13/14/15"
```

---

### Task 25: Phase 1 smoke — single ReAct query end-to-end

**Files:**
- Create: `repro/tests/integration/test_phase1_smoke.py`

Note: This test requires `llama-server` running. The test harness starts and stops it.

- [ ] **Step 1: Write the integration test**

Create `/Users/imdonghyeon/agentic_rag/repro/tests/integration/test_phase1_smoke.py`:
```python
import json
import os
import subprocess
import time
import pytest
import requests
from pathlib import Path

REPO = Path("/Users/imdonghyeon/agentic_rag")
START = REPO / "repro/setup/start_server.sh"

@pytest.fixture(scope="module")
def llama_server():
    proc = subprocess.Popen([str(START)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    # Wait for /health
    for _ in range(60):
        try:
            r = requests.get("http://127.0.0.1:8000/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        proc.terminate()
        raise RuntimeError("llama-server did not start within 60s")
    yield "http://127.0.0.1:8000"
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def test_metrics_endpoint(llama_server):
    r = requests.get(f"{llama_server}/metrics", timeout=5)
    assert r.status_code == 200
    text = r.text
    assert "llamacpp:prompt_seconds_total" in text
    assert "llamacpp:tokens_predicted_seconds_total" in text


def test_slots_endpoint(llama_server):
    r = requests.get(f"{llama_server}/slots", timeout=5)
    assert r.status_code == 200
    slots = r.json()
    assert isinstance(slots, list)
    assert len(slots) >= 1
    assert "is_processing" in slots[0]
    assert "n_prompt_tokens_cache" in slots[0]


def test_single_react_query_completes(llama_server, tmp_path):
    """Run one HotpotQA query via run_one; verify trace fields populated."""
    from measurement.metrics_collector import MetricsCollector
    from sweep.agent_runner import run_one

    collector = MetricsCollector(llama_server).start()
    try:
        qt = run_one(
            agent_type="react", fewshot=5, iteration_limit=10,
            sample_idx=0, collector=collector,
            run_id="smoke",
        )
    finally:
        collector.stop()

    # Spec §9 Phase 2 gate criteria
    assert qt.n_llm_calls >= 1, f"got {qt.n_llm_calls} LLM calls"
    assert qt.tokens_output_total > 0
    assert qt.e2e_latency_s > 0
    assert isinstance(qt.correct, bool)
    assert qt.final_answer != ""
```

- [ ] **Step 2: Run smoke test**

```bash
cd /Users/imdonghyeon/agentic_rag/repro
source .venv/bin/activate
# Ensure patches are applied
cd /Users/imdonghyeon/agentic_rag/AgentBench
git apply --check /Users/imdonghyeon/agentic_rag/repro/patches/config.patch || \
    git apply /Users/imdonghyeon/agentic_rag/repro/patches/config.patch
# ... (and similarly for deterministic_select, entry_points)
cd /Users/imdonghyeon/agentic_rag/repro
pytest tests/integration/test_phase1_smoke.py -v -s
```
Expected: 3 passed. (First run will be slow due to model load — ~30s extra.)

- [ ] **Step 3: Inspect trace.txt for sanity**

```bash
cat /Users/imdonghyeon/agentic_rag/AgentBench/trace.txt | head -50
```
Expected: ReAct's Thought/Action/Observation log for one sample.

- [ ] **Step 4: Commit (integration test only — model and patches are gitignored / separately tracked)**

```bash
cd /Users/imdonghyeon/agentic_rag
git add repro/tests/integration/test_phase1_smoke.py
git commit -m "feat: Phase 1 smoke integration test"
```

---

### Task 26: Phase 3 mini sweep — 5 samples/cell across all 3 configs

**Files:**
- Create: `repro/tests/integration/test_phase3_mini_sweep.py`

Note: This is the Gate 3 check. It overrides each YAML to use 5 samples per cell instead of 50/25.

- [ ] **Step 1: Write override-and-run script**

Create `/Users/imdonghyeon/agentic_rag/repro/tests/integration/test_phase3_mini_sweep.py`:
```python
"""Phase 3 (Gate 3): run all 3 sweep configs at 5 samples/cell.
Verifies every cell completes and resume works.
"""
import copy
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import requests
import yaml

REPO = Path("/Users/imdonghyeon/agentic_rag")
CONFIGS_DIR = REPO / "repro/sweep/configs"
OUT_DIR = REPO / "repro/results/raw"

@pytest.fixture(scope="module")
def llama_server():
    # Reuse Phase 1 fixture
    from tests.integration.test_phase1_smoke import llama_server as ls
    yield from ls

def _override_samples(cfg: dict, n: int = 5) -> dict:
    cfg = copy.deepcopy(cfg)
    if "samples_per_cell" in cfg:
        cfg["samples_per_cell"] = n
    if "samples_per_agent" in cfg:
        for k in cfg["samples_per_agent"]:
            cfg["samples_per_agent"][k] = n
    return cfg

@pytest.mark.parametrize("cfg_file", ["fig14_iteration.yaml", "fig15_fewshot.yaml", "fig13_pareto.yaml"])
def test_mini_sweep_cell(llama_server, tmp_path, cfg_file):
    from sweep.cells import enumerate_cells
    from sweep.sweep_runner import run_sweep

    with open(CONFIGS_DIR / cfg_file) as f:
        cfg = _override_samples(yaml.safe_load(f), n=5)

    out = tmp_path / f"mini_{cfg['run_id']}.jsonl"
    run_sweep(cfg, out_path=str(out), resume=False, base_url=llama_server)

    # Verify row count
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    expected = sum(1 for _ in enumerate_cells(cfg))
    assert len(lines) == expected, f"{cfg_file}: expected {expected} rows, got {len(lines)}"

    # Verify every row parses
    for line in lines:
        row = json.loads(line)
        assert row["e2e_latency_s"] >= 0
        assert isinstance(row["correct"], bool)

def test_resume_after_kill(llama_server, tmp_path):
    """Run half the cells, kill, restart with --resume, verify completion."""
    from sweep.cells import enumerate_cells
    from sweep.sweep_runner import run_sweep

    with open(CONFIGS_DIR / "fig14_iteration.yaml") as f:
        cfg = _override_samples(yaml.safe_load(f), n=2)
    out = tmp_path / "resume_test.jsonl"

    # Phase A: run first half by truncating cells temporarily
    cfg_a = copy.deepcopy(cfg)
    cfg_a["sweeps"]["iteration_limit"] = cfg["sweeps"]["iteration_limit"][:3]
    run_sweep(cfg_a, out_path=str(out), resume=False, base_url=llama_server)
    first_count = len([l for l in out.read_text().splitlines() if l.strip()])
    assert first_count == 3 * 2  # 3 values × 2 samples

    # Phase B: resume with full config — should only run remaining cells
    run_sweep(cfg, out_path=str(out), resume=True, base_url=llama_server)
    full_count = len([l for l in out.read_text().splitlines() if l.strip()])
    assert full_count == 7 * 2  # 7 values × 2 samples (full sweep)
```

- [ ] **Step 2: Run mini sweep**

```bash
cd /Users/imdonghyeon/agentic_rag/repro
source .venv/bin/activate
pytest tests/integration/test_phase3_mini_sweep.py -v -s --timeout=14400
```
Expected: 4 tests pass within ~3-5 hours. (One per fig + resume).

- [ ] **Step 3: Extrapolate Phase 4 timing**

Run:
```bash
python - <<'EOF'
import json, statistics
from pathlib import Path
for p in Path("/Users/imdonghyeon/agentic_rag/repro/results/raw").glob("mini_*.jsonl"):
    if not p.exists(): continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    by_agent = {}
    for r in rows:
        by_agent.setdefault(r["agent_type"], []).append(r["e2e_latency_s"])
    for agent, lats in by_agent.items():
        m = statistics.mean(lats)
        # Extrapolate to full sample size
        full_n = {"lats": 25, "react": 50, "reflexion": 50, "llmcompiler": 50}.get(agent, 50)
        cell_total_s = m * full_n
        print(f"{p.name}/{agent}: mean {m:.1f}s, full cell {cell_total_s/60:.1f} min")
EOF
```
Expected: per-cell estimates. **Gate 3 criterion**: total extrapolated wall-clock ≤ 40h.

- [ ] **Step 4: Conditional — apply `tool_retry.patch` if Wikipedia errors observed**

Check the mini-sweep traces for Wikipedia tool errors:
```bash
python - <<'EOF'
import json
from pathlib import Path
n_total, n_tool_err, n_wiki_err = 0, 0, 0
for f in Path("/Users/imdonghyeon/agentic_rag/repro/results/raw").glob("mini_*.jsonl"):
    for line in f.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        n_total += 1
        for tc in row.get("tool_calls", []):
            if tc.get("error"):
                n_tool_err += 1
                if "wiki" in (tc.get("tool_name") or "").lower() or "429" in (tc.get("error") or ""):
                    n_wiki_err += 1
print(f"queries: {n_total}, tool errors: {n_tool_err}, wiki/429: {n_wiki_err}")
EOF
```

**If `wiki/429 > 0`**: create and apply `repro/patches/tool_retry.patch`. Wrap `WikipediaTool._run` / `_arun` in `src/tools/hotpotqa_tools/wikipedia.py` with `tenacity.retry`:
```python
# Add at top of file
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

# Decorate _run and _arun
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type((requests.HTTPError, requests.ConnectionError, TimeoutError)),
    reraise=True,
)
def _run(self, query: str, ...) -> str:
    # existing body
```
Generate the patch the same way as Tasks 8-13: `git diff src/tools/hotpotqa_tools/wikipedia.py > repro/patches/tool_retry.patch`, then commit and apply.

**If `wiki/429 == 0`**: skip — tool_retry.patch is unnecessary; comment that it was evaluated and not needed.

- [ ] **Step 5: Commit**

```bash
cd /Users/imdonghyeon/agentic_rag
git add repro/tests/integration/test_phase3_mini_sweep.py
# If Step 4 created the patch:
git add repro/patches/tool_retry.patch 2>/dev/null || true
git commit -m "feat: Phase 3 mini sweep integration test (+optional tool_retry)"
```

---

### Task 27: Phase 4 full sweep — overnight × 2 nights

**Files:**
- Create: `repro/sweep/run_full.sh`

This task is **execution-only** (no code besides the wrapper). It's documented for the engineer to understand the overnight workflow.

- [ ] **Step 1: Write `run_full.sh` wrapper**

Create `/Users/imdonghyeon/agentic_rag/repro/sweep/run_full.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPRO="$SCRIPT_DIR/.."
LOG_DIR="$REPRO/results/sweep_logs"
mkdir -p "$LOG_DIR"
ts="$(date +%Y%m%d_%H%M%S)"

cd "$REPRO"
source .venv/bin/activate

CONFIG="$1"
RUN_ID="$(/usr/bin/python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['run_id'])" "sweep/configs/$CONFIG")"
OUT="results/raw/${RUN_ID}.jsonl"

# Start llama-server in background if not running
if ! curl -s -o /dev/null http://localhost:8000/health; then
    echo "Starting llama-server..."
    ./setup/start_server.sh > "$LOG_DIR/llamaserver_$ts.log" 2>&1 &
    sleep 10
fi

# Run sweep with --resume (default)
echo "Running $CONFIG → $OUT"
python -m sweep.sweep_runner --config "sweep/configs/$CONFIG" --out "$OUT" \
    2>&1 | tee "$LOG_DIR/sweep_${RUN_ID}_$ts.log"

echo "Done. JSONL at: $OUT"
```

```bash
chmod +x /Users/imdonghyeon/agentic_rag/repro/sweep/run_full.sh
```

- [ ] **Step 2: Night 1 — Fig 14 + Fig 15 (ReAct sweeps, ~22h)**

```bash
cd /Users/imdonghyeon/agentic_rag/repro
nohup sweep/run_full.sh fig14_iteration.yaml > results/sweep_logs/night1_fig14.out 2>&1 &
wait
nohup sweep/run_full.sh fig15_fewshot.yaml > results/sweep_logs/night1_fig15.out 2>&1 &
wait
```

Verify:
```bash
wc -l results/raw/fig14_iteration_sweep.jsonl    # expect 350
wc -l results/raw/fig15_fewshot_sweep.jsonl      # expect 300
```

- [ ] **Step 3: Night 2 — Fig 13 (4-agent Pareto, ~12h)**

```bash
nohup sweep/run_full.sh fig13_pareto.yaml > results/sweep_logs/night2_fig13.out 2>&1 &
wait
wc -l results/raw/fig13_pareto.jsonl    # expect 175
```

- [ ] **Step 4: Verify Gate 4**

```bash
total=$(wc -l < results/raw/fig13_pareto.jsonl)
total=$((total + $(wc -l < results/raw/fig14_iteration_sweep.jsonl)))
total=$((total + $(wc -l < results/raw/fig15_fewshot_sweep.jsonl)))
echo "Total rows: $total (expect 825)"

# Timeout rate
python - <<'EOF'
import json
from pathlib import Path
files = list(Path("results/raw").glob("fig*.jsonl"))
n_total = 0
n_timeout = 0
for f in files:
    for line in f.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        n_total += 1
        if row.get("meta", {}).get("timeout"):
            n_timeout += 1
print(f"timeout rate: {n_timeout}/{n_total} = {100*n_timeout/n_total:.1f}%")
EOF
```
**Gate 4 criterion**: total = 825, timeout rate ≤ 5%.

- [ ] **Step 5: Commit logs (raw JSONL ignored, but log summaries can be useful)**

```bash
cd /Users/imdonghyeon/agentic_rag
git add repro/sweep/run_full.sh
git commit -m "feat: overnight sweep wrapper script"
```

---

## Phase 5 — Analysis (Tasks 28–32)

### Task 28: `analysis/shared.py` — common loading and styling

**Files:**
- Create: `repro/analysis/shared.py`

- [ ] **Step 1: Write `shared.py`**

```python
"""Common helpers for figure generation: load JSONL, plotting style."""
import json
from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

RESULTS_DIR = Path("/Users/imdonghyeon/agentic_rag/repro/results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(name: str) -> pd.DataFrame:
    """Load one of fig13_pareto / fig14_iteration_sweep / fig15_fewshot_sweep."""
    path = RESULTS_DIR / "raw" / f"{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def setup_plot_style() -> None:
    mpl.rcParams.update({
        "figure.figsize": (8, 5),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
    })


AGENT_COLORS = {
    "react": "#d62728",       # red
    "reflexion": "#bcbd22",   # olive
    "lats": "#1f77b4",        # blue
    "llmcompiler": "#7f7f7f", # gray
}
AGENT_MARKERS = {
    "react": "s", "reflexion": "o", "lats": "^", "llmcompiler": "D",
}
```

- [ ] **Step 2: Commit**

```bash
cd /Users/imdonghyeon/agentic_rag
git add repro/analysis/shared.py
git commit -m "feat: shared analysis helpers"
```

---

### Task 29: `plot_fig4.py` — LLM/tool call counts per agent

**Files:**
- Create: `repro/analysis/plot_fig4.py`

- [ ] **Step 1: Write script**

```python
"""Fig 4: mean LLM calls and tool calls per agent on HotpotQA."""
import matplotlib.pyplot as plt
from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR, AGENT_COLORS

def main():
    setup_plot_style()
    df = load_jsonl("fig13_pareto")
    agg = df.groupby("agent_type")[["n_llm_calls", "n_tool_calls"]].mean()
    order = ["react", "reflexion", "llmcompiler", "lats"]
    agg = agg.reindex(order)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(agg))
    w = 0.35
    ax.bar([i - w/2 for i in x], agg["n_llm_calls"], width=w,
           label="LLM calls", color="#d62728")
    ax.bar([i + w/2 for i in x], agg["n_tool_calls"], width=w,
           label="Tool calls", color="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index)
    ax.set_ylabel("Mean per query")
    ax.set_title("Fig 4: LLM and Tool calls per request (HotpotQA)")
    ax.legend()
    out = FIGURES_DIR / "fig4_calls.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # Verification criterion (spec §12)
    if "lats" not in agg.index or "react" not in agg.index:
        raise SystemExit(
            f"FAIL: missing agent rows. Have: {list(agg.index)}; expected react+lats"
        )
    react_mean = agg.loc["react", "n_llm_calls"]
    lats_mean = agg.loc["lats", "n_llm_calls"]
    ratio = lats_mean / react_mean if react_mean else 0
    print(f"LATS/ReAct LLM call ratio: {ratio:.2f} (must be ≥ 5.0)")
    assert ratio >= 5.0, f"FAIL: ratio {ratio:.2f} < 5.0"
    print("Fig 4 verification: PASS")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test script syntax + commit (real data later)**

```bash
cd /Users/imdonghyeon/agentic_rag/repro
python -c "import ast; ast.parse(open('analysis/plot_fig4.py').read()); print('syntax ok')"
git add repro/analysis/plot_fig4.py
git commit -m "feat: Fig 4 plot + LATS≥5×ReAct verification"
```

- [ ] **Step 3: Run after sweep completes**

```bash
cd /Users/imdonghyeon/agentic_rag/repro
source .venv/bin/activate
python -m analysis.plot_fig4
```
Expected: `Fig 4 verification: PASS`. PNG saved.

---

### Task 30: `plot_fig7.py` — 95th-percentile latency distribution

**Files:**
- Create: `repro/analysis/plot_fig7.py`

- [ ] **Step 1: Write script**

```python
"""Fig 7: e2e latency distribution for ReAct on HotpotQA."""
import numpy as np
import matplotlib.pyplot as plt
from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR

def main():
    setup_plot_style()
    df = load_jsonl("fig13_pareto")
    react_lat = df[df.agent_type == "react"]["e2e_latency_s"].values

    fig, ax = plt.subplots()
    ax.hist(react_lat, bins=20, density=True, color="#d62728", alpha=0.8)
    p50 = np.percentile(react_lat, 50)
    p95 = np.percentile(react_lat, 95)
    ax.axvline(p50, ls="--", color="black", label=f"p50 = {p50:.1f}s")
    ax.axvline(p95, ls="--", color="red", label=f"p95 = {p95:.1f}s")
    ax.set_xlabel("End-to-end latency (s)")
    ax.set_ylabel("Frequency density")
    ax.set_title("Fig 7: HotpotQA ReAct latency distribution")
    ax.legend()
    out = FIGURES_DIR / "fig7_latency_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    ratio = p95 / p50 if p50 else 0
    print(f"p95/p50 = {ratio:.2f} (must be ≥ 2.0)")
    assert ratio >= 2.0, f"FAIL: ratio {ratio:.2f} < 2.0"
    print("Fig 7 verification: PASS")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add repro/analysis/plot_fig7.py
git commit -m "feat: Fig 7 latency distribution + p95/p50 ≥ 2.0 verification"
```

---

### Task 31: `plot_fig13.py` — accuracy-latency Pareto

**Files:**
- Create: `repro/analysis/plot_fig13.py`

- [ ] **Step 1: Write script**

```python
"""Fig 13: accuracy vs e2e latency Pareto across 4 agents."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR, AGENT_COLORS, AGENT_MARKERS

def main():
    setup_plot_style()
    df = load_jsonl("fig13_pareto")
    agg = df.groupby("agent_type").agg(
        accuracy=("correct", "mean"),
        mean_latency=("e2e_latency_s", "mean"),
    )

    fig, ax = plt.subplots()
    for agent, row in agg.iterrows():
        ax.scatter(row["mean_latency"], 100*row["accuracy"],
                   s=180, color=AGENT_COLORS[agent], marker=AGENT_MARKERS[agent],
                   label=agent)
    ax.set_xlabel("Mean end-to-end latency (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Fig 13: Accuracy vs Latency Pareto on HotpotQA")
    ax.legend(title="Agent")
    out = FIGURES_DIR / "fig13_pareto.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # Verification (spec §12)
    rho, p = spearmanr(agg["mean_latency"], agg["accuracy"])
    print(f"Spearman ρ(latency, accuracy) = {rho:.3f} (must be ≥ 0.6)")
    react_acc = agg.loc["react", "accuracy"]
    lats_acc = agg.loc["lats", "accuracy"]
    react_lat = agg.loc["react", "mean_latency"]
    lats_lat = agg.loc["lats", "mean_latency"]
    print(f"LATS accuracy: {lats_acc:.3f}; ReAct accuracy: {react_acc:.3f}")
    print(f"LATS/ReAct accuracy ratio: {lats_acc/react_acc:.3f} (must be ≥ 0.9)")
    print(f"LATS latency: {lats_lat:.1f}s; ReAct latency: {react_lat:.1f}s")

    assert rho >= 0.6, f"FAIL: Spearman {rho:.3f} < 0.6"
    assert lats_acc >= 0.9 * react_acc, f"FAIL: LATS/ReAct {lats_acc/react_acc:.3f} < 0.9"
    assert lats_lat > react_lat, f"FAIL: LATS latency not > ReAct"
    print("Fig 13 verification: PASS")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add repro/analysis/plot_fig13.py
git commit -m "feat: Fig 13 Pareto + spec §12 verification gates"
```

---

### Task 32: `plot_fig14.py` and `plot_fig15.py`

**Files:**
- Create: `repro/analysis/plot_fig14.py`
- Create: `repro/analysis/plot_fig15.py`

- [ ] **Step 1: `plot_fig14.py`**

```python
"""Fig 14: iteration budget sweep (mean & p95 latency, accuracy) for ReAct."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR

def main():
    setup_plot_style()
    df = load_jsonl("fig14_iteration_sweep")
    agg = df.groupby("iteration_limit").agg(
        mean_lat=("e2e_latency_s", "mean"),
        p95_lat=("e2e_latency_s", lambda x: np.percentile(x, 95)),
        accuracy=("correct", "mean"),
    ).sort_index()

    fig, ax1 = plt.subplots()
    ax1.plot(agg.index, agg["mean_lat"], "o-", color="black", label="mean latency")
    ax1.plot(agg.index, agg["p95_lat"], "s--", color="gray", label="p95 latency")
    ax1.set_xlabel("Iteration budget")
    ax1.set_ylabel("Latency (s)")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(agg.index, 100*agg["accuracy"], "D-", color="red", label="accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="upper right")
    ax1.set_title("Fig 14: ReAct iteration budget sweep on HotpotQA")
    out = FIGURES_DIR / "fig14_iteration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # Verification: mean plateaus, p95 monotone Spearman ρ ≥ 0.8
    vals = sorted(agg.index)
    mean_lat = agg["mean_lat"].values
    p95_lat = agg["p95_lat"].values
    n = len(vals)
    lower_slope = (mean_lat[n//2] - mean_lat[0]) / (n//2) if n//2 else 0
    upper_slope = (mean_lat[-1] - mean_lat[n//2]) / max(n - n//2 - 1, 1)
    print(f"mean latency lower-half slope: {lower_slope:.2f}, upper-half slope: {upper_slope:.2f}")
    assert upper_slope < 0.25 * abs(lower_slope) + 0.5, "FAIL: mean latency did not plateau"

    rho, _ = spearmanr(vals, p95_lat)
    print(f"Spearman ρ(iter, p95_lat) = {rho:.3f} (must be ≥ 0.8)")
    assert rho >= 0.8, f"FAIL: p95 not monotone"
    print("Fig 14 verification: PASS")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `plot_fig15.py`**

```python
"""Fig 15: few-shot count sweep (accuracy & latency) for ReAct."""
import matplotlib.pyplot as plt
from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR

def main():
    setup_plot_style()
    df = load_jsonl("fig15_fewshot_sweep")
    agg = df.groupby("fewshot").agg(
        mean_lat=("e2e_latency_s", "mean"),
        accuracy=("correct", "mean"),
    ).sort_index()

    fig, ax1 = plt.subplots()
    ax1.plot(agg.index, agg["mean_lat"], "o-", color="black", label="mean latency")
    ax1.set_xlabel("Few-shot count")
    ax1.set_ylabel("Latency (s)")
    ax2 = ax1.twinx()
    ax2.plot(agg.index, 100*agg["accuracy"], "D-", color="red", label="accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax1.set_title("Fig 15: ReAct few-shot sweep on HotpotQA")
    fig.legend(loc="upper right")
    out = FIGURES_DIR / "fig15_fewshot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # Verification: accuracy is non-monotone OR plateaus by fewshot=5
    acc = agg["accuracy"].values
    n = len(acc)
    monotone_increasing = all(acc[i] <= acc[i+1] for i in range(n-1))
    plateau_in_upper_half = (acc[-1] - acc[n//2]) < 0.005 * (n - n//2)   # < 0.5%-pt per step
    non_monotone = (acc[-1] < max(acc))
    print(f"accuracies: {acc}")
    print(f"monotone increasing: {monotone_increasing}")
    print(f"non-monotone (max not at last): {non_monotone}")
    print(f"plateau in upper half: {plateau_in_upper_half}")
    assert non_monotone or plateau_in_upper_half, "FAIL: accuracy strictly increases"
    print("Fig 15 verification: PASS")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add repro/analysis/plot_fig14.py repro/analysis/plot_fig15.py
git commit -m "feat: Fig 14 + Fig 15 plots with verification gates"
```

- [ ] **Step 4: Run all plots after Phase 4 sweep completes**

```bash
cd /Users/imdonghyeon/agentic_rag/repro
source .venv/bin/activate
python -m analysis.plot_fig4
python -m analysis.plot_fig7
python -m analysis.plot_fig13
python -m analysis.plot_fig14
python -m analysis.plot_fig15
ls results/figures/
```
Expected: all 5 scripts print `verification: PASS`; 5 PNG files in `results/figures/`.

---

### Task 33: `README.md`

**Files:**
- Create: `repro/README.md`

- [ ] **Step 1: Write README**

```markdown
# HotpotQA Reproduction on M3 Pro

Local reproduction of Fig 4/7/13/14/15 from arXiv 2506.04301v2 (KAIST AgentBench paper).

**Spec**: `../docs/superpowers/specs/2026-05-25-hotpotqa-reproduction-design.md`
**Plan**: `../docs/superpowers/plans/2026-05-25-hotpotqa-reproduction.md`

## Quick start

```bash
# Phase 0 — Setup (~1 hour)
./setup/install_llamacpp.sh
./setup/download_model.sh
pyenv local 3.13.0
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Apply AgentBench patches
cd ../AgentBench
git apply ../repro/patches/config.patch
git apply ../repro/patches/deterministic_select.patch
git apply ../repro/patches/entry_points.patch

# Phase 1 — Smoke
cd ../repro
./setup/start_server.sh &
pytest tests/integration/test_phase1_smoke.py -v

# Phase 3 — Mini sweep (Gate 3)
pytest tests/integration/test_phase3_mini_sweep.py -v -s

# Phase 4 — Overnight × 2 nights
nohup sweep/run_full.sh fig14_iteration.yaml > results/sweep_logs/n1_fig14.out 2>&1
nohup sweep/run_full.sh fig15_fewshot.yaml > results/sweep_logs/n1_fig15.out 2>&1
nohup sweep/run_full.sh fig13_pareto.yaml > results/sweep_logs/n2_fig13.out 2>&1

# Phase 5 — Analysis
python -m analysis.plot_fig4
python -m analysis.plot_fig7
python -m analysis.plot_fig13
python -m analysis.plot_fig14
python -m analysis.plot_fig15
```

See spec §9 for phase gates and §12 for verification criteria.
```

- [ ] **Step 2: Commit**

```bash
cd /Users/imdonghyeon/agentic_rag
git add repro/README.md
git commit -m "docs: repro/ README with phase-by-phase quick start"
```

---

## Self-review

After all tasks above are complete, verify:

1. **Spec coverage** — each subsection of the spec has a task:
   - Spec §3.2 software stack → Tasks 2, 3, 4, 5, 7
   - Spec §3.3 time budget → Task 27 (full sweep)
   - Spec §4 architecture → emergent from Tasks 16–23
   - Spec §5 file structure → Task 1, then all subsequent
   - Spec §5.1 contract → Tasks 22, 23
   - Spec §5.2 final answer extraction → Task 20
   - Spec §5.3 canonical server startup → Task 6
   - Spec §6 schema → Task 14
   - Spec §7.1–7.4 measurement → Tasks 16, 17, 18, 22 (finalize_trace)
   - Spec §8 sweep matrix → Task 24
   - Spec §9 execution phases → Tasks 25, 26, 27
   - Spec §9.5 timeout diagnostics → Task 22 (watchdog in run_one)
   - Spec §10 out of scope → not implemented (by design)
   - Spec §11 risks — mitigations distributed across tasks
   - Spec §12 verification criteria → Tasks 29, 30, 31, 32

2. **No placeholders**: every code step shows actual code; every command shows expected output. Tasks 10–13 (entry_points patches) include explicit "discover the real AgentBench helper names" steps before writing code — the placeholder names in the template code MUST be replaced by the engineer using grep.

3. **Type consistency**:
   - `QueryTrace` schema is defined once in Task 14 and reused identically in Tasks 18, 22, 23, 28-32.
   - `ResumeKey` shape is `(agent_type, (fewshot, iteration_limit), sample_idx)` — defined by `_signature` in Task 19, returned by `Cell.resume_key()` (Task 19), reconstructed from JSONL by `_resume_key_from_row` (Task 23). All three must agree.
   - `n_reflections` incremented in Task 11 (Reflexion entry_points); `n_tree_expansions` set in Task 12 (LATS entry_points). Otherwise these schema fields stay at 0.
   - `max_tokens=2048` applied in all four entry_points (Tasks 10/11/12/13) — Q4 instability cap per spec §11.

4. **Spec §9.5 + §11 mitigations**:
   - KV eviction detection: implemented in `finalize_trace` (Task 22) and `MetricsCollector.detect_kv_eviction` (Task 17). Marks `meta.kv_eviction_detected=True`.
   - Force-restart watchdog: `_force_restart_llama_server()` in Task 22, triggered after 3 consecutive timeouts.
   - No-decode-progress detector: `MetricsCollector.detect_no_decode_progress` (Task 17). Caller wiring deferred — the in-process `request_timeout=600` + wall-clock timeout cover the common case; this detector is available for future use.
   - Wikipedia retry: `tool_retry.patch` conditionally created in Task 26 Step 4 based on Phase 3 observation.

If any of these check fails, edit the corresponding task before declaring the plan complete.

---

## Execution Handoff

**Plan complete and saved to** `/Users/imdonghyeon/agentic_rag/docs/superpowers/plans/2026-05-25-hotpotqa-reproduction.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
