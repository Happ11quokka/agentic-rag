"""Phase 3 mini sweep integration test.

Per plan: 5 samples/cell exercise ~= 3-5 hours of compute on M3 Pro.
For this session: only `test_resume_after_kill` was run during implementation.

To run the full mini sweep (USER-TRIGGERED, ~3-5h):
    pytest tests/integration/test_phase3_mini_sweep.py -v -s --timeout=21600

Gate 3 criterion: every cell completes; row counts match; plot scripts run; resume works.

Note on Plan Task 26 Step 4 (Wikipedia tool retry patch):
    The conditional `tool_retry.patch` creation is DEFERRED until after the
    user runs the full mini sweep. The 429-rate analysis (see plan Step 4
    snippet) should be evaluated against the resulting JSONL traces. If
    `wiki/429 > 0`, create the patch as described in the plan; otherwise
    document that it was evaluated and skipped.
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
START = REPO / "repro/setup/start_server.sh"

# langchain-openai requires OPENAI_API_KEY even when talking to a local
# OpenAI-compatible server. Set a dummy key + point the SDK at llama-server.
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-local")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")


@pytest.fixture(scope="module")
def llama_server():
    """Same body as test_phase1_smoke.llama_server -- duplicated rather than
    re-exported because pytest fixtures are not directly iterable from
    another module."""
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

    # Warm up the slot so /slots reports the full schema.
    try:
        requests.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={
                "model": "Meta-Llama-3.1-8B-Instruct-Q4_K_M",
                "messages": [{"role": "user", "content": "Say OK."}],
                "max_tokens": 4,
                "temperature": 0.0,
            },
            timeout=120,
        )
    except Exception:
        pass

    yield "http://127.0.0.1:8000"
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


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
    assert first_count == 3 * 2  # 3 values x 2 samples

    # Phase B: resume with full config -- should only run remaining cells
    run_sweep(cfg, out_path=str(out), resume=True, base_url=llama_server)
    full_count = len([l for l in out.read_text().splitlines() if l.strip()])
    assert full_count == 7 * 2  # 7 values x 2 samples (full sweep)
