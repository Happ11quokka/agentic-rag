import json
import os
import subprocess
import time
import pytest
import requests
from pathlib import Path

REPO = Path("/Users/imdonghyeon/agentic_rag")
START = REPO / "repro/setup/start_server.sh"

# langchain-openai requires OPENAI_API_KEY even when talking to a local
# OpenAI-compatible server. Set a dummy key + point the SDK at llama-server.
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-local")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")


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

    # Warm up the slot so /slots reports the full schema (n_prompt_tokens_cache,
    # etc. only appear after the slot has processed at least one task).
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
