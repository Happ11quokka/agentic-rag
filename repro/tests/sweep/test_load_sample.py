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
