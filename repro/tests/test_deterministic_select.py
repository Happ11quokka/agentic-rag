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
    # load_dataset uses relative paths like "dataset/hotpot_dev_fullwiki_v1.json"
    prev_cwd = os.getcwd()
    os.chdir(AB)
    try:
        data = load_dataset("hotpotqa", shuffle=False)
    finally:
        os.chdir(prev_cwd)
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
