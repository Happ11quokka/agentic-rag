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
