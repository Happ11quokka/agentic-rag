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
                        lambda **_: type("C", (), {"start": lambda s: s, "stop": lambda s: None})())
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
                        lambda **_: type("C", (), {"start": lambda s: s, "stop": lambda s: None})())
    sweep_runner.run_sweep(CFG, out_path=str(out), resume=True)
    # Should have only run the remaining 2 cells (iteration_limit=20, samples 0 and 1)
    assert len(calls) == 2
    # Each remaining call has signature (fewshot=5, iteration_limit=20)
    assert all(c[1] == (5, 20) for c in calls)
