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
