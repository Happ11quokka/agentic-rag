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
