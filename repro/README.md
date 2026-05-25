# HotpotQA Reproduction on M3 Pro

Local reproduction of Fig 4/5/7/8/9/13/14/15/16 from arXiv 2506.04301v2 (KAIST AgentBench paper).

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

## Additional figures (Fig 5, 8, 9, 16)

Four extra figures from the paper. Fig 5 and Fig 8 reuse the existing
`fig13_pareto.jsonl`; Fig 9 needs a second sweep with prefix caching on;
Fig 16 has its own three sweeps.

### Extra sweep configs

| Config                              | Cells | Purpose                                                |
|-------------------------------------|-------|--------------------------------------------------------|
| `fig13_pareto_cache_on.yaml`        | 175   | Pareto rerun with `llama-server --cache-reuse 256`     |
| `fig16a_reflexion_sequential.yaml`  | 100   | Reflexion sweep: `reflection_limit ∈ {2,4,8,16}`       |
| `fig16b_lats_sequential.yaml`       |  40   | LATS sweep: `iteration_limit ∈ {4,8,16,32}` (64 cut)   |
| `fig16c_lats_parallel.yaml`         |  40   | LATS sweep: `n_generate_sample ∈ {1,2,4,8}`            |

Note: `fig16b` omits `iteration_limit=64` — too slow on M3 Pro Q4_K_M.

### Run commands

```bash
# Fig 9 (prefix caching). MUST pass `cache_on` as the second arg so
# run_full.sh starts the right llama-server variant.
nohup sweep/run_full.sh fig13_pareto_cache_on.yaml cache_on \
    > results/sweep_logs/n3_fig9.out 2>&1

# Fig 16 (three independent sweeps; can run sequentially overnight)
nohup sweep/run_full.sh fig16a_reflexion_sequential.yaml \
    > results/sweep_logs/n4_fig16a.out 2>&1
nohup sweep/run_full.sh fig16b_lats_sequential.yaml \
    > results/sweep_logs/n5_fig16b.out 2>&1
nohup sweep/run_full.sh fig16c_lats_parallel.yaml \
    > results/sweep_logs/n6_fig16c.out 2>&1
```

### Plot commands

```bash
python -m analysis.plot_fig5      # needs fig13_pareto.jsonl
python -m analysis.plot_fig8      # needs fig13_pareto.jsonl with tokens_by_role
python -m analysis.plot_fig9      # needs fig13_pareto + fig13_pareto_cache_on
python -m analysis.plot_fig16     # needs 3x fig16{a,b,c} JSONLs
```

Each plot script is independent — `plot_fig9` exits cleanly with a
`SKIPPED` message if its cache-on input is missing; `plot_fig16` annotates
any missing panel rather than failing the whole figure.
