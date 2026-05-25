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
