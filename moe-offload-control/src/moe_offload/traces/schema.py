"""Trace schema — canonical data definitions for MoE offloading traces.

This is the sole authoritative definition file for all trace-related
dataclasses.  Every other module that needs these types must import
from here (or from ``moe_offload.traces``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Core trace dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TraceStep:
    """A single routing decision at one layer for one token-batch."""

    step_idx: int
    """Global monotonically increasing step index (0-based)."""

    layer_id: int
    """Layer index, must be in ``[0, num_layers)``."""

    active_experts: list[int]
    """Expert IDs activated at this step, each in ``[0, num_experts)``."""

    regime_id: str | None = None
    """Optional workload regime label (e.g. ``'locality'``, ``'drift'``)."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Arbitrary extra fields."""


@dataclass
class ForecastStep:
    """A predictor's output for one future step at one layer."""

    layer_id: int
    """Layer index."""

    expert_probs: dict[int, float]
    """Predicted activation probability per expert."""

    uncertainty: dict[int, float]
    """Uncertainty estimate per expert."""


@dataclass
class ForecastWindow:
    """A window of forecast steps starting from a given position."""

    current_step_idx: int
    """The step index this forecast was made at."""

    horizon: int
    """Number of future steps in the window."""

    steps: list[ForecastStep]
    """Forecast for each future step."""


@dataclass
class WorkloadRegime:
    """Metadata describing a workload regime (phase) within a trace."""

    regime_id: str
    """Unique string identifier for this regime."""

    description: str = ""
    """Human-readable description."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Arbitrary extra fields."""


@dataclass
class TraceEpisode:
    """A complete trace episode — the input to a simulation run."""

    steps: list[TraceStep]
    """Ordered sequence of trace steps."""

    num_experts: int
    """Total number of experts in the model."""

    num_layers: int
    """Total number of MoE layers."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Arbitrary extra fields (e.g. source, generation params)."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_episode(episode: TraceEpisode) -> None:
    """Validate a :class:`TraceEpisode` against the schema invariants.

    Raises :class:`ValueError` on any violation:
    1. ``steps`` must contain at least 1 step.
    2. ``step_idx`` values must be monotonically increasing.
    3. Every expert id in ``active_experts`` must be in ``[0, num_experts)``.
    4. Every ``layer_id`` must be in ``[0, num_layers)``.
    """
    if not episode.steps:
        raise ValueError("TraceEpisode must contain at least 1 step")
    if episode.num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if episode.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    
    prev_idx: int | None = None
    for i, step in enumerate(episode.steps):
        # Monotonically increasing step_idx
        if prev_idx is not None and step.step_idx <= prev_idx:
            raise ValueError(
                f"step_idx must be monotonically increasing: "
                f"step[{i}].step_idx={step.step_idx} <= previous {prev_idx}"
            )
        prev_idx = step.step_idx

        # layer_id in range
        if not (0 <= step.layer_id < episode.num_layers):
            raise ValueError(
                f"step[{i}].layer_id={step.layer_id} out of range "
                f"[0, {episode.num_layers})"
            )

        # expert ids in range
        for eid in step.active_experts:
            if not (0 <= eid < episode.num_experts):
                raise ValueError(
                    f"step[{i}] has expert_id={eid} out of range "
                    f"[0, {episode.num_experts})"
                )
