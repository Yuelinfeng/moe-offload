"""Simulator state — canonical data definitions for simulation state,
controller decisions, step metrics, and episode results.

This is the sole authoritative definition file for all simulator-side
dataclasses.  Other modules import from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Core simulator dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SimulatorState:
    """Snapshot of the simulator at a single timestep."""

    step_idx: int
    """Current step index in the trace episode."""

    resident_experts: set[int]
    """Set of expert IDs currently loaded in GPU memory."""

    pinned_experts: set[int]
    """Set of expert IDs that cannot be evicted by normal eviction."""

    memory_used: int | float
    """Number of expert slots currently occupied."""

    memory_capacity: int | float
    """Maximum number of expert slots available."""

    bandwidth_capacity: int | float
    """Maximum number of experts that can be transferred per step."""

    # controller_context: dict = field(default_factory=dict)
    controller_context: dict[str, object] = field(default_factory=dict)
    """Opaque dict for controller-specific bookkeeping across steps."""


@dataclass
class ControllerDecision:
    """A controller's action for a single timestep."""

    fetch_experts: list[int]
    """Expert IDs to load into GPU memory this step."""

    evict_experts: list[int]
    """Expert IDs to remove from GPU memory this step."""

    defer_experts: list[int]
    """Expert IDs the controller chose to defer (not act on now)."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Arbitrary extra fields for diagnostics / logging."""


@dataclass
class StepMetrics:
    """Metrics collected for a single simulation step."""

    stall_latency: float
    """Latency stall due to cache misses and bandwidth overflow."""

    transfer_cost: float
    """Total transfer cost (number of experts fetched)."""

    misprefetch_count: int
    """Experts fetched but unused within the lookahead window."""

    reload_count: int
    """Experts evicted then needed again within the lookahead window."""

    cache_miss_count: int
    """Active experts not found in the resident set."""

    service_cost: float
    """Composite cost: stall + weighted transfer + weighted misprefetch + weighted reload."""


@dataclass
class EpisodeResult:
    """Summary result for a complete simulation episode."""

    config_id: str
    """Identifier for the configuration used in this run."""

    controller_name: str
    """Name of the controller used."""

    predictor_name: str
    """Name of the predictor used."""

    aggregated_metrics: dict[str, float]
    """Aggregated metrics across all steps (e.g. mean_stall, total_cost)."""

    step_records_path: str | None = None
    """Optional path to a file containing per-step records."""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def make_initial_state(config: dict) -> SimulatorState:
    """Create a fresh :class:`SimulatorState` from a config dict.

    Expected keys in *config*:
    - ``memory_capacity`` (int): max expert slots
    - ``bandwidth_capacity`` (int): max experts transferred per step
    - ``pinned_experts`` (list[int], optional): initially pinned expert IDs
    """
    pinned = set(config.get("pinned_experts", []))
    return SimulatorState(
        step_idx=0,
        resident_experts=set(pinned),  # pinned experts start resident
        pinned_experts=pinned,
        memory_used=len(pinned),
        memory_capacity=config["memory_capacity"],
        bandwidth_capacity=config["bandwidth_capacity"],
        controller_context={},
    )
