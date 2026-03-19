"""Simulator environment — minimal main simulation loop.

Extreme-minimal implementation: no event loop, no callback system,
no observation-space abstraction. Designed to be driven step-by-step
from test code or a runner.

Important
---------
In this minimal simulator:
- memory acts like a hard capacity constraint
- bandwidth acts like a soft per-step cap that induces overflow stall
- the environment normalizes requested controller actions into an
  *effective decision* before both costing and state transition
"""

from __future__ import annotations

from moe_offload.simulator.bandwidth_model import BandwidthModel
from moe_offload.simulator.cost_model import CostModel
from moe_offload.simulator.memory_model import MemoryModel
from moe_offload.simulator.state import (
    ControllerDecision,
    SimulatorState,
    StepMetrics,
    make_initial_state,
)
from moe_offload.traces.schema import TraceEpisode, validate_episode


class SimulatorEnv:
    """Minimal trace-driven simulation environment."""

    def __init__(self, config: dict) -> None:
        self._config = dict(config)  # defensive copy
        self._memory = MemoryModel(capacity=config["memory_capacity"])
        self._bandwidth = BandwidthModel(
            capacity=config["bandwidth_capacity"],
            per_expert_latency=config.get("per_expert_latency", 1.0),
        )
        self._cost = CostModel(
            alpha_transfer=config.get("alpha_transfer", 0.1),
            alpha_misprefetch=config.get("alpha_misprefetch", 1.0),
            alpha_reload=config.get("alpha_reload", 1.0),
            misprefetch_window=config.get("misprefetch_window", 3),
            reload_window=config.get("reload_window", 3),
            bandwidth_model=self._bandwidth,
        )

        self._episode: TraceEpisode | None = None
        self._state: SimulatorState | None = None
        self._step_cursor: int = 0

    def reset(self, episode: TraceEpisode) -> SimulatorState:
        """Initialize the environment with a new trace episode."""
        validate_episode(episode)
        self._episode = episode
        self._step_cursor = 0

        self._state = make_initial_state(self._config)

        pinned = set(self._config.get("pinned_experts", []))
        self._memory.reset(initial_residents=pinned)

        return self._state

    def step(
        self, decision: ControllerDecision
    ) -> tuple[SimulatorState, StepMetrics]:
        """Advance the simulation by one step.

        The environment first converts the requested decision into an
        *effective decision* that reflects what can actually happen:
        - pinned experts cannot be evicted
        - non-resident experts cannot be evicted
        - already-resident experts are not re-fetched
        - explicit fetches must fit after effective eviction
        """
        if self._episode is None or self._state is None:
            raise RuntimeError("Must call reset() before step()")
        if self.is_done():
            raise RuntimeError("Episode is already done")

        ground_truth = self._episode.steps[self._step_cursor]
        future_steps = self._episode.steps[self._step_cursor + 1 :]

        effective_decision = self._normalize_decision(self._state, decision)

        # Compute metrics using the effective decision so costing matches execution
        metrics = self._cost.compute(
            decision=effective_decision,
            state=self._state,
            ground_truth=ground_truth,
            future_steps=future_steps,
        )

        # Apply effective decision to memory model
        if effective_decision.evict_experts:
            self._memory.release(effective_decision.evict_experts)

        if effective_decision.fetch_experts:
            try:
                self._memory.allocate(effective_decision.fetch_experts)
            except ValueError as exc:
                raise ValueError(
                    "Illegal controller decision: explicit fetch exceeds memory capacity"
                ) from exc

        # Demand-fetch missed active experts so they become resident for the next step.
        # Since demand_fetch only includes currently non-resident experts, can_fit(len(...))
        # is semantically correct here.
        active = set(ground_truth.active_experts)
        pinned = self._state.pinned_experts

        demand_fetch = [
            eid for eid in active
            if not self._memory.is_resident(eid)
        ]

        if demand_fetch:
            while not self._memory.can_fit(len(demand_fetch)):
                evictable = self._memory.resident_set() - pinned - active
                if not evictable:
                    break
                victim = min(evictable)  # deterministic minimal victim for smoke-test stability
                self._memory.release([victim])

            try:
                self._memory.allocate(demand_fetch)
            except ValueError as exc:
                raise RuntimeError(
                    "Demand fetch failed after attempting to free capacity"
                ) from exc

        # Build next state
        self._step_cursor += 1
        self._state = SimulatorState(
            step_idx=self._step_cursor,
            resident_experts=self._memory.resident_set(),
            pinned_experts=set(pinned),
            memory_used=self._memory.current_usage(),
            memory_capacity=self._state.memory_capacity,
            bandwidth_capacity=self._state.bandwidth_capacity,
            controller_context=self._state.controller_context,
        )

        return self._state, metrics

    def _normalize_decision(
        self,
        state: SimulatorState,
        decision: ControllerDecision,
    ) -> ControllerDecision:
        """Convert a requested controller decision into an effective decision.

        Rules:
        1. Only currently resident, non-pinned experts may be effectively evicted.
        2. Effective fetches are computed after effective eviction.
        3. Explicit fetches that would exceed memory capacity are rejected.
        4. Duplicate expert IDs are removed while preserving deterministic order.
        """
        requested_evict = self._unique_preserve_order(decision.evict_experts)
        requested_fetch = self._unique_preserve_order(decision.fetch_experts)
        requested_defer = self._unique_preserve_order(decision.defer_experts)

        if set(requested_evict) & set(requested_fetch):
            raise ValueError(
                "Illegal controller decision: same expert appears in both fetch and evict"
            )

        current_residents = set(state.resident_experts)
        pinned = set(state.pinned_experts)

        # Effective evictions: only resident and non-pinned experts
        effective_evict = [
            eid for eid in requested_evict
            if eid in current_residents and eid not in pinned
        ]

        residents_after_evict = current_residents - set(effective_evict)

        # Effective fetches: only experts not resident after effective eviction
        effective_fetch = [
            eid for eid in requested_fetch
            if eid not in residents_after_evict
        ]

        # Memory feasibility check for explicit fetches
        post_explicit_usage = len(residents_after_evict) + len(effective_fetch)
        if post_explicit_usage > state.memory_capacity:
            raise ValueError(
                "Illegal controller decision: explicit fetch exceeds memory capacity "
                f"({post_explicit_usage} > {state.memory_capacity})"
            )

        return ControllerDecision(
            fetch_experts=effective_fetch,
            evict_experts=effective_evict,
            defer_experts=requested_defer,
            metadata=dict(decision.metadata),
        )

    @staticmethod
    def _unique_preserve_order(items: list[int]) -> list[int]:
        """Remove duplicates while preserving original order."""
        seen: set[int] = set()
        result: list[int] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def is_done(self) -> bool:
        """Return True if all steps in the episode have been consumed."""
        if self._episode is None:
            return True
        return self._step_cursor >= len(self._episode.steps)

    @property
    def current_step_idx(self) -> int:
        return self._step_cursor

    @property
    def episode_length(self) -> int:
        if self._episode is None:
            return 0
        return len(self._episode.steps)