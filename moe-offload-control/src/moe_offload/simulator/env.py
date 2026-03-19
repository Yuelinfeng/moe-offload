"""Simulator environment — minimal main simulation loop.

Extreme-minimal implementation: no event loop, no callback system,
no observation-space abstraction.  Designed to be driven step-by-step
from test code or a runner.
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
    """Minimal trace-driven simulation environment.

    Usage::

        env = SimulatorEnv(config)
        state = env.reset(episode)
        while not env.is_done():
            decision = some_controller(state)
            state, metrics = env.step(decision)

    Parameters
    ----------
    config : dict
        Must contain at minimum:
        - ``memory_capacity`` (int)
        - ``bandwidth_capacity`` (int)

        Optional keys:
        - ``per_expert_latency`` (float, default 1.0)
        - ``alpha_transfer`` (float, default 0.1)
        - ``alpha_misprefetch`` (float, default 1.0)
        - ``alpha_reload`` (float, default 1.0)
        - ``misprefetch_window`` (int, default 3)
        - ``reload_window`` (int, default 3)
        - ``pinned_experts`` (list[int], default [])
    """

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

        # Set after reset()
        self._episode: TraceEpisode | None = None
        self._state: SimulatorState | None = None
        self._step_cursor: int = 0

    def reset(self, episode: TraceEpisode) -> SimulatorState:
        """Initialize the environment with a new trace episode.

        Returns the initial :class:`SimulatorState`.
        """
        validate_episode(episode)
        self._episode = episode
        self._step_cursor = 0

        # Initialize state
        self._state = make_initial_state(self._config)

        # Sync memory model with initial pinned experts
        pinned = set(self._config.get("pinned_experts", []))
        self._memory.reset(initial_residents=pinned)

        return self._state

    def step(
        self, decision: ControllerDecision
    ) -> tuple[SimulatorState, StepMetrics]:
        """Advance the simulation by one step.

        Parameters
        ----------
        decision : ControllerDecision
            The controller's fetch / evict / defer action.

        Returns
        -------
        state : SimulatorState
            Updated state *after* applying the decision.
        metrics : StepMetrics
            Metrics for this step.

        Raises
        ------
        RuntimeError
            If the environment has not been reset or the episode is done.
        """
        if self._episode is None or self._state is None:
            raise RuntimeError("Must call reset() before step()")
        if self.is_done():
            raise RuntimeError("Episode is already done")

        ground_truth = self._episode.steps[self._step_cursor]
        future_steps = self._episode.steps[self._step_cursor + 1 :]

        # --- Compute metrics BEFORE applying decision (using pre-state) ---
        metrics = self._cost.compute(
            decision=decision,
            state=self._state,
            ground_truth=ground_truth,
            future_steps=future_steps,
        )

        # --- Apply decision to memory model ---
        # Evict first, then fetch (order matters for capacity)
        pinned = self._state.pinned_experts
        safe_evict = [eid for eid in decision.evict_experts if eid not in pinned]
        self._memory.release(safe_evict)
        to_fetch = [
            eid
            for eid in decision.fetch_experts
            if not self._memory.is_resident(eid)
        ]
        if to_fetch:
            # Respect capacity limits when applying controller-directed fetches.
            available_slots = self._memory.capacity - self._memory.current_usage()
            if available_slots > 0:
                alloc = to_fetch[:available_slots]
                if alloc:
                    self._memory.allocate(alloc)

        # Demand-fetch missed active experts (so they become resident for
        # the next step even though they caused a stall this step)
        active = set(ground_truth.active_experts)
        demand_fetch = [
            eid for eid in active
            if not self._memory.is_resident(eid)
        ]
        if demand_fetch:
            # Need space — evict oldest non-pinned non-active if needed.
            # This block is intentionally conservative and may leave some
            # active experts non-resident if capacity is fully consumed by
            # pinned or currently-active experts.
            while not self._memory.can_fit(len(demand_fetch)):
                evictable = (
                    self._memory.resident_set() - pinned - active
                )
                if not evictable:
                    break  # cannot make room beyond what is already available
                victim = min(evictable)  # deterministic: pick lowest ID
                self._memory.release([victim])

            # Allocate only as many demand-fetched experts as can fit.
            available_slots = self._memory.capacity - self._memory.current_usage()
            if available_slots > 0:
                to_allocate = demand_fetch[:available_slots]
                if to_allocate:
                    self._memory.allocate(to_allocate)

        # --- Build new state ---
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
