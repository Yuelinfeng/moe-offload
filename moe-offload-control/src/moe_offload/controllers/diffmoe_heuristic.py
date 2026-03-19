from __future__ import annotations

from typing import Dict, List, Set

from moe_offload.simulator.state import ControllerDecision, SimulatorState
from moe_offload.traces.schema import ForecastWindow, TraceStep

from .base import BaseController


class DiffMoeHeuristicController(BaseController):
    """Diff-MoE-style heuristic controller (semantic approximation).

    Phase 2 version keeps only the semantic core:

    - A global hot expert set is provided explicitly via the constructor
      and treated as pinned from the controller's perspective.
    - Each expert maintains a simple local priority score based on
      recent activations.
    - Admission rule: prefetch experts whose priority exceeds a fixed
      threshold and that are predicted to be used soon.
    - Next-step style: prefers experts in the near forecast horizon.
    """

    def __init__(
        self,
        global_hot_experts: Set[int] | List[int],
        priority_increment: float = 1.0,
        priority_decay: float = 0.9,
        admission_threshold: float = 1.5,
        lookahead_steps: int = 1,
    ) -> None:
        # Global hot experts are passed in explicitly per Phase 2 rules.
        self._global_hot: Set[int] = set(global_hot_experts)
        self._priority_increment = float(priority_increment)
        self._priority_decay = float(priority_decay)
        self._admission_threshold = float(admission_threshold)
        self._lookahead_steps = int(lookahead_steps)

        # Simple in-controller priority table: expert_id -> score.
        self._priority: Dict[int, float] = {}

    def reset(self) -> None:
        self._priority.clear()

    def _bump_priorities(self, active_experts: List[int]) -> None:
        # Decay all existing scores.
        for eid in list(self._priority.keys()):
            self._priority[eid] *= self._priority_decay
            if self._priority[eid] < 1e-6:
                del self._priority[eid]

        # Increase priority for currently active experts.
        for eid in active_experts:
            self._priority[eid] = self._priority.get(eid, 0.0) + self._priority_increment

    def plan(
        self,
        state: SimulatorState,
        forecast: ForecastWindow | None,
        trace_step: TraceStep | None = None,
    ) -> ControllerDecision:
        resident: Set[int] = set(state.resident_experts)
        pinned: Set[int] = set(state.pinned_experts) | self._global_hot

        # We never explicitly evict; env handles capacity and also
        # prevents eviction of pinned experts.
        fetch: List[int] = []

        if forecast is not None and forecast.steps:
            # Consider only the first few forecast steps as \"next-step\"
            # style prefetch.
            window_steps = forecast.steps[: self._lookahead_steps]

            # Aggregate simple occurrence counts across the short horizon.
            candidate_scores: Dict[int, float] = {}
            for f_step in window_steps:
                for eid, prob in f_step.expert_probs.items():
                    if prob <= 0.0:
                        continue
                    candidate_scores[eid] = candidate_scores.get(eid, 0.0) + float(prob)

            # Decide which experts to admit based on local priority.
            for eid, _ in sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0])):
                if eid in resident or eid in pinned:
                    continue
                priority = self._priority.get(eid, 0.0)
                if priority >= self._admission_threshold:
                    fetch.append(eid)

        return ControllerDecision(
            fetch_experts=fetch,
            evict_experts=[],
            defer_experts=[],
            metadata={},
        )

    def observe(
        self,
        prev_state: SimulatorState,
        decision: ControllerDecision,
        trace_step: TraceStep,
        metrics: "StepMetrics",
        next_state: SimulatorState,
    ) -> None:
        # Update local priorities using the ground-truth active experts.
        self._bump_priorities(trace_step.active_experts)

