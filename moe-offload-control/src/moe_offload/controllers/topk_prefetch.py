from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Set

from moe_offload.simulator.state import ControllerDecision, SimulatorState
from moe_offload.traces.schema import ForecastStep, ForecastWindow, TraceStep

from .base import BaseController


class TopKPrefetchController(BaseController):
    """Simple prediction-triggered top-k prefetch controller.

    Aggregates expert probability across the forecast horizon and
    prefetches the top-k non-resident, non-pinned experts. Does not
    perform any explicit eviction; relies on the environment for
    capacity management and demand fetch.
    """

    def __init__(self, k: int | None = None) -> None:
        # If k is None, we will default to bandwidth_capacity at runtime.
        self._k = k

    def _aggregate_scores(self, forecast: ForecastWindow) -> Counter:
        scores: Counter = Counter()
        for step in forecast.steps:
            self._accumulate_step(scores, step)
        return scores

    @staticmethod
    def _accumulate_step(scores: Counter, step: ForecastStep) -> None:
        for eid, p in step.expert_probs.items():
            scores[eid] += float(p)

    def plan(
        self,
        state: SimulatorState,
        forecast: ForecastWindow | None,
        trace_step: TraceStep | None = None,
    ) -> ControllerDecision:
        if forecast is None or not forecast.steps:
            return ControllerDecision(
                fetch_experts=[],
                evict_experts=[],
                defer_experts=[],
                metadata={},
            )

        scores = self._aggregate_scores(forecast)

        resident: Set[int] = set(state.resident_experts)
        pinned: Set[int] = set(state.pinned_experts)

        # Default k to bandwidth capacity if not explicitly provided.
        k = self._k if self._k is not None else int(state.bandwidth_capacity)
        if k <= 0:
            fetch: List[int] = []
        else:
            # Filter to non-resident, non-pinned experts with positive score.
            candidates: Iterable[tuple[int, float]] = (
                (eid, score)
                for eid, score in scores.items()
                if eid not in resident and eid not in pinned and score > 0.0
            )
            # Sort by descending score, then by expert id for determinism.
            sorted_eids = sorted(
                candidates,
                key=lambda item: (-item[1], item[0]),
            )
            fetch = [eid for eid, _ in sorted_eids[:k]]

        return ControllerDecision(
            fetch_experts=fetch,
            evict_experts=[],
            defer_experts=[],
            metadata={},
        )

