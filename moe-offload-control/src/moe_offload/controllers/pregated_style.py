from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Set

from moe_offload.simulator.state import ControllerDecision, SimulatorState
from moe_offload.traces.schema import ForecastStep, ForecastWindow, TraceStep

from .base import BaseController


class PregatedStyleController(BaseController):
    """Pregated-style controller with front-loaded prefetching.

    Uses a geometrically decaying weight over the forecast horizon so
    that near-future gating information contributes more strongly to
    the aggregate expert scores, encouraging earlier prefetch.
    """

    def __init__(self, k: int | None = None, decay: float = 0.7) -> None:
        self._k = k
        self._decay = float(decay)

    def _aggregate_scores(self, forecast: ForecastWindow) -> Counter:
        scores: Counter = Counter()
        weight = 1.0
        for step in forecast.steps:
            self._accumulate_step(scores, step, weight)
            weight *= self._decay
        return scores

    @staticmethod
    def _accumulate_step(scores: Counter, step: ForecastStep, weight: float) -> None:
        for eid, p in step.expert_probs.items():
            scores[eid] += float(p) * weight

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

        k = self._k if self._k is not None else int(state.bandwidth_capacity)
        if k <= 0:
            fetch: List[int] = []
        else:
            candidates: Iterable[tuple[int, float]] = (
                (eid, score)
                for eid, score in scores.items()
                if eid not in resident and eid not in pinned and score > 0.0
            )
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

