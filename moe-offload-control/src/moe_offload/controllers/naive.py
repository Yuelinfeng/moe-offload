from __future__ import annotations

from moe_offload.simulator.state import ControllerDecision, SimulatorState
from moe_offload.traces.schema import ForecastWindow, TraceStep

from .base import BaseController


class NaiveController(BaseController):
    """Baseline controller that performs no proactive actions.

    - Ignores any forecast information.
    - Never prefetches or evicts; relies entirely on env's demand fetch.
    - Always returns empty fetch/evict/defer lists, as required by Phase 2.
    """

    def plan(
        self,
        state: SimulatorState,
        forecast: ForecastWindow | None,
        trace_step: TraceStep | None = None,
    ) -> ControllerDecision:
        # Explicitly construct an empty, type-correct decision.
        return ControllerDecision(
            fetch_experts=[],
            evict_experts=[],
            defer_experts=[],
            metadata={},
        )

