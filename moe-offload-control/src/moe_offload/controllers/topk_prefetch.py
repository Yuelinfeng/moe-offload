"""Top-K prefetch controller."""

from moe_offload.controllers.base import BaseController
from moe_offload.simulator.state import ControllerDecision, SimulatorState
from moe_offload.traces.schema import ForecastWindow


class TopKPrefetchController(BaseController):
    """Fetches the top K experts from the immediate future step prediction.
    
    A pure prediction-triggered baseline without value-aware valuation.
    Does not respect uncertainty or opportunity costs.
    """

    def __init__(self, k: int = 2) -> None:
        self.k = k

    @property
    def name(self) -> str:
        return f"topk_prefetch_k{self.k}"

    def plan(
        self, state: SimulatorState, forecast: ForecastWindow | None = None
    ) -> ControllerDecision:
        if not forecast or not forecast.steps:
            return ControllerDecision([], [], [])

        # Look only at the immediate next step (horizon=0 locally in window)
        next_step = forecast.steps[0]
        
        # Sort experts descending primarily by predicted probability
        sorted_experts = sorted(
            next_step.expert_probs.keys(),
            key=lambda e: next_step.expert_probs[e],
            reverse=True,
        )

        topk = sorted_experts[: self.k]

        # Filter strictly those not already resident
        to_fetch = [eid for eid in topk if eid not in state.resident_experts]

        return ControllerDecision(
            fetch_experts=to_fetch,
            evict_experts=[],  # Let env handle overflow eviction
            defer_experts=[],
            metadata={"topk": topk},
        )
