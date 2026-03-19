"""Pregated-style semantic controller."""

from moe_offload.controllers.base import BaseController
from moe_offload.simulator.state import ControllerDecision, SimulatorState
from moe_offload.traces.schema import ForecastWindow


class PregatedStyleController(BaseController):
    """Aggregates short-horizon predictions conceptually representing early gating.
    
    Not a one-to-one replica of a Pregated hardware patch, but simulates the 
    semantic value of looking ahead 1 or 2 steps using deterministic aggregation
    and prefetching components that exceed a rigid gating threshold.
    """

    def __init__(self, lookahead_steps: int = 2, threshold: float = 0.5) -> None:
        self.lookahead = lookahead_steps
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"pregated_style_h{self.lookahead}"

    def plan(
        self, state: SimulatorState, forecast: ForecastWindow | None = None
    ) -> ControllerDecision:
        if not forecast or not forecast.steps:
            return ControllerDecision([], [], [])

        agg_probs: dict[int, float] = {}
        for step in forecast.steps[: self.lookahead]:
            for eid, prob in step.expert_probs.items():
                agg_probs[eid] = max(agg_probs.get(eid, 0.0), prob)

        to_fetch = [
            eid
            for eid, prob in agg_probs.items()
            if prob >= self.threshold and eid not in state.resident_experts
        ]

        return ControllerDecision(
            fetch_experts=to_fetch,
            evict_experts=[],
            defer_experts=[],
            metadata={"agg_probs": agg_probs},
        )
