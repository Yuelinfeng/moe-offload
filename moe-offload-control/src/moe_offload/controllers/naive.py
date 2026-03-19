"""Naive controller that does zero prefetching."""

from moe_offload.controllers.base import BaseController
from moe_offload.simulator.state import ControllerDecision, SimulatorState
from moe_offload.traces.schema import ForecastWindow


class NaiveController(BaseController):
    """Simplest baseline: no prefetch, rely entirely on environment demand fetch.
    
    This acts as the weakest lower bound. It assumes the simulator's internal
    capacity constraints and default demand fetch/evict strategies will resolve 
    misses gracefully (though expensively).
    """

    @property
    def name(self) -> str:
        return "naive"

    def plan(
        self, state: SimulatorState, forecast: ForecastWindow | None = None
    ) -> ControllerDecision:
        """Fetch nothing and evict nothing proactively.
        
        The environment will trigger demand-fetch for active disjoints automatically
        and blindly discard the lowest resident IDs if capacity limits are breached.
        """
        return ControllerDecision(
            fetch_experts=[],
            evict_experts=[],
            defer_experts=[],
            metadata={},
        )
