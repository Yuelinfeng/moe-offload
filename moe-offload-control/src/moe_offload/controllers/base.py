"""Base interface for all controllers."""

import abc

from moe_offload.simulator.state import ControllerDecision, SimulatorState, StepMetrics
from moe_offload.traces.schema import ForecastWindow


class BaseController(abc.ABC):
    """Abstract base class for all control policies.
    
    Controllers strictly make fetch, evict, and defer decisions based on 
    the provided environment state and an optional forecast window.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """String identifier for the controller."""
        pass

    def reset(self, config: dict | None = None) -> None:
        """Reset internal states before a new simulation episode."""
        pass

    @abc.abstractmethod
    def plan(
        self, state: SimulatorState, forecast: ForecastWindow | None = None
    ) -> ControllerDecision:
        """Make a control decision given the simulator state and forecast.
        
        Parameters
        ----------
        state : SimulatorState
            The current simulated state (resident experts, pinned experts, capacity, etc).
        forecast : ForecastWindow | None
            A forecast of future steps provided by a predictor.
            
        Returns
        -------
        ControllerDecision
            A valid decision struct prescribing fetches, evictions, and deferrals.
        """
        pass

    def observe(self, state: SimulatorState, metrics: StepMetrics) -> None:
        """Observe the outcome to update internal models or statistics.
        
        This is called after `env.step()` using the new environment state and metrics.
        """
        pass
