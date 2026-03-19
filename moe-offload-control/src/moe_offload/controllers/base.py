from __future__ import annotations

from abc import ABC, abstractmethod

from moe_offload.simulator.state import ControllerDecision, SimulatorState, StepMetrics
from moe_offload.traces.schema import ForecastWindow, TraceStep


class BaseController(ABC):
    """Abstract interface for fast-timescale offloading controllers.

    Phase 2 keeps this deliberately small: controllers receive only the
    current simulator state, an optional forecast window, and (optionally)
    the current ground-truth trace step.
    """

    def reset(self) -> None:
        """Reset any internal controller state between episodes."""
        # Default implementation is a no-op for stateless controllers.
        return None

    @abstractmethod
    def plan(
        self,
        state: SimulatorState,
        forecast: ForecastWindow | None,
        trace_step: TraceStep | None = None,
    ) -> ControllerDecision:
        """Compute a one-step offloading decision.

        Implementations must never evict ``state.pinned_experts``; tests
        will validate this property even though the environment also
        enforces it defensively.
        """
        raise NotImplementedError

    def observe(
        self,
        prev_state: SimulatorState,
        decision: ControllerDecision,
        trace_step: TraceStep,
        metrics: StepMetrics,
        next_state: SimulatorState,
    ) -> None:
        """Optional feedback hook called after :meth:`plan`.

        Baseline controllers in Phase 2 are free to ignore this, but
        the method exists to make online adaptation and two-timescale
        designs easier to add later.
        """
        return None

