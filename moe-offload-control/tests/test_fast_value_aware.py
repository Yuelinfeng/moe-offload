"""Tests for the Fast Value-Aware formulation."""

from moe_offload.controllers.fast_value_aware import FastValueAwareController
from moe_offload.simulator.state import SimulatorState
from moe_offload.traces.schema import ForecastStep, ForecastWindow


def _make_state() -> SimulatorState:
    return SimulatorState(
        step_idx=0,
        resident_experts=set(),
        pinned_experts=set(),
        memory_used=0.0,
        memory_capacity=4.0,
        bandwidth_capacity=2.0,
        controller_context={},
    )


def test_fast_uncertainty_suppression() -> None:
    """Demonstrate that high uncertainty strictly defers fetch requests."""
    c = FastValueAwareController(alpha_uncertainty=2.0, alpha_transfer=0.0)
    state = _make_state()
    
    # Scenario A: Low uncertainty
    fw_low = ForecastWindow(
        current_step_idx=0,
        horizon=1,
        steps=[ForecastStep(0, {0: 0.8}, {0: 0.0})],
    )
    d_low = c.plan(state, fw_low)
    
    # Scenario B: High uncertainty leading to negative utility
    fw_high = ForecastWindow(
        current_step_idx=0,
        horizon=1,
        steps=[ForecastStep(0, {0: 0.8}, {0: 0.8})],
    )
    d_high = c.plan(state, fw_high)
    
    # Due to high uncertainty penalty, the item should be excluded in plan B
    assert len(d_low.fetch_experts) > len(d_high.fetch_experts)
    assert 0 in d_low.fetch_experts
    assert 0 not in d_high.fetch_experts
