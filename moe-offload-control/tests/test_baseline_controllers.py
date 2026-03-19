"""Tests for baseline controllers."""

from moe_offload.controllers import (
    DiffMoEHeuristicController,
    NaiveController,
    PregatedStyleController,
    TopKPrefetchController,
)
from moe_offload.simulator.state import ControllerDecision, SimulatorState
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


def test_naive_controller() -> None:
    """Naive controller should strictly return empty decisions."""
    c = NaiveController()
    state = _make_state()
    decision = c.plan(state, None)
    
    assert isinstance(decision, ControllerDecision)
    assert len(decision.fetch_experts) == 0
    assert len(decision.evict_experts) == 0


def test_topk_controller() -> None:
    """TopK controller should fetch highest probability experts."""
    c = TopKPrefetchController(k=1)
    state = _make_state()
    
    fw = ForecastWindow(
        current_step_idx=0,
        horizon=1,
        steps=[ForecastStep(0, {0: 0.9, 1: 0.1, 2: 0.5}, {})],
    )
    decision = c.plan(state, fw)
    
    assert decision.fetch_experts == [0]


def test_pregated_controller() -> None:
    """Pregated controller aggregates probs and checks threshold."""
    c = PregatedStyleController(lookahead_steps=2, threshold=0.7)
    state = _make_state()
    
    fw = ForecastWindow(
        current_step_idx=0,
        horizon=2,
        steps=[
            ForecastStep(0, {0: 0.6, 1: 0.1}, {}),
            ForecastStep(0, {0: 0.8, 1: 0.4}, {}),  # Expert 0 hits 0.8 > 0.7
        ],
    )
    decision = c.plan(state, fw)
    
    assert decision.fetch_experts == [0]


def test_diffmoe_controller() -> None:
    """DiffMoE validates structural fetch candidates based on localized heuristics."""
    c = DiffMoEHeuristicController(prefetch_k=1, threshold_hot=0.5)
    state = _make_state()
    
    fw = ForecastWindow(
        current_step_idx=0,
        horizon=1,
        steps=[ForecastStep(0, {3: 0.9}, {})],
    )
    decision = c.plan(state, fw)
    
    # 0.9 prob immediately breaches forecast check or priority bounds depending on tuning
    assert isinstance(decision, ControllerDecision)
    assert 3 in decision.fetch_experts
