import pytest

from moe_offload.controllers import (
    DiffMoeHeuristicController,
    NaiveController,
    PregatedStyleController,
    TopKPrefetchController,
)
from moe_offload.predictors import (
    BadPredictor,
    MediumNoisePredictor,
    OracleLitePredictor,
)
from moe_offload.simulator.env import SimulatorEnv
from moe_offload.simulator.state import ControllerDecision, SimulatorState, StepMetrics
from moe_offload.traces.schema import ForecastWindow, TraceEpisode, TraceStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode() -> TraceEpisode:
    steps = [
        TraceStep(step_idx=0, layer_id=0, active_experts=[0, 1]),
        TraceStep(step_idx=1, layer_id=0, active_experts=[1, 2]),
        TraceStep(step_idx=2, layer_id=0, active_experts=[0]),
        TraceStep(step_idx=3, layer_id=0, active_experts=[2]),
    ]
    return TraceEpisode(steps=steps, num_experts=4, num_layers=1, metadata={})


def _make_env_config() -> dict:
    return {
        "memory_capacity": 2,
        "bandwidth_capacity": 2,
        "per_expert_latency": 1.0,
        "alpha_transfer": 0.1,
        "alpha_misprefetch": 1.0,
        "alpha_reload": 1.0,
        "misprefetch_window": 3,
        "reload_window": 3,
        "pinned_experts": [0],
    }


class _DummyController:
    """Small helper used to assert typing with SupportsPredict, etc."""

    def plan(self, state: SimulatorState, forecast: ForecastWindow | None, trace_step=None) -> ControllerDecision:  # type: ignore[override]
        return ControllerDecision(
            fetch_experts=[],
            evict_experts=[],
            defer_experts=[],
            metadata={},
        )


# ---------------------------------------------------------------------------
# Predictor smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "predictor_cls",
    [OracleLitePredictor, MediumNoisePredictor, BadPredictor],
)
def test_predictors_return_valid_forecast_window(predictor_cls):
    episode = _make_episode()
    predictor = predictor_cls()

    window = predictor.predict(episode, current_index=0, horizon=3)
    assert isinstance(window, ForecastWindow)
    assert window.current_step_idx == 0
    assert 0 <= len(window.steps) <= 3

    for step in window.steps:
        assert isinstance(step.layer_id, int)
        assert isinstance(step.expert_probs, dict)
        assert isinstance(step.uncertainty, dict)
        # Keys and values types
        for k, v in step.expert_probs.items():
            assert isinstance(k, int)
            assert isinstance(v, float)
        for k, v in step.uncertainty.items():
            assert isinstance(k, int)
            assert isinstance(v, float)
        # Probabilities should roughly form a distribution.
        total_prob = sum(step.expert_probs.values())
        assert total_prob == pytest.approx(1.0, rel=1e-2)


def test_predictors_handle_tail_and_zero_horizon():
    episode = _make_episode()
    predictor = OracleLitePredictor()

    # Near the end of the episode
    window_tail = predictor.predict(episode, current_index=3, horizon=5)
    assert len(window_tail.steps) == 0

    # Zero horizon
    window_zero = predictor.predict(episode, current_index=0, horizon=0)
    assert len(window_zero.steps) == 0


# ---------------------------------------------------------------------------
# Controller legality tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "controller",
    [
        NaiveController(),
        TopKPrefetchController(k=2),
        PregatedStyleController(k=2),
        DiffMoeHeuristicController(global_hot_experts=[0, 1]),
    ],
)
def test_controllers_interface_and_pinned_safety(controller):
    episode = _make_episode()
    state = SimulatorState(
        step_idx=0,
        resident_experts={0},
        pinned_experts={0},
        memory_used=1,
        memory_capacity=2,
        bandwidth_capacity=2,
        controller_context={},
    )

    # Build a small forecast window from OracleLite for convenience.
    forecast = OracleLitePredictor().predict(episode, current_index=0, horizon=2)

    decision = controller.plan(state, forecast, episode.steps[0])
    assert isinstance(decision, ControllerDecision)
    assert isinstance(decision.fetch_experts, list)
    assert isinstance(decision.evict_experts, list)
    assert isinstance(decision.defer_experts, list)

    # None of the controllers are allowed to explicitly evict pinned experts.
    assert not (set(decision.evict_experts) & state.pinned_experts)


def test_naive_controller_returns_empty_actions():
    episode = _make_episode()
    state = SimulatorState(
        step_idx=0,
        resident_experts=set(),
        pinned_experts=set(),
        memory_used=0,
        memory_capacity=2,
        bandwidth_capacity=2,
        controller_context={},
    )
    controller = NaiveController()
    forecast = OracleLitePredictor().predict(episode, current_index=0, horizon=2)

    decision = controller.plan(state, forecast, episode.steps[0])
    assert decision.fetch_experts == []
    assert decision.evict_experts == []
    assert decision.defer_experts == []


# ---------------------------------------------------------------------------
# Env integration smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "controller",
    [
        NaiveController(),
        TopKPrefetchController(k=2),
    ],
)
def test_env_integration_with_baseline_controllers(controller):
    episode = _make_episode()
    env = SimulatorEnv(_make_env_config())
    state = env.reset(episode)

    predictor = OracleLitePredictor()

    step_metrics: list[StepMetrics] = []

    while not env.is_done():
        idx = env.current_step_idx
        forecast = predictor.predict(episode, current_index=idx, horizon=2)
        trace_step = episode.steps[idx]

        decision = controller.plan(state, forecast, trace_step)
        state, metrics = env.step(decision)
        step_metrics.append(metrics)

        # If controller supports observe, call it.
        if hasattr(controller, "observe"):
            controller.observe(
                prev_state=state,
                decision=decision,
                trace_step=trace_step,
                metrics=metrics,
                next_state=state,
            )

    assert len(step_metrics) == len(episode.steps)
    for m in step_metrics:
        assert isinstance(m, StepMetrics)
        assert m.stall_latency >= 0
        assert m.transfer_cost >= 0
        assert m.service_cost >= 0

