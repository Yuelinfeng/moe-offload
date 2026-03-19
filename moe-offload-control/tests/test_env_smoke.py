"""Smoke tests for the SimulatorEnv — minimal end-to-end loop."""

import pytest

from moe_offload.simulator.env import SimulatorEnv
from moe_offload.simulator.state import ControllerDecision
from moe_offload.traces.schema import TraceEpisode, TraceStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 5-step episode, 1 layer, 3 experts total, varying active sets
_STEPS = [
    TraceStep(step_idx=0, layer_id=0, active_experts=[0, 1], regime_id=None, metadata={}),
    TraceStep(step_idx=1, layer_id=0, active_experts=[1, 2], regime_id=None, metadata={}),
    TraceStep(step_idx=2, layer_id=0, active_experts=[0],    regime_id=None, metadata={}),
    TraceStep(step_idx=3, layer_id=0, active_experts=[2],    regime_id=None, metadata={}),
    TraceStep(step_idx=4, layer_id=0, active_experts=[0, 1], regime_id=None, metadata={}),
]

_EPISODE = TraceEpisode(
    steps=_STEPS,
    num_experts=3,
    num_layers=1,
    metadata={"source": "test"},
)

_CONFIG = {
    "memory_capacity": 2,
    "bandwidth_capacity": 2,
    "per_expert_latency": 1.0,
    "alpha_transfer": 0.1,
    "alpha_misprefetch": 1.0,
    "alpha_reload": 1.0,
    "misprefetch_window": 3,
    "reload_window": 3,
}

_NOOP_DECISION = ControllerDecision(
    fetch_experts=[], evict_experts=[], defer_experts=[], metadata={},
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnvReset:
    def test_reset_returns_initial_state(self):
        env = SimulatorEnv(_CONFIG)
        state = env.reset(_EPISODE)
        assert state.step_idx == 0
        assert state.memory_capacity == 2
        assert state.bandwidth_capacity == 2
        assert state.resident_experts == set()
        assert not env.is_done()

    def test_step_before_reset_raises(self):
        env = SimulatorEnv(_CONFIG)
        with pytest.raises(RuntimeError, match="reset"):
            env.step(_NOOP_DECISION)


class TestEnvNoopController:
    """With a no-op controller, every step should produce cache misses."""

    def test_noop_produces_misses(self):
        env = SimulatorEnv(_CONFIG)
        env.reset(_EPISODE)

        all_metrics = []
        while not env.is_done():
            state, metrics = env.step(_NOOP_DECISION)
            all_metrics.append(metrics)

        assert len(all_metrics) == 5

        # First step: active=[0,1], resident=empty → 2 misses
        assert all_metrics[0].cache_miss_count == 2
        assert all_metrics[0].stall_latency > 0

        # Every step should have non-negative metrics
        for m in all_metrics:
            assert m.stall_latency >= 0
            assert m.transfer_cost >= 0
            assert m.service_cost >= 0
            assert m.cache_miss_count >= 0
            assert m.misprefetch_count >= 0
            assert m.reload_count >= 0

    def test_step_after_done_raises(self):
        env = SimulatorEnv(_CONFIG)
        env.reset(_EPISODE)
        while not env.is_done():
            env.step(_NOOP_DECISION)
        with pytest.raises(RuntimeError, match="done"):
            env.step(_NOOP_DECISION)


class TestEnvPerfectController:
    """With a perfect controller, active experts are always pre-loaded."""

    def test_perfect_no_stall(self):
        env = SimulatorEnv(_CONFIG)
        state = env.reset(_EPISODE)

        for i, trace_step in enumerate(_STEPS):
            needed = set(trace_step.active_experts)
            resident = state.resident_experts

            # Evict experts not needed (to make room)
            to_evict = list(resident - needed)
            # Fetch experts not yet resident
            to_fetch = list(needed - resident)

            decision = ControllerDecision(
                fetch_experts=to_fetch,
                evict_experts=to_evict,
                defer_experts=[],
                metadata={},
            )
            state, metrics = env.step(decision)

            # With a perfect controller, no cache misses: all needed
            # experts were fetched in the decision
            assert metrics.cache_miss_count == 0, (
                f"step {i}: expected 0 misses, got {metrics.cache_miss_count}"
            )
            assert metrics.stall_latency == 0.0, (
                f"step {i}: expected 0 stall, got {metrics.stall_latency}"
            )

        assert env.is_done()


class TestEnvStateProgression:
    """Verify that state transitions are consistent."""

    def test_step_idx_increments(self):
        env = SimulatorEnv(_CONFIG)
        env.reset(_EPISODE)

        for expected_idx in range(1, 6):
            state, _ = env.step(_NOOP_DECISION)
            assert state.step_idx == expected_idx

    def test_memory_usage_consistent(self):
        env = SimulatorEnv(_CONFIG)
        env.reset(_EPISODE)

        while not env.is_done():
            state, _ = env.step(_NOOP_DECISION)
            assert state.memory_used == len(state.resident_experts)
            assert state.memory_used <= state.memory_capacity
