"""Tests for trace schema dataclasses and validation."""

import pytest

from moe_offload.traces.schema import (
    TraceEpisode,
    TraceStep,
    validate_episode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    step_idx: int,
    layer_id: int = 0,
    experts: list[int] | None = None,
) -> TraceStep:
    """Shorthand for creating a TraceStep."""
    return TraceStep(
        step_idx=step_idx,
        layer_id=layer_id,
        active_experts=experts if experts is not None else [0],
        regime_id=None,
        metadata={},
    )


def _make_episode(
    steps: list[TraceStep],
    num_experts: int = 4,
    num_layers: int = 1,
) -> TraceEpisode:
    """Shorthand for creating a TraceEpisode for tests."""
    return TraceEpisode(
        steps=steps,
        num_experts=num_experts,
        num_layers=num_layers,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Valid episodes
# ---------------------------------------------------------------------------


class TestValidEpisode:
    def test_single_step(self):
        ep = _make_episode([_make_step(0)])
        validate_episode(ep)  # should not raise

    def test_multi_step(self):
        ep = _make_episode([
            _make_step(0, experts=[0, 1]),
            _make_step(1, experts=[2, 3]),
            _make_step(2, experts=[1]),
        ])
        validate_episode(ep)

    def test_non_contiguous_step_idx(self):
        """step_idx just needs to be monotonically increasing, not contiguous."""
        ep = _make_episode([
            _make_step(0),
            _make_step(5),
            _make_step(100),
        ])
        validate_episode(ep)


# ---------------------------------------------------------------------------
# Invalid episodes
# ---------------------------------------------------------------------------


class TestInvalidEpisode:
    def test_empty_episode(self):
        ep = _make_episode([])
        with pytest.raises(ValueError, match="at least 1 step"):
            validate_episode(ep)

    def test_num_experts_must_be_positive(self):
        ep = _make_episode([_make_step(0)], num_experts=0)
        with pytest.raises(ValueError, match="num_experts must be positive"):
            validate_episode(ep)

    def test_num_layers_must_be_positive(self):
        ep = _make_episode([_make_step(0)], num_layers=0)
        with pytest.raises(ValueError, match="num_layers must be positive"):
            validate_episode(ep)

    def test_non_monotonic_step_idx(self):
        ep = _make_episode([
            _make_step(0),
            _make_step(2),
            _make_step(1),  # goes backward
        ])
        with pytest.raises(ValueError, match="monotonically increasing"):
            validate_episode(ep)

    def test_duplicate_step_idx(self):
        ep = _make_episode([
            _make_step(0),
            _make_step(0),  # duplicate
        ])
        with pytest.raises(ValueError, match="monotonically increasing"):
            validate_episode(ep)

    def test_expert_id_out_of_range(self):
        ep = _make_episode(
            [_make_step(0, experts=[4])],  # num_experts=4, so max valid=3
            num_experts=4,
        )
        with pytest.raises(ValueError, match="expert_id=4 out of range"):
            validate_episode(ep)

    def test_negative_expert_id(self):
        ep = _make_episode([_make_step(0, experts=[-1])])
        with pytest.raises(ValueError, match="expert_id=-1 out of range"):
            validate_episode(ep)

    def test_layer_id_out_of_range(self):
        ep = _make_episode(
            [_make_step(0, layer_id=1)],  # num_layers=1, so max valid=0
            num_layers=1,
        )
        with pytest.raises(ValueError, match="layer_id=1 out of range"):
            validate_episode(ep)