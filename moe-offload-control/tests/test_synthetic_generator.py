"""Tests for synthetic trace generator."""

from moe_offload.traces.synthetic_generator import generate_synthetic_trace


def test_synthetic_validity() -> None:
    """Test structural correctness of generated traces."""
    ep = generate_synthetic_trace(num_steps=10)
    assert len(ep.steps) == 10
    
    # Check monotonicity
    assert ep.steps[0].step_idx == 0
    assert ep.steps[1].step_idx == 1


def test_synthetic_reproducibility() -> None:
    """Test that traces identically respect the random seed."""
    ep1 = generate_synthetic_trace(seed=8, num_steps=5, regime="burst")
    ep2 = generate_synthetic_trace(seed=8, num_steps=5, regime="burst")
    
    assert ep1.steps[0].active_experts == ep2.steps[0].active_experts
    assert ep1.steps[4].active_experts == ep2.steps[4].active_experts
