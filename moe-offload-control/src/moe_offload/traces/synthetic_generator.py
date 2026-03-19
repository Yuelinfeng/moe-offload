"""Minimal synthetic trace generator."""

import random

from moe_offload.traces.schema import TraceEpisode, TraceStep


def generate_synthetic_trace(
    num_steps: int = 100,
    num_experts: int = 8,
    num_layers: int = 1,
    active_per_step: int = 2,
    regime: str = "locality",
    seed: int = 42,
) -> TraceEpisode:
    """Generate simple traces showcasing burst, drift, or temporal locality.
    
    Parameters
    ----------
    regime : str
        One of 'locality', 'burst', or 'drift'.
        
    Returns
    -------
    TraceEpisode
        A valid trace episode populated with synthetic interactions.
    """
    random.seed(seed)
    steps = []

    # Simple base state for drift
    hot_set = list(range(active_per_step + 1))

    for i in range(num_steps):
        if regime == "drift" and i > 0 and i % 20 == 0:
            # Shift the hot set periodically
            hot_set = [(e + 1) % num_experts for e in hot_set]

        if regime == "burst" and i % 10 == 0:
            # Occasional spikes in active expert count
            k = min(num_experts, active_per_step + 2)
            current_active = random.sample(range(num_experts), k)
        elif regime in ("locality", "drift"):
            # High temporal locality drawn from a small hot set
            current_active = random.sample(hot_set, active_per_step)
        else:
            # Pure uniform random
            current_active = random.sample(range(num_experts), active_per_step)

        steps.append(
            TraceStep(
                step_idx=i,
                layer_id=0,
                active_experts=current_active,
                regime_id=regime,
                metadata={},
            )
        )

    return TraceEpisode(
        steps=steps,
        num_experts=num_experts,
        num_layers=num_layers,
        metadata={"regime": regime, "seed": seed},
    )
