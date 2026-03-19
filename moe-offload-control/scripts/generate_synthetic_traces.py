"""Generate synthetic traces for testing."""

import os

from moe_offload.traces.synthetic_generator import generate_synthetic_trace


def main() -> None:
    """Entry point for trace generation."""
    os.makedirs("data/traces/synthetic", exist_ok=True)
    
    # Generate a sample trace with drift
    trace = generate_synthetic_trace(
        num_steps=100,
        num_experts=16,
        active_per_step=3,
        regime="drift",
        seed=42,
    )
    
    print(
        f"Generated trace with {len(trace.steps)} steps, "
        f"{trace.num_experts} experts, regime '{trace.metadata.get('regime')}'."
    )


if __name__ == "__main__":
    main()
