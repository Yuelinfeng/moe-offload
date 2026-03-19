"""Run problem existence evaluation."""

from moe_offload.experiments.registry import build_controller, build_predictor
from moe_offload.experiments.run_experiment import run_experiment
from moe_offload.traces.synthetic_generator import generate_synthetic_trace


def main() -> None:
    """Run baseline controllers against FastValueAware to demonstrate problem existence."""
    episode = generate_synthetic_trace(
        num_steps=100, num_experts=16, active_per_step=2, regime="drift", seed=42
    )
    
    # Highly constrained environment
    env_config = {
        "memory_capacity": 6,  # Can only hold 6 experts at a time
        "bandwidth_capacity": 2, # Soft cap
        "alpha_transfer": 1.0,
        "alpha_misprefetch": 1.0,
        "alpha_reload": 1.0,
        "misprefetch_window": 3,
        "reload_window": 3,
    }
    
    # Predictor with medium errors
    predictor = build_predictor("noisy", preset="medium")
    
    controllers = ["naive", "topk", "pregated", "diffmoe", "fast_value"]
    
    print(f"{'Controller':<15} | {'Total Cost':<10} | {'Misses':<8} | {'Misprefetch':<12}")
    print("-" * 55)
    
    for c_name in controllers:
        c = build_controller(
            c_name, 
            # Controller specific args if needed, registry handles kwargs dynamically
        )
        res = run_experiment(
            episode, env_config, c, predictor, horizon=3, config_id="p_exist"
        )
        metrics = res.aggregated_metrics
        print(
            f"[{c_name:<13}] | {metrics['total_service_cost']:<10.2f} "
            f"| {metrics['total_cache_miss_count']:<8.0f} | {metrics['total_misprefetch_count']:<12.0f}"
        )


if __name__ == "__main__":
    main()
