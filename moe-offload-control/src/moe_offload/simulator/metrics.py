"""Helper for metrics aggregation."""

from moe_offload.simulator.state import StepMetrics


def aggregate_metrics(metrics_list: list[StepMetrics]) -> dict[str, float]:
    """Aggregate a list of per-step metrics into a summary dictionary."""
    if not metrics_list:
        return {}

    total_service = sum(m.service_cost for m in metrics_list)
    total_transfer = sum(m.transfer_cost for m in metrics_list)
    total_misprefetch = sum(m.misprefetch_count for m in metrics_list)
    total_reload = sum(m.reload_count for m in metrics_list)
    total_miss = sum(m.cache_miss_count for m in metrics_list)
    mean_stall = sum(m.stall_latency for m in metrics_list) / len(metrics_list)

    return {
        "mean_stall_latency": mean_stall,
        "total_service_cost": total_service,
        "total_transfer_cost": total_transfer,
        "total_misprefetch_count": total_misprefetch,
        "total_reload_count": total_reload,
        "total_cache_miss_count": total_miss,
    }
