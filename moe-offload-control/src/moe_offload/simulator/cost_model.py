"""Cost model — computes per-step metrics from decisions and ground truth.

Implements the cost formula::

    service_cost = stall_latency
                 + alpha_transfer   * transfer_cost
                 + alpha_misprefetch * misprefetch_count
                 + alpha_reload     * reload_count

Extreme-minimal implementation: no OO hierarchy, no plugin system.
"""

from __future__ import annotations

from moe_offload.simulator.bandwidth_model import BandwidthModel
from moe_offload.simulator.state import ControllerDecision, SimulatorState, StepMetrics
from moe_offload.traces.schema import TraceStep


class CostModel:
    """Computes :class:`StepMetrics` for a single simulation step.

    Parameters
    ----------
    alpha_transfer : float
        Weight for transfer cost in the composite ``service_cost``.
    alpha_misprefetch : float
        Weight for misprefetch count in the composite ``service_cost``.
    alpha_reload : float
        Weight for reload count in the composite ``service_cost``.
    misprefetch_window : int
        Number of future steps to look ahead when judging misprefetch.
    reload_window : int
        Number of future steps to look ahead when judging reload.
    bandwidth_model : BandwidthModel
        Bandwidth model used to convert fetch count into transfer cost
        and stall latency.
    """

    def __init__(
        self,
        alpha_transfer: float = 0.1,
        alpha_misprefetch: float = 1.0,
        alpha_reload: float = 1.0,
        misprefetch_window: int = 3,
        reload_window: int = 3,
        bandwidth_model: BandwidthModel | None = None,
    ) -> None:
        self.alpha_transfer = alpha_transfer
        self.alpha_misprefetch = alpha_misprefetch
        self.alpha_reload = alpha_reload
        self.misprefetch_window = misprefetch_window
        self.reload_window = reload_window
        self._bw = bandwidth_model or BandwidthModel(capacity=4)

    def compute(
        self,
        decision: ControllerDecision,
        state: SimulatorState,
        ground_truth: TraceStep,
        future_steps: list[TraceStep],
    ) -> StepMetrics:
        """Compute metrics for one step.

        Parameters
        ----------
        decision : ControllerDecision
            The controller's action for this step.
        state : SimulatorState
            State *before* applying the decision.
        ground_truth : TraceStep
            The actual routing outcome for this step.
        future_steps : list[TraceStep]
            Subsequent trace steps for lookahead evaluation of
            misprefetch and reload.
        """
        active = set(ground_truth.active_experts)

        # --- Determine post-decision resident set ---
        post_residents = set(state.resident_experts)
        post_residents -= set(decision.evict_experts)
        post_residents |= set(decision.fetch_experts)

        # --- Cache misses: active experts not in post-decision resident set ---
        misses = active - post_residents
        cache_miss_count = len(misses)

        # --- Transfer cost and stall ---
        # Stall comes from two sources:
        #   1. fetching the explicitly requested experts (bandwidth overflow)
        #   2. demand-fetching the missed experts on top of that
        total_fetch = len(decision.fetch_experts) + cache_miss_count
        transfer_cost, bw_stall = self._bw.compute_transfer(total_fetch)

        # Each cache miss also incurs a per-miss stall (demand fetch latency)
        miss_stall = float(cache_miss_count)  # 1.0 per miss
        stall_latency = bw_stall + miss_stall

        # --- Misprefetch: fetched but unused in near future ---
        future_active = self._collect_future_active(
            future_steps, self.misprefetch_window
        )
        misprefetch_count = 0
        for eid in decision.fetch_experts:
            if eid not in active and eid not in future_active:
                misprefetch_count += 1

        # --- Reload: evicted but needed in near future ---
        future_active_reload = self._collect_future_active(
            future_steps, self.reload_window
        )
        reload_count = 0
        for eid in decision.evict_experts:
            if eid in future_active_reload:
                reload_count += 1

        # --- Composite service cost ---
        service_cost = (
            stall_latency
            + self.alpha_transfer * transfer_cost
            + self.alpha_misprefetch * misprefetch_count
            + self.alpha_reload * reload_count
        )

        return StepMetrics(
            stall_latency=stall_latency,
            transfer_cost=transfer_cost,
            misprefetch_count=misprefetch_count,
            reload_count=reload_count,
            cache_miss_count=cache_miss_count,
            service_cost=service_cost,
        )

    @staticmethod
    def _collect_future_active(
        future_steps: list[TraceStep], window: int
    ) -> set[int]:
        """Collect all active expert IDs in the next *window* steps."""
        result: set[int] = set()
        for step in future_steps[:window]:
            result.update(step.active_experts)
        return result
