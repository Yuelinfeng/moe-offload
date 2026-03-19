"""Fast value-aware uncertainty-aware controller.

This is the fast-timescale semantic controller for the paper's main method.

It does NOT implement full MPC / ADP. Instead, it implements a minimal,
interpretable, value-based online policy:

1. Aggregate short-horizon forecast information into a future value proxy.
2. Penalize uncertainty.
3. Penalize transfer for new fetches.
4. Compare candidate fetch value against resident keep value.
5. Perform only beneficial replacements.
6. Defer positive-value candidates that are not worth immediate replacement.

This is intentionally simple but aligned with the decision-theoretic story.
"""

from __future__ import annotations

from moe_offload.controllers.base import BaseController
from moe_offload.simulator.state import ControllerDecision, SimulatorState
from moe_offload.traces.schema import ForecastWindow


class FastValueAwareController(BaseController):
    """Fast-timescale value-aware uncertainty-aware controller.

    Parameters
    ----------
    alpha_transfer : float
        Transfer penalty for fetching a non-resident expert.
    alpha_uncertainty : float
        Penalty applied to uncertainty in future value estimation.
    alpha_eviction : float
        Optional margin used when deciding whether replacement is worth it.
        A replacement is made only if:

            fetch_value > keep_value + alpha_eviction

    horizon_discount : float
        Discount factor for farther forecast steps. Must be in (0, 1].
        If 1.0, all steps are weighted equally.
    """

    def __init__(
        self,
        alpha_transfer: float = 1.0,
        alpha_uncertainty: float = 0.5,
        alpha_eviction: float = 0.0,
        horizon_discount: float = 1.0,
    ) -> None:
        self.alpha_transfer = alpha_transfer
        self.alpha_uncertainty = alpha_uncertainty
        self.alpha_eviction = alpha_eviction
        self.horizon_discount = horizon_discount

    @property
    def name(self) -> str:
        return "fast_value_aware"

    def reset(self, config: dict | None = None) -> None:
        # Stateless first implementation
        return None

    def plan(
        self, state: SimulatorState, forecast: ForecastWindow | None = None
    ) -> ControllerDecision:
        if not forecast or not forecast.steps:
            return ControllerDecision(
                fetch_experts=[],
                evict_experts=[],
                defer_experts=[],
                metadata={"scores": {}},
            )

        # ------------------------------------------------------------
        # 1) Estimate future value for each expert
        # ------------------------------------------------------------
        # future_value[e] = discounted sum over forecast window of:
        #                   prob - alpha_uncertainty * uncertainty
        future_value: dict[int, float] = {}

        for t, step in enumerate(forecast.steps):
            weight = self.horizon_discount ** t
            for eid, prob in step.expert_probs.items():
                uncert = step.uncertainty.get(eid, 0.0)
                value_increment = weight * (prob - self.alpha_uncertainty * uncert)
                future_value[eid] = future_value.get(eid, 0.0) + value_increment

        # ------------------------------------------------------------
        # 2) Split into fetch values and keep values
        # ------------------------------------------------------------
        current_residents = set(state.resident_experts)
        pinned = set(state.pinned_experts)

        # For non-resident experts, fetching costs transfer
        fetch_value: dict[int, float] = {}
        for eid, val in future_value.items():
            if eid not in current_residents:
                fetch_value[eid] = val - self.alpha_transfer

        # For resident experts, keep_value is just their future value proxy
        keep_value: dict[int, float] = {}
        for eid in current_residents:
            keep_value[eid] = future_value.get(eid, 0.0)

        # ------------------------------------------------------------
        # 3) Directly fetch positive-value experts if there is free room
        # ------------------------------------------------------------
        free_slots = int(state.memory_capacity - state.memory_used)

        candidate_fetches = [
            eid for eid, val in fetch_value.items()
            if val > 0.0
        ]
        candidate_fetches.sort(key=lambda eid: fetch_value[eid], reverse=True)

        to_fetch: list[int] = []
        to_evict: list[int] = []
        deferred: list[int] = []

        # Fill existing free slots first
        direct_fetches = candidate_fetches[:free_slots]
        to_fetch.extend(direct_fetches)

        remaining_candidates = candidate_fetches[free_slots:]

        # ------------------------------------------------------------
        # 4) Replacement logic: fetch candidate vs lowest-value resident
        # ------------------------------------------------------------
        # Only non-pinned residents are eligible victims
        evictable_residents = [
            eid for eid in current_residents if eid not in pinned
        ]
        evictable_residents.sort(key=lambda eid: keep_value.get(eid, 0.0))

        victim_idx = 0

        for eid in remaining_candidates:
            if victim_idx >= len(evictable_residents):
                deferred.append(eid)
                continue

            victim = evictable_residents[victim_idx]
            candidate_val = fetch_value[eid]
            victim_keep_val = keep_value.get(victim, 0.0)

            # Replacement is beneficial only if the new candidate
            # clearly dominates the victim by at least alpha_eviction
            if candidate_val > victim_keep_val + self.alpha_eviction:
                to_fetch.append(eid)
                to_evict.append(victim)
                victim_idx += 1
            else:
                deferred.append(eid)

        # ------------------------------------------------------------
        # 5) Also defer any non-positive fetch values
        # ------------------------------------------------------------
        for eid, val in fetch_value.items():
            if val <= 0.0:
                deferred.append(eid)

        # Deduplicate defer list while preserving stable order
        defer_unique: list[int] = []
        seen: set[int] = set()
        for eid in deferred:
            if eid not in seen and eid not in to_fetch:
                seen.add(eid)
                defer_unique.append(eid)

        return ControllerDecision(
            fetch_experts=to_fetch,
            evict_experts=to_evict,
            defer_experts=defer_unique,
            metadata={
                "future_value": dict(future_value),
                "fetch_value": dict(fetch_value),
                "keep_value": dict(keep_value),
            },
        )