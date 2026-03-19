"""Diff-MoE heuristic semantic controller.

This controller is a simulator-level semantic approximation of the
Diff-MoE style heuristic. It does NOT reproduce FT/HPC/MPC/LPC buffer
mechanics. Instead, it preserves the intended decision logic:

- global hot experts act like strong keepers (pinned-like bias)
- local priority scores track short-horizon reuse tendency
- next-step / short-horizon forecast drives prefetch candidates
- if space is needed, evict low-priority, non-global, non-pinned residents
"""

from __future__ import annotations

from moe_offload.controllers.base import BaseController
from moe_offload.simulator.state import ControllerDecision, SimulatorState
from moe_offload.traces.schema import ForecastWindow


class DiffMoEHeuristicController(BaseController):
    """Semantic Diff-MoE-style heuristic controller.

    Parameters
    ----------
    global_hot : list[int] | None
        Experts treated as globally hot and given a strong retention bias.
    prefetch_k : int
        Maximum number of uncached experts to prefetch.
    threshold_hot : float
        Priority threshold for admitting a local candidate.
    delta_inc : float
        Increment added to priority when short-horizon forecast suggests hotness.
    delta_in_dec : float
        Small decay when an expert remains relevant but not strongly hot.
    delta_out_dec : float
        Larger decay when an expert is not forecast as useful.
    lookahead_steps : int
        Number of forecast steps to aggregate when updating local priority.
    prefetch_prob_threshold : float
        Direct probability-based admission fallback for clearly hot candidates.
    """

    def __init__(
        self,
        global_hot: list[int] | None = None,
        prefetch_k: int = 2,
        threshold_hot: float = 0.6,
        delta_inc: float = 0.2,
        delta_in_dec: float = 0.05,
        delta_out_dec: float = 0.10,
        lookahead_steps: int = 2,
        prefetch_prob_threshold: float = 0.8,
    ) -> None:
        self.global_hot = set(global_hot or [])
        self.prefetch_k = prefetch_k
        self.threshold_hot = threshold_hot
        self.delta_inc = delta_inc
        self.delta_in_dec = delta_in_dec
        self.delta_out_dec = delta_out_dec
        self.lookahead_steps = lookahead_steps
        self.prefetch_prob_threshold = prefetch_prob_threshold

        # Local heuristic memory
        self.priority_scores: dict[int, float] = {}

    @property
    def name(self) -> str:
        return "diffmoe_heuristic"

    def reset(self, config: dict | None = None) -> None:
        self.priority_scores.clear()
        if config and "global_hot_experts" in config:
            self.global_hot = set(config["global_hot_experts"])

    def plan(
        self, state: SimulatorState, forecast: ForecastWindow | None = None
    ) -> ControllerDecision:
        if not forecast or not forecast.steps:
            return ControllerDecision(
                fetch_experts=[],
                evict_experts=[],
                defer_experts=[],
                metadata={"priority": {}},
            )

        # ------------------------------------------------------------
        # 1) Aggregate short-horizon forecast
        # ------------------------------------------------------------
        agg_probs: dict[int, float] = {}
        num_used_steps = min(self.lookahead_steps, len(forecast.steps))

        for step in forecast.steps[:num_used_steps]:
            for eid, prob in step.expert_probs.items():
                agg_probs[eid] = agg_probs.get(eid, 0.0) + prob

        # Normalize aggregated probability into [0, 1] scale if using >1 step
        if num_used_steps > 0:
            agg_probs = {eid: prob_sum / num_used_steps for eid, prob_sum in agg_probs.items()}

        # ------------------------------------------------------------
        # 2) Update local priority scores
        # ------------------------------------------------------------
        # Heuristic interpretation:
        # - clearly hot in the short horizon -> boost
        # - mildly relevant -> small decay
        # - irrelevant -> stronger decay
        for eid, agg_prob in agg_probs.items():
            curr = self.priority_scores.get(eid, 0.0)
            if agg_prob >= self.threshold_hot:
                new_score = curr + self.delta_inc
            elif agg_prob > 0.0:
                new_score = curr - self.delta_in_dec
            else:
                new_score = curr - self.delta_out_dec

            self.priority_scores[eid] = max(0.0, min(1.0, new_score))

        # Also decay stale experts not appearing in current aggregated forecast
        missing_ids = set(self.priority_scores.keys()) - set(agg_probs.keys())
        for eid in missing_ids:
            curr = self.priority_scores[eid]
            self.priority_scores[eid] = max(0.0, curr - self.delta_out_dec)

        # ------------------------------------------------------------
        # 3) Choose uncached prefetch candidates
        # ------------------------------------------------------------
        current_residents = set(state.resident_experts)
        pinned = set(state.pinned_experts)

        # Diff-MoE-style intuition:
        #   - global hot experts are special and should not be treated as normal
        #     speculative prefetch candidates
        #   - uncached experts with high forecast and/or high local priority
        #     are eligible for prefetch
        uncached_candidates = [
            eid for eid in agg_probs
            if eid not in current_residents and eid not in self.global_hot
        ]

        uncached_candidates.sort(
            key=lambda eid: (
                self.priority_scores.get(eid, 0.0),
                agg_probs.get(eid, 0.0),
            ),
            reverse=True,
        )

        admitted_fetches: list[int] = []
        deferred_fetches: list[int] = []

        for eid in uncached_candidates:
            pri = self.priority_scores.get(eid, 0.0)
            prob = agg_probs.get(eid, 0.0)

            # Two ways to get admitted:
            # - strong local priority
            # - clearly strong direct short-horizon probability
            if pri >= self.threshold_hot or prob >= self.prefetch_prob_threshold:
                admitted_fetches.append(eid)
            else:
                deferred_fetches.append(eid)

            if len(admitted_fetches) >= self.prefetch_k:
                break

        # ------------------------------------------------------------
        # 4) If no room, evict low-priority residents
        # ------------------------------------------------------------
        available_slots = int(state.memory_capacity - state.memory_used)
        needed_slots = max(0, len(admitted_fetches) - available_slots)

        evict_experts: list[int] = []
        if needed_slots > 0:
            # Prefer evicting residents that are:
            # - not pinned
            # - not global hot
            # - low local priority
            # - low short-horizon aggregated probability
            resident_candidates = [
                eid for eid in current_residents
                if eid not in pinned and eid not in self.global_hot
            ]

            resident_candidates.sort(
                key=lambda eid: (
                    self.priority_scores.get(eid, 0.0),
                    agg_probs.get(eid, 0.0),
                )
            )

            evict_experts = resident_candidates[:needed_slots]

            # If still not enough room, trim fetches conservatively
            freed_slots = len(evict_experts)
            accepted_fetches = available_slots + freed_slots
            if accepted_fetches < len(admitted_fetches):
                deferred_fetches.extend(admitted_fetches[accepted_fetches:])
                admitted_fetches = admitted_fetches[:accepted_fetches]

        return ControllerDecision(
            fetch_experts=admitted_fetches,
            evict_experts=evict_experts,
            defer_experts=deferred_fetches,
            metadata={
                "priority": dict(self.priority_scores),
                "agg_probs": dict(agg_probs),
                "global_hot": sorted(self.global_hot),
            },
        )