from __future__ import annotations

from typing import Dict, List

from moe_offload.traces.schema import ForecastStep, ForecastWindow, TraceEpisode

from .base import BasePredictor


class OracleLitePredictor(BasePredictor):
    """Near-oracle predictor based on future ground-truth steps.

    - Uses the true future active experts from the trace episode.
    - Assigns high probability and low uncertainty to actually active experts.
    - Assigns small tail probability and higher uncertainty to others.
    - Deterministic by design (no randomness), to keep tests stable.
    """

    def __init__(
        self,
        active_mass: float = 0.9,
        inactive_tail_mass: float = 0.1,
        active_uncertainty: float = 0.1,
        inactive_uncertainty: float = 0.5,
    ) -> None:
        self._active_mass = active_mass
        self._inactive_tail_mass = inactive_tail_mass
        self._active_uncertainty = active_uncertainty
        self._inactive_uncertainty = inactive_uncertainty

    def predict(
        self, episode: TraceEpisode, current_index: int, horizon: int
    ) -> ForecastWindow:
        horizon = max(horizon, 0)
        num_steps = len(episode.steps)
        if current_index >= num_steps - 1 or horizon == 0:
            return ForecastWindow(
                current_step_idx=current_index,
                horizon=horizon,
                steps=[],
            )

        # Clamp window to episode tail
        max_future = num_steps - (current_index + 1)
        window_len = min(horizon, max_future)

        steps: List[ForecastStep] = []
        all_experts = range(episode.num_experts)

        for offset in range(1, window_len + 1):
            trace_step = episode.steps[current_index + offset]
            active_set = set(trace_step.active_experts)

            probs: Dict[int, float] = {}
            uncs: Dict[int, float] = {}

            if active_set:
                per_active = self._active_mass / float(len(active_set))
            else:
                per_active = 0.0

            num_inactive = episode.num_experts - len(active_set)
            per_inactive = (
                self._inactive_tail_mass / float(num_inactive)
                if num_inactive > 0
                else 0.0
            )

            for eid in all_experts:
                if eid in active_set:
                    probs[eid] = per_active
                    uncs[eid] = self._active_uncertainty
                else:
                    probs[eid] = per_inactive
                    uncs[eid] = self._inactive_uncertainty

            # Small deterministic renormalization to avoid drift from rounding.
            total = sum(probs.values()) or 1.0
            for eid in probs:
                probs[eid] = probs[eid] / total

            steps.append(
                ForecastStep(
                    layer_id=trace_step.layer_id,
                    expert_probs=probs,
                    uncertainty=uncs,
                )
            )

        return ForecastWindow(
            current_step_idx=current_index,
            horizon=horizon,
            steps=steps,
        )

