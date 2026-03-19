from __future__ import annotations

from typing import Dict, List

from moe_offload.traces.schema import ForecastStep, ForecastWindow, TraceEpisode

from .base import BasePredictor


class _BaseNoisyPredictor(BasePredictor):
    """Shared helpers for medium/bad predictors."""

    def __init__(
        self,
        active_mass: float,
        inactive_tail_mass: float,
        active_uncertainty: float,
        inactive_uncertainty: float,
        extra_hot_fraction: float,
    ) -> None:
        self._active_mass = active_mass
        self._inactive_tail_mass = inactive_tail_mass
        self._active_uncertainty = active_uncertainty
        self._inactive_uncertainty = inactive_uncertainty
        self._extra_hot_fraction = extra_hot_fraction

    def _build_window(
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

        max_future = num_steps - (current_index + 1)
        window_len = min(horizon, max_future)

        steps: List[ForecastStep] = []
        all_experts = list(range(episode.num_experts))

        for offset in range(1, window_len + 1):
            trace_step = episode.steps[current_index + offset]
            true_active = set(trace_step.active_experts)

            probs: Dict[int, float] = {}
            uncs: Dict[int, float] = {}

            # Choose a small, deterministic subset of inactive experts as
            # \"extra hot\" to simulate systematic mistakes.
            inactive = [e for e in all_experts if e not in true_active]
            k_extra = int(len(inactive) * self._extra_hot_fraction)
            extra_hot = set(inactive[:k_extra]) if k_extra > 0 else set()

            effective_active = true_active | extra_hot

            if effective_active:
                per_active = self._active_mass / float(len(effective_active))
            else:
                per_active = 0.0

            num_inactive = episode.num_experts - len(effective_active)
            per_inactive = (
                self._inactive_tail_mass / float(num_inactive)
                if num_inactive > 0
                else 0.0
            )

            for eid in all_experts:
                if eid in effective_active:
                    probs[eid] = per_active
                    uncs[eid] = self._active_uncertainty
                else:
                    probs[eid] = per_inactive
                    uncs[eid] = self._inactive_uncertainty

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


class MediumNoisePredictor(_BaseNoisyPredictor):
    """Predictor with moderate systematic errors and uncertainty."""

    def __init__(self) -> None:
        super().__init__(
            active_mass=0.75,
            inactive_tail_mass=0.25,
            active_uncertainty=0.3,
            inactive_uncertainty=0.7,
            extra_hot_fraction=0.2,
        )

    def predict(
        self, episode: TraceEpisode, current_index: int, horizon: int
    ) -> ForecastWindow:
        return self._build_window(episode, current_index, horizon)


class BadPredictor(_BaseNoisyPredictor):
    """Predictor with high error and high uncertainty.

    Approximates a nearly-uniform, poorly calibrated forecast that still
    weakly reflects the underlying trace structure.
    """

    def __init__(self) -> None:
        super().__init__(
            active_mass=0.55,
            inactive_tail_mass=0.45,
            active_uncertainty=0.6,
            inactive_uncertainty=0.9,
            extra_hot_fraction=0.4,
        )

    def predict(
        self, episode: TraceEpisode, current_index: int, horizon: int
    ) -> ForecastWindow:
        return self._build_window(episode, current_index, horizon)

