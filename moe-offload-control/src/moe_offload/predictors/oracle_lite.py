"""Oracle Lite predictor that peeks at future trace steps."""

import random

from moe_offload.predictors.base import BasePredictor
from moe_offload.traces.schema import ForecastStep, ForecastWindow, TraceEpisode


class OracleLitePredictor(BasePredictor):
    """Perfect or near-perfect forecaster using ground truth with optional tiny noise.
    
    It assigns high probability to practically active experts and low probability to others,
    with an optional controllable uniform noise injected into both probability and uncertainty.
    """

    def __init__(self, noise_level: float = 0.0) -> None:
        self.noise_level = noise_level

    @property
    def name(self) -> str:
        return f"oracle_lite_n{self.noise_level:.2f}"

    def predict(
        self, trace_episode: TraceEpisode, current_index: int, horizon: int
    ) -> ForecastWindow:
        steps = []
        num_experts = trace_episode.num_experts

        for i in range(1, horizon + 1):
            future_idx = current_index + i
            if future_idx >= len(trace_episode.steps):
                break

            true_step = trace_episode.steps[future_idx]
            true_active = set(true_step.active_experts)
            probs: dict[int, float] = {}
            uncert: dict[int, float] = {}

            for eid in range(num_experts):
                base_p = 1.0 if eid in true_active else 0.0
                jitter = random.uniform(-self.noise_level, self.noise_level)
                probs[eid] = max(0.0, min(1.0, base_p + jitter))
                uncert[eid] = self.noise_level if self.noise_level > 0 else 0.0

            steps.append(
                ForecastStep(
                    layer_id=true_step.layer_id,
                    expert_probs=probs,
                    uncertainty=uncert,
                )
            )

        return ForecastWindow(
            current_step_idx=current_index,
            horizon=horizon,
            steps=steps,
        )
