"""Noisy predictor with configurable error levels."""

import random

from moe_offload.predictors.base import BasePredictor
from moe_offload.traces.schema import ForecastStep, ForecastWindow, TraceEpisode


class NoisyPredictor(BasePredictor):
    """Configurable predictor to simulate 'medium' and 'bad' accuracy presets.
    
    Generates non-perfect forecasts by probabilistically missing true active
    experts (false negatives) and hallucinating inactive ones (false positives).
    Uncertainty strictly scales with the preset noise magnitude.
    """

    def __init__(self, preset: str = "medium") -> None:
        self.preset = preset
        if preset == "medium":
            self.base_noise = 0.3
            self.miss_prob = 0.2
        elif preset == "bad":
            self.base_noise = 0.6
            self.miss_prob = 0.6
        else:
            self.base_noise = 0.0
            self.miss_prob = 0.0

    @property
    def name(self) -> str:
        return f"noisy_predictor_{self.preset}"

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
                real_active = eid in true_active
                
                # False negative: mispredict an active expert as low prob
                if real_active and random.random() < self.miss_prob:
                    prob = random.uniform(0.0, 0.4)
                # False positive: hallucinate an inactive expert as high prob
                elif not real_active and random.random() < self.miss_prob:
                    prob = random.uniform(0.6, 1.0)
                # Correct-ish prediction but with jitter
                else:
                    prob = random.uniform(0.7, 1.0) if real_active else random.uniform(0.0, 0.3)

                probs[eid] = prob
                uncert[eid] = max(0.0, self.base_noise + random.uniform(-0.1, 0.1))

            steps.append(
                ForecastStep(
                    layer_id=true_step.layer_id, 
                    expert_probs=probs, 
                    uncertainty=uncert
                )
            )

        return ForecastWindow(
            current_step_idx=current_index, horizon=horizon, steps=steps
        )
