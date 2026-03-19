"""Minimal simulator runner loop."""

from moe_offload.controllers.base import BaseController
from moe_offload.predictors.base import BasePredictor
from moe_offload.simulator.env import SimulatorEnv
from moe_offload.simulator.episode_record import StepRecord
from moe_offload.simulator.metrics import aggregate_metrics
from moe_offload.simulator.state import EpisodeResult
from moe_offload.traces.schema import TraceEpisode


def run_episode(
    env: SimulatorEnv,
    episode: TraceEpisode,
    controller: BaseController,
    predictor: BasePredictor,
    horizon: int = 3,
    config_id: str = "default",
) -> tuple[EpisodeResult, list[StepRecord]]:
    """Execute a full episode in the environment with the given predictor & controller."""
    state = env.reset(episode)
    controller.reset()
    records: list[StepRecord] = []

    while not env.is_done():
        idx = env._step_cursor  # accessing internal cursor explicitly since no public getter
        
        # 1. Forecast future
        forecast = predictor.predict(episode, idx, horizon)
        
        # 2. Plan decision
        decision = controller.plan(state, forecast)
        
        # 3. Apply to environment
        state, metrics = env.step(decision)
        
        # 4. Observe outcomes
        controller.observe(state, metrics)

        active = episode.steps[idx].active_experts
        records.append(StepRecord(idx, active, decision, metrics))

    # Compile result
    agg = aggregate_metrics([r.metrics for r in records])
    result = EpisodeResult(
        config_id=config_id,
        controller_name=controller.name,
        predictor_name=predictor.name,
        aggregated_metrics=agg,
    )
    
    return result, records
