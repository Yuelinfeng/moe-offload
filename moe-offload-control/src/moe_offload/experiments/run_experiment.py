"""Experiment runner wrapper."""

import json
import os

from moe_offload.simulator.env import SimulatorEnv
from moe_offload.simulator.runner import run_episode


def run_experiment(
    episode,
    env_config,
    controller,
    predictor,
    horizon,
    config_id,
    out_dir="data/results/raw/",
):
    """Encapsulates a single experiment execution and output IO."""
    env = SimulatorEnv(env_config)
    result, _ = run_episode(env, episode, controller, predictor, horizon, config_id)

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir, f"{config_id}_{controller.name}_{predictor.name}.json"
    )

    with open(out_file, "w") as f:
        json.dump(result.aggregated_metrics, f, indent=2)

    return result
