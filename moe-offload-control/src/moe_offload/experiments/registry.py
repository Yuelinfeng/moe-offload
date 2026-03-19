"""Simple factory registry."""

from moe_offload.controllers import (
    DiffMoEHeuristicController,
    FastValueAwareController,
    NaiveController,
    PregatedStyleController,
    TopKPrefetchController,
)
from moe_offload.predictors import NoisyPredictor, OracleLitePredictor


def build_predictor(name: str, **kwargs):
    """Instantiate a predictor by name."""
    if name == "oracle_lite":
        return OracleLitePredictor(**kwargs)
    if name == "noisy":
        return NoisyPredictor(**kwargs)
    raise ValueError(f"Unknown predictor {name}")


def build_controller(name: str, **kwargs):
    """Instantiate a controller by name."""
    if name == "naive":
        return NaiveController(**kwargs)
    if name == "topk":
        return TopKPrefetchController(**kwargs)
    if name == "pregated":
        return PregatedStyleController(**kwargs)
    if name == "diffmoe":
        return DiffMoEHeuristicController(**kwargs)
    if name == "fast_value":
        return FastValueAwareController(**kwargs)
    raise ValueError(f"Unknown controller {name}")
