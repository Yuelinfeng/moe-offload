"""Base interfaces for predictors."""

import abc

from moe_offload.traces.schema import ForecastWindow, TraceEpisode


class BasePredictor(abc.ABC):
    """Abstract base class for all predictors.
    
    Predictors strictly make forecasts using future trace steps and optionally
    controlled noise. They do not maintain realistic neural model weights.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """String identifier for the predictor."""
        pass

    @abc.abstractmethod
    def predict(
        self,
        trace_episode: TraceEpisode,
        current_index: int,
        horizon: int,
    ) -> ForecastWindow:
        """Forecast the next `horizon` steps based on the trace episode.
        
        Parameters
        ----------
        trace_episode : TraceEpisode
            The full trace episode containing ground truth future.
        current_index : int
            The current step index in the simulation.
        horizon : int
            Number of future steps to predict.
            
        Returns
        -------
        ForecastWindow
            The forecast for the requested window.
        """
        pass
