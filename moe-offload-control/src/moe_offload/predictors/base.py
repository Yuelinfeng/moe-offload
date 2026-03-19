from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from moe_offload.traces.schema import ForecastWindow, TraceEpisode


class SupportsPredict(Protocol):
    """Structural protocol for simple type checking in tests."""

    def predict(
        self, episode: TraceEpisode, current_index: int, horizon: int
    ) -> ForecastWindow: ...


class BasePredictor(ABC):
    """Abstract interface for trace-driven expert demand predictors.

    Phase 2 intentionally keeps this interface minimal: predictors are
    pure functions of the trace episode and the current index, and do
    not hold references to the simulator environment.
    """

    @abstractmethod
    def predict(
        self, episode: TraceEpisode, current_index: int, horizon: int
    ) -> ForecastWindow:
        """Produce a :class:`ForecastWindow` starting after ``current_index``.

        Implementations must:

        - Treat ``current_index`` as the index of the *current* step in
          ``episode.steps``. The first predicted step therefore
          corresponds to ``episode.steps[current_index + 1]`` when
          available.
        - Truncate the horizon if the episode tail is shorter than
          ``horizon``.
        - Allow ``horizon == 0`` or ``current_index`` already at the
          last step, in which case ``ForecastWindow.steps`` should be
          an empty list while other fields remain well-formed.
        """
        raise NotImplementedError

