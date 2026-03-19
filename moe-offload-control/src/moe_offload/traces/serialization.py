"""Minimal placeholder for trace serialization."""

from moe_offload.traces.schema import TraceEpisode


def save_trace(episode: TraceEpisode, path: str) -> None:
    """Serialize trace to JSON."""
    # Placeholder: currently handled informally or not critical for Phase 2 memory-only test
    pass


def load_trace(path: str) -> TraceEpisode:
    """Load trace from JSON."""
    raise NotImplementedError("load_trace not implemented yet.")
