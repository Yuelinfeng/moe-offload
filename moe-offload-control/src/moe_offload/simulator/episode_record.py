"""Structure for recording un-merged history."""

from dataclasses import dataclass

from moe_offload.simulator.state import ControllerDecision, StepMetrics


@dataclass
class StepRecord:
    """Records the full context of a single simulation step."""
    step_idx: int
    active_experts: list[int]
    decision: ControllerDecision
    metrics: StepMetrics
