# Trace Schema Reference

This document defines the canonical data structures for traces in the MoE
offloading simulator. All definitions live in `src/moe_offload/traces/schema.py`.

## TraceStep

A single routing decision at one layer for one token-batch.

| Field | Type | Description |
|---|---|---|
| `step_idx` | `int` | Global monotonically increasing step index (0-based) |
| `layer_id` | `int` | Layer index, must be in `[0, num_layers)` |
| `active_experts` | `list[int]` | Expert IDs activated at this step, each in `[0, num_experts)` |
| `regime_id` | `str \| None` | Optional workload regime label (e.g. `"locality"`, `"drift"`) |
| `metadata` | `dict[str, object]` | Arbitrary extra fields |

## ForecastStep

A predictor's output for one future step at one layer.

| Field | Type | Description |
|---|---|---|
| `layer_id` | `int` | Layer index |
| `expert_probs` | `dict[int, float]` | Predicted activation probability per expert |
| `uncertainty` | `dict[int, float]` | Uncertainty estimate per expert |

## ForecastWindow

A window of forecast steps starting from a given position.

| Field | Type | Description |
|---|---|---|
| `current_step_idx` | `int` | The step index this forecast was made at |
| `horizon` | `int` | Number of future steps in the window |
| `steps` | `list[ForecastStep]` | Forecast for each future step |

## WorkloadRegime

Metadata describing a workload regime (phase) within a trace.

| Field | Type | Description |
|---|---|---|
| `regime_id` | `str` | Unique identifier for this regime |
| `description` | `str` | Human-readable description |
| `metadata` | `dict[str, object]` | Arbitrary extra fields |

## TraceEpisode

A complete trace episode — the input to a simulation run.

| Field | Type | Description |
|---|---|---|
| `steps` | `list[TraceStep]` | Ordered sequence of trace steps |
| `num_experts` | `int` | Total number of experts in the model |
| `num_layers` | `int` | Total number of MoE layers |
| `metadata` | `dict[str, object]` | Arbitrary extra fields (e.g. source, generation params) |

### Validation Rules (`validate_episode`)

1. `steps` must contain at least 1 step
2. `step_idx` values must be monotonically increasing
3. Every `active_experts[i]` must be in `[0, num_experts)`
4. Every `layer_id` must be in `[0, num_layers)`

### Example

```python
from moe_offload.traces.schema import TraceStep, TraceEpisode, validate_episode

episode = TraceEpisode(
    steps=[
        TraceStep(step_idx=0, layer_id=0, active_experts=[1, 3], regime_id=None, metadata={}),
        TraceStep(step_idx=1, layer_id=0, active_experts=[0, 2], regime_id=None, metadata={}),
    ],
    num_experts=4,
    num_layers=1,
    metadata={"source": "manual"},
)
validate_episode(episode)  # passes silently
```
