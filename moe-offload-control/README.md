# MoE Offload Control — Trace-Driven Simulator

A minimal trace-driven simulator for evaluating MoE expert offloading control
strategies under constrained GPU memory and host-GPU bandwidth.

## Research Question

> Under the same predictor and resource budget, is uncertainty-aware dynamic
> control more effective than heuristic prediction-triggered caching for MoE
> expert offloading?

## Current Status: Phase 1 / Batch 1

This batch implements the **minimal runnable simulator skeleton** only:

- Frozen dataclass definitions (TraceStep, SimulatorState, etc.)
- Trace schema validation
- Minimal CostModel / MemoryModel / BandwidthModel / SimulatorEnv
- Smoke tests

**Not included in this batch:**

- Controllers (naive, pregated_style, diffmoe_heuristic, topk_prefetch,
  fast_value_aware, two_timescale)
- Predictors (oracle_lite, noisy_predictor, forecast_adapter)
- Runtime adapters for Pregated MoE integration
- Experiment orchestration, sweep, analysis

## Relationship to external/Pregated_MoE

`external/Pregated_MoE/` is an external reference project. It is **not** part of
this codebase and is **not modified** by any code here. Future integration will
be done through `runtime_adapters/` (not yet implemented).

## Quick Start

```bash
pip install -e .
pip install -r requirements-dev.txt
pytest tests/test_trace_schema.py tests/test_env_smoke.py -v
```

## Directory Structure

```
moe-offload-control/
├── src/moe_offload/
│   ├── traces/          # Trace schema and data definitions
│   │   └── schema.py    # TraceStep, ForecastStep, TraceEpisode, etc.
│   └── simulator/       # Simulation engine
│       ├── state.py     # SimulatorState, ControllerDecision, StepMetrics
│       ├── cost_model.py
│       ├── memory_model.py
│       ├── bandwidth_model.py
│       └── env.py       # SimulatorEnv main loop
├── tests/
├── configs/             # (not yet populated)
├── scripts/             # (not yet populated)
├── analysis/            # (not yet populated)
└── docs/
```
