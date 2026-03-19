# Phase 1 · Batch 1：最小可运行 Simulator（严格 16 文件）

## 本批目标

搭出最小可运行 simulator 骨架：

- `pip install -e .` 成功
- `pytest tests/test_trace_schema.py tests/test_env_smoke.py -v` 全绿
- 可手工构造 5-step trace + inline trivial controller，跑完 env 循环拿到合法 `StepMetrics`

---

## 本批冻结接口

### traces/schema.py 冻结

| Dataclass | 必须字段 |
|---|---|
| `TraceStep` | `step_idx: int`, `layer_id: int`, `active_experts: list[int]`, `regime_id: str \| None`, `metadata: dict[str, object]` |
| `ForecastStep` | `layer_id: int`, `expert_probs: dict[int, float]`, `uncertainty: dict[int, float]` |
| `ForecastWindow` | `current_step_idx: int`, `horizon: int`, `steps: list[ForecastStep]` |
| `WorkloadRegime` | `regime_id: str`, `description: str`, `metadata: dict[str, object]` |
| `TraceEpisode` | `steps: list[TraceStep]`, `num_experts: int`, `num_layers: int`, `metadata: dict[str, object]` |

冻结函数：`validate_episode(episode: TraceEpisode) -> None`

### simulator/state.py 冻结

| Dataclass | 必须字段 |
|---|---|
| `SimulatorState` | `step_idx: int`, `resident_experts: set[int]`, `pinned_experts: set[int]`, `memory_used: int \| float`, `memory_capacity: int \| float`, `bandwidth_capacity: int \| float`, `controller_context: dict` |
| `ControllerDecision` | `fetch_experts: list[int]`, `evict_experts: list[int]`, `defer_experts: list[int]`, `metadata: dict[str, object]` |
| `StepMetrics` | `stall_latency: float`, `transfer_cost: float`, `misprefetch_count: int`, `reload_count: int`, `cache_miss_count: int`, `service_cost: float` |
| `EpisodeResult` | `config_id: str`, `controller_name: str`, `predictor_name: str`, `aggregated_metrics: dict[str, float]`, `step_records_path: str \| None` |

冻结函数：`make_initial_state(config: dict) -> SimulatorState`

---

## 第一批最终文件清单（严格 16 个）

| # | 相对路径 | 类别 |
|---|---|---|
| 1 | [README.md](file:///d:/workspace/moe-offload-control/README.md) | 基础设施 |
| 2 | [pyproject.toml](file:///d:/workspace/moe-offload-control/pyproject.toml) | 基础设施 |
| 3 | [requirements-dev.txt](file:///d:/workspace/moe-offload-control/requirements-dev.txt) | 基础设施 |
| 4 | [Makefile](file:///d:/workspace/moe-offload-control/Makefile) | 基础设施 |
| 5 | `docs/trace_schema.md` | 文档 |
| 6 | [src/moe_offload/__init__.py](file:///d:/workspace/moe-offload-control/src/moe_offload/__init__.py) | 包入口 |
| 7 | `src/moe_offload/traces/__init__.py` | 子包入口 |
| 8 | [src/moe_offload/traces/schema.py](file:///d:/workspace/moe-offload-control/src/moe_offload/traces/schema.py) | 核心定义 |
| 9 | `src/moe_offload/simulator/__init__.py` | 子包入口 |
| 10 | [src/moe_offload/simulator/state.py](file:///d:/workspace/moe-offload-control/src/moe_offload/simulator/state.py) | 核心定义 |
| 11 | [src/moe_offload/simulator/cost_model.py](file:///d:/workspace/moe-offload-control/src/moe_offload/simulator/cost_model.py) | 模型 |
| 12 | `src/moe_offload/simulator/memory_model.py` | 模型 |
| 13 | `src/moe_offload/simulator/bandwidth_model.py` | 模型 |
| 14 | [src/moe_offload/simulator/env.py](file:///d:/workspace/moe-offload-control/src/moe_offload/simulator/env.py) | 主循环 |
| 15 | [tests/test_trace_schema.py](file:///d:/workspace/moe-offload-control/tests/test_trace_schema.py) | 测试 |
| 16 | [tests/test_env_smoke.py](file:///d:/workspace/moe-offload-control/tests/test_env_smoke.py) | 测试 |

> [!NOTE]
> **[.gitignore](file:///d:/workspace/moe-offload-control/.gitignore) 不在本批**。原因：本批专注可运行代码骨架，[.gitignore](file:///d:/workspace/moe-offload-control/.gitignore) 是仓库管理文件，不影响 `pip install` 或 `pytest`，放入 Batch 2 一并处理。

---

## 逐文件职责

### 1. [README.md](file:///d:/workspace/moe-offload-control/README.md)
项目说明文档。必须明确写出：
- `external/Pregated_MoE` 是外部参考项目，**不参与** Batch 1 代码实现
- Batch 1 只实现 minimal trace-driven simulator
- 不包含 controller / predictor / runtime adapter
- 快速上手指南（install + test 两条命令）

### 2. `pyproject.toml`
包元数据 + editable install 配置。
- **零 runtime 依赖**（Batch 1 代码只用 stdlib `dataclasses`, `typing`）
- `[project.optional-dependencies]` dev 引用 `requirements-dev.txt`

### 3. `requirements-dev.txt`
仅两项：`pytest`, `ruff`

### 4. `Makefile`
四个 target：`install`（pip install -e .）, `test`（pytest）, `lint`（ruff check）, `clean`（rm 产物）

### 5. `docs/trace_schema.md`
Trace schema 的人类可读文档：每个 dataclass 各字段语义、合法值范围、一个简短示例

### 6. `src/moe_offload/__init__.py`
从 `traces.schema` 和 `simulator.state` re-export 全部冻结 dataclass，用户 `from moe_offload import TraceStep` 即可

### 7. `src/moe_offload/traces/__init__.py`
re-export：`TraceStep`, `ForecastStep`, `ForecastWindow`, `TraceEpisode`, `WorkloadRegime`, `validate_episode`

### 8. `src/moe_offload/traces/schema.py`
**Trace 子系统唯一权威定义文件**。5 个 dataclass + `validate_episode()`。检查：step_idx 单调递增、expert ID ∈ [0, num_experts)、layer_id ∈ [0, num_layers)、至少 1 step

### 9. `src/moe_offload/simulator/__init__.py`
re-export：`SimulatorState`, `ControllerDecision`, `StepMetrics`, `EpisodeResult`, `SimulatorEnv`

### 10. `src/moe_offload/simulator/state.py`
4 个 dataclass + `make_initial_state(config) -> SimulatorState` 纯函数。**无 class、无状态更新逻辑**

### 11. `src/moe_offload/simulator/cost_model.py`
**极简实现**，一个 `CostModel` class（或纯函数集合），具体要求：
- 构造参数：`alpha_transfer`, `alpha_misprefetch`, `alpha_reload`（直接传数值，不依赖外部 config 文件）
- `compute(decision, state, ground_truth_step, future_steps) -> StepMetrics`
- miss = active_expert not in resident_set → stall
- misprefetch = fetched but unused within lookahead window
- reload = evicted but needed within lookahead window
- `service_cost = stall + α_t·transfer + α_m·misprefetch + α_r·reload`
- **不做**：OO 继承体系、插件注册、配置文件加载

### 12. `src/moe_offload/simulator/memory_model.py`
**极简实现**，一个 `MemoryModel` class，具体要求：
- 构造参数：`capacity: int`（最大 expert 槽数）
- `can_fit(n: int) -> bool`
- `allocate(expert_ids: list[int])` — 加入，超容量抛异常
- `release(expert_ids: list[int])` — 移除
- `current_usage() -> int`
- 内部用一个 `set` 即可
- **不做**：多级缓存、LRU 框架、per-expert 大小差异

### 13. `src/moe_offload/simulator/bandwidth_model.py`
**极简实现**，一个 `BandwidthModel` class，具体要求：
- 构造参数：`capacity: int`（每 step 最大可传输 expert 数）
- `compute_transfer(n_fetch: int) -> (transfer_cost: float, stall: float)`
- 若 `n_fetch <= capacity`：`transfer_cost = n_fetch`, `stall = 0`
- 若 `n_fetch > capacity`：`transfer_cost = n_fetch`, `stall = (n_fetch - capacity) * per_expert_latency`
- **不做**：多 stream、overlap、queueing

### 14. `src/moe_offload/simulator/env.py`
**极简实现**，一个 `SimulatorEnv` class，具体要求：
- `__init__(config: dict)` — 内部创建 `MemoryModel`, `BandwidthModel`, `CostModel`，所有参数从 config dict 取
- `reset(episode: TraceEpisode) -> SimulatorState` — 初始化状态
- `step(decision: ControllerDecision) -> tuple[SimulatorState, StepMetrics]` — 单步推进：apply decision to resident set（via memory_model）→ 计算 metrics（via cost_model）→ 推进 step_idx → 返回
- `is_done() -> bool`
- 不含 controller 调度循环（留给 runner / test 手工驱动）
- **不做**：event loop、callback 系统、observation space 抽象

### 15. `tests/test_trace_schema.py`
- 创建合法 `TraceEpisode` → `validate_episode` 无异常
- step_idx 非单调 → `ValueError`
- expert ID 越界 → `ValueError`
- 空 episode → `ValueError`

### 16. `tests/test_env_smoke.py`
- 构造 5-step 手工 trace（3 experts, 1 layer, memory_capacity=2）
- no-op controller（空 decision）跑完：每步 cache_miss > 0, stall > 0
- perfect controller（每步先 evict 腾位 + fetch 正确 experts）跑完：stall = 0
- 验证 metrics 值非负、类型正确

---

## 文件依赖关系

```
README.md                    (standalone)
pyproject.toml               (standalone)
requirements-dev.txt         (standalone)
Makefile                     (standalone)
docs/trace_schema.md         (standalone)

traces/schema.py             (standalone — 纯 stdlib)
traces/__init__.py       ──► traces/schema.py

simulator/state.py           (standalone — 纯 stdlib)
simulator/memory_model.py    (standalone — 纯 stdlib)
simulator/bandwidth_model.py (standalone — 纯 stdlib)
simulator/cost_model.py  ──► simulator/state.py, traces/schema.py
simulator/env.py         ──► simulator/state.py, simulator/cost_model.py,
                             simulator/memory_model.py, simulator/bandwidth_model.py,
                             traces/schema.py
simulator/__init__.py    ──► simulator/state.py, simulator/env.py

__init__.py              ──► traces/__init__.py, simulator/__init__.py

tests/test_trace_schema.py ──► traces/schema.py
tests/test_env_smoke.py    ──► simulator/env.py, simulator/state.py, traces/schema.py
```

---

## 本批暂不实现

| 类别 | 内容 |
|---|---|
| simulator 补充 | `metrics.py`, `episode_record.py`, `runner.py` |
| controllers | `base.py` 及全部具体 controller |
| predictors | 全部 |
| experiments | 全部 |
| runtime_adapters | 全部 |
| scripts / analysis | 全部 |
| utils 子包 | `io.py`, `seed.py`, `logging.py` |
| 辅助类型 | `types.py`, `constants.py` |
| 配置文件 | `configs/**/*.yaml` |
| 文档 | `docs/` 下除 `trace_schema.md` 外全部 |
| 测试 | 除上述两个外全部 |
| 仓库管理 | `.gitignore`（Batch 2） |
