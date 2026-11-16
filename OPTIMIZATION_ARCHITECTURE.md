# ETL Performance Intelligence: Integration Architecture

## Design principles

- Extend the existing Streamlit ETL flow; do not create a second application.
- Keep profiling, execution, correctness checks, and benchmark arithmetic deterministic. Qwen3 can explain evidence and suggest candidates, but cannot supply measurements or authorize execution.
- Keep the approved baseline immutable. Optimization creates a separate candidate plan and output until validation and final user selection succeed.
- Use typed structured results between services and persist them through the existing SQLite workflow database.
- Add optional capabilities behind narrow interfaces so the current Pandas/Python and SQLite paths remain usable without an LLM or optional engine.
- Start sequentially and cap resources. Enable concurrency, caches, or out-of-core engines only when measurements and correctness checks justify them.

## Target integration flow

```text
Streamlit upload / existing ETL plan
              |
              v
      Baseline plan approval
              |
              v
   Instrumented deterministic run
      | metrics + output + audit
      v
 Profiling and bottleneck analysis
      |
      +--> SQL/query-plan inspection
      +--> memory and resource assessment
      +--> conservative transformation analysis
      +--> dependency/parallelism analysis
      +--> cache eligibility and source fingerprint
              |
              v
  Structured candidate recommendations
    (rules first; optional Qwen3 explanation)
              |
              v
 User reviews / edits / approves candidate
              |
              v
 Isolated bounded candidate execution
              |
              v
 Correctness + quality validation
              |
              v
 Comparable benchmark and honest deltas
              |
              v
 User selects baseline or candidate
              |
              v
 Persist immutable version, report, and audit events
```

## Proposed service boundaries

| Service | Responsibility and evidence |
|---|---|
| `performance_profiler` | Wrap each existing operation and extraction/load boundaries. Capture `perf_counter` duration, row counts, throughput, Python peak allocations (`tracemalloc`), process CPU/RSS snapshots (`psutil` when installed), and stage labels. Clearly distinguish sampled RSS from Python allocations and unavailable disk/network/GPU metrics. |
| `resource_allocator` | Read available memory, core count, current process/system load, and free disk space. Produce conservative resource recommendations with configured ceilings; it does not start work by itself. |
| `memory_optimizer` | Estimate DataFrame size from actual `memory_usage(deep=True)` and compare against available memory and safety reserve. Recommend in-memory, selective/optimized dtypes, chunking, or DuckDB/Parquet only when supported by the operation and verified dependencies. |
| `transformation_optimizer` | Analyze allowlisted operation metadata and explicit input/output columns. Build a dependency DAG, identify only provably redundant steps, and provide before/after plans plus a reason per proposed change. No source-text similarity pruning. |
| `parallel_execution_planner` | Identify independent DAG nodes and estimate whether their work and memory permit bounded concurrency. Phase implementation should expose a recommendation first; execute in parallel only after profiling and equality tests show benefit. |
| `query_optimizer` | Generate SQLite `EXPLAIN QUERY PLAN` evidence for supported SQL; later add DuckDB explain/profiling where suitable. Report scan/sort/join evidence and index suggestions with write/storage trade-offs. Do not automatically create indexes or run arbitrary generated SQL. |
| `join_optimizer` | Inspect only structured join operations when present. Use table sizes, key nulls/uniqueness, and native query plans. Recommend candidates without changing join order/semantics unless equivalence tests pass. |
| `caching_engine` | Key entries by content fingerprint, canonical operation/configuration, engine/version, and relevant dependencies. Verify key and artifact integrity, apply expiry/size policy, record hits/misses, and never cache unvalidated results. |
| `incremental_processor` | Introduce only for sources with reliable change identity/partition metadata. Initially support full refresh and a safe change-based mode; refuse incremental reuse when the operation is not known to be incremental-safe. |
| `bottleneck_detector` | Rank bottlenecks from measured operation time, memory, CPU, throughput, and explain-plan evidence. Include the evidence and confidence; do not infer unavailable network/GPU/disk metrics. |
| `multi_objective_optimizer` | Score deterministic candidate plans according to user-selected speed, memory, balanced, cost-aware, or quality-first profile, subject to correctness and resource constraints. Present trade-offs; no single weighted score overrides hard correctness constraints. |
| `optimization_orchestrator` | Coordinate profiling, candidate generation, approval boundaries, isolated execution, validation, comparison, and persistence. Use the existing direct call architecture initially; avoid adding LangGraph until concrete orchestration complexity justifies it. |
| `optimization_validator` | Compare outputs with explicit semantics: schema/order policy, row multiplicity, null behavior, and value equality (including dtype policy). Validate quality metrics and stop on mismatches or failed safety gates. |
| `benchmark_engine` | Run baseline and candidate under equivalent settings, collect repeated measurements where practical, and compute actual time/memory/CPU/throughput deltas with negative results and measurement caveats retained. Include warm/cold cache state where controllable. |
| `workflow_state` extension | Add versioned optimization runs, recommendations, approval decisions, benchmark results, and immutable dataset fingerprints to existing SQLite tables. Add migrations that preserve `workflow_events` and `workflow_snapshots`. |
| Streamlit performance section | Add performance profile, bottleneck evidence, candidate plan and diff, approval/rejection/edit controls, execution progress/status, correctness report, benchmark comparison, profile selector, history, and rollback/version selection. |

## Qwen3 integration

The current implementation uses a local Ollama endpoint and heuristics, not Hugging Face Transformers. Preserve the current Ollama route while introducing a provider interface in the AI phase:

- Provider options: current Ollama-compatible local endpoint and optional Hugging Face Transformers local inference.
- Configurable model identifier; do not load multiple large models at once.
- Optimization prompts contain only sanitized plan metadata, query plans, aggregated metrics, and relevant schema descriptors—not full datasets or secrets.
- Validate model output against a strict typed schema, allowlisted optimization kinds, known operation IDs, and known columns.
- Treat provider unavailability or invalid output as an explicit status; deterministic profiler and optimizer remain functional.
- Do not let an LLM author executable Python/SQL that bypasses the existing operation registry.

## Persistence model

Extend `workflow_history.db` with additive migrations, using stable run/version identifiers and JSON for versioned structured payloads where appropriate. Candidate tables should represent:

- Optimization run: source fingerprint, baseline plan/version, profile, timestamps, engine/runtime versions, and status.
- Recommendation: evidence, proposed change, expected impact as an estimate, trade-offs, validation requirements, and approval decision.
- Benchmark: raw repeated measurements for baseline and candidate, output-equivalence result, computed deltas, cache state, and measurement caveats.
- Selected version: baseline/candidate selection and rollback reference.

Do not store raw source data in SQLite JSON snapshots. A dataset cache/artifact must use explicit storage limits, content hashes, safe local paths, and retention policy; otherwise require the source upload again to resume execution.

## Safety and execution boundaries

- Reuse the explicit operation allowlist. Reject unknown operations rather than treating them as successful no-ops.
- Treat optimized SQL as data to validate. Restrict execution to the configured local engine and supported statements; use bound parameters for values.
- Apply timeout, memory headroom, maximum concurrency, and artifact-size limits before execution.
- Run candidate plans against a copy or isolated temporary database. Keep baseline and original inputs intact.
- Gate each optimization proposal, optimized execution, correctness result, and final version selection on user approval.
- Preserve complete audit events and surface errors; never report success after a failed candidate step.

## Integration touchpoints

- `app.py`: compose a performance-intelligence section after baseline ETL plan approval/execution; add profile selection and version selection.
- `src/etl/execution_engine.py`: add an instrumentation wrapper around existing operations without changing operation semantics in the first profiling phase.
- `src/etl/sql_generator.py`: expose structured query/operation metadata for explain and analysis; do not blindly rewrite strings.
- `src/etl/llm_service.py`: retain ETL planning API; add a separate optimization advisor/provider interface later.
- `src/etl/workflow_state.py`: additive schema migration and run/recommendation/benchmark persistence.
- `requirements.txt`: add only dependencies proven necessary by implementation and test environment; `psutil` is a likely profiling dependency but is not currently declared.
