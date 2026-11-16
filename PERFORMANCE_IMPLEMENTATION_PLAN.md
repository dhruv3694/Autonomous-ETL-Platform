# ETL Performance Intelligence: Implementation Plan

This plan follows the requested incremental phases. Phase 0, Phase 1, and the SQL-focused Phase 2.1 scope are complete; the remaining later phases are not implemented.

## Phase 0 — Repository audit (complete)

- [x] Inspect current app, ingestion, LLM, execution, SQL, persistence, dependencies, and test structure.
- [x] Record existing capabilities, discrepancies, reuse opportunities, and safety constraints.
- [x] Define integration architecture and phase sequence.
- [x] Make no runtime, dependency, or database changes.

## Phase 1 — Deterministic profiling and bottleneck evidence

- [x] Define stage/operation metric records and instrumentation helpers.
- [x] Measure extraction, schema profiling, individual transformations, and the SQLite source-load boundary with `perf_counter`.
- [x] Capture records processed, throughput, process CPU time, sampled RSS, Python peak allocations, and DataFrame byte estimates.
- [x] Record unsupported GPU utilization as unavailable and identify disk/network counters as system-wide snapshots rather than process-attributed metrics.
- [x] Add evidence-based bottleneck ranking and unit tests using deterministic synthetic inputs.
- [x] Persist profile data through existing workflow events and run snapshots; render an operation-level performance view.
- [x] Regression-test representative Python and SQLite operation results and verify instrumentation is attached; broader operation-by-operation regression remains for Phase 8.

## Phase 2 — Conservative SQL, transformation, and join analysis

- [x] Add structured SQL metadata for supported operations and their input/output columns.
- [x] Capture SQLite `EXPLAIN QUERY PLAN`; keep it distinct from runtime measurement.
- [x] Provide conservative evidence-based SQL guidance without auto-creating indexes.
- [x] Generate a narrow no-op SQL candidate; compare outputs and benchmark repeated runs.
- [ ] Add Python transformation analysis and join analysis only in separately scoped future work.

The SQL-focused Phase 2.1 scope is complete; see `SQL_OPTIMIZATION_ARCHITECTURE.md`. No general performance improvement is claimed.

## Phase 3 — Resource-aware and memory-aware execution

- [ ] Measure resource availability and establish configurable safety reserves/limits.
- [ ] Choose in-memory versus chunked/out-of-core strategy from measured input size, operation support, and available RAM.
- [ ] Add bounded parallelism recommendations from the operation DAG; benchmark before enabling execution.
- [ ] Evaluate DuckDB/Parquet paths only with representative benchmarks and correctness tests.
- [ ] Test resource ceilings, cancellation, timeout, and failure surfacing.

## Phase 4 — Fingerprinted caching and safe incremental processing

- [ ] Compute source content fingerprints and canonical transformation/configuration hashes.
- [ ] Define local cache storage, integrity verification, expiry, maximum size, and invalidation policy.
- [ ] Cache only deterministic, validated intermediate outputs.
- [ ] Report cache hit/miss and cold/warm runs.
- [ ] Implement incremental refresh only for source/change metadata and operations with known incremental semantics; default to full refresh otherwise.
- [ ] Test unchanged inputs, changed inputs, tampered entries, expiry, and invalidation.

## Phase 5 — Optimization advisor and multi-objective selection

- [ ] Keep measured facts deterministic and independent of model responses.
- [ ] Add a provider abstraction preserving local Ollama and adding optional local Hugging Face Transformers/Qwen3.
- [ ] Support configurable model selection, one active model at a time, CPU/GPU availability, and optional quantization only where compatible.
- [ ] Send summarized metrics/query plans only; validate structured recommendations and allowlisted operation references.
- [ ] Implement speed, memory, balanced, cost-aware, and quality-first candidate scoring with hard correctness constraints.
- [ ] Test provider absence, malformed model output, and non-interference with deterministic profiles.

## Phase 6 — Human approval, bounded execution, and rollback

- [ ] Add explicit profile review, proposal approval/edit/reject, candidate execution approval, correctness review, benchmark review, and final version selection gates.
- [ ] Execute candidates in isolated temporary state with limits and timeouts; never overwrite the approved baseline.
- [ ] Persist immutable baseline/candidate versions and decision audit events.
- [ ] Ensure failures stop progression and expose actionable errors.
- [ ] Test approval state transitions, refusal, cancellation, rollback, and resume.

## Phase 7 — Benchmarking, performance dashboard, and exportable report

- [ ] Benchmark baseline and candidate under comparable inputs/configuration, with repeated runs when practical.
- [ ] Record time, peak memory, CPU, throughput, output equivalence, cache state, and measurement caveats.
- [ ] Calculate actual percentage changes; display regressions honestly.
- [ ] Add operation timeline, bottleneck evidence, resource charts, comparison dashboard, and performance history to existing Streamlit UI.
- [ ] Export the selected final dataset and a machine-readable performance report.

## Phase 8 — Integration, regression, and reliability hardening

- [ ] Add automated tests for profiler, resource allocator, query analyzer, transformation graph, caching, optimizer, validator, benchmarker, persistence, and UI-facing service contracts.
- [ ] Cover deterministic datasets at practical sizes and record reproducible test methodology.
- [ ] Run full existing-feature regression and SQL/Python equivalence tests.
- [ ] Validate bounded memory/concurrency, timeouts, cache invalidation, schema migration, and restart/resume.
- [ ] Update user documentation and completion checklist with actual results only.

## Phase gates and reporting

At the end of each phase, report:

1. Features implemented.
2. Files created.
3. Files modified.
4. Dependencies installed or changed.
5. Tests executed and results.
6. Actual measured performance results, if the phase includes benchmarking.
7. Remaining limitations and next phase.

Do not mark later-phase items complete based on a design, recommendation, smoke test, or unmeasured expectation.
