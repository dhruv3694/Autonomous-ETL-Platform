# Phase 2 Optimization Audit

## Scope

This is the Phase 2.0 inspection of the current ETL platform for the SQL, transformation, and join optimization work. No runtime behavior or dependencies were changed while preparing this audit.

## Existing implementation

| Area | Current state |
|---|---|
| Streamlit workflow | `app.py` provides upload and selection, plan generation/approval, Python or SQLite execution, dataset export, persistent snapshots/history, and an ETL Performance Intelligence view. |
| SQL generation | `src/etl/sql_generator.py` generates statements from validated structured operation metadata, deterministic internal table IDs, and collision-resistant source-column mappings. |
| SQL execution | `src/etl/execution_engine.py` loads schema-aware source columns with bound parameters, restores original labels by position, registers SQLite median/mode aggregates, and raises structured failures instead of returning stale results. |
| SQL plan analysis | `src/etl/sql_optimizer.py` captures per-operation SQLite `EXPLAIN QUERY PLAN` rows. These are structural evidence and are displayed separately from profiler measurements. |
| SQL optimization | The deterministic advisor reports conservative index guidance and proposes only removing generated no-op `validate_dataset` projections. Candidates are checked for schema/type/row-multiset equivalence and repeatedly benchmarked. |
| Python transformations | The executor applies a fixed set of explicit Pandas operations and profiles each step. No AST-based or metadata-based transformation optimizer exists. |
| Joins | No explicit join operation exists in the current ETL operation registry or planner/executor path. Join optimization is out of the Phase 2.1 scope. |
| LLM | `src/etl/llm_service.py` retains the Ollama planner and offers optional constrained Qwen3 plan advice. Advice is validated against operation IDs and cannot alter SQL or candidate generation. No Hugging Face Transformers model is loaded. |
| Performance profiler | `src/etl/performance_profiler.py` already provides callable timing, process CPU/RSS and Python allocation metrics, DataFrame size/throughput, system snapshots, and elapsed-time-share bottleneck detection. Reuse this; do not introduce a second profiler. |
| Persistence | `src/etl/workflow_state.py` persists `workflow_events` and `workflow_snapshots` as SQLite JSON fields. Optimization history should use these existing mechanisms or additive schema migration, not replace the database. |
| Tests | The profiler, SQL failure, metadata, query-plan, candidate-equivalence, and benchmark tests are in `tests/`; see the Phase 2 completion report for the latest suite result. |
| Dependencies | `requirements.txt` includes Pandas and `psutil`; no SQL parser dependency is declared. `sqlglot` availability has not been established as a project guarantee. |

## Original SQL path: verified safety and correctness concerns (addressed in Phase 2.1)

1. Generated output table identifiers are derived from plan-provided step names. This is not a trusted internal identifier strategy.
2. `sanitize_identifier` is not injective: distinct source labels can map to the same SQL identifier. The SQL loader does not check for collisions before creating the table.
3. Source fields are declared `TEXT` regardless of DataFrame dtype. Numeric SQL semantics and aggregations can differ from Pandas semantics; query results may also return values with different types.
4. The SQL generator references sanitized columns while the original schema can contain names that collide after sanitization.
5. Unknown operations fall through to a copy-shaped `SELECT *`, which can look successful instead of failing clearly.
6. SQL is assembled using string interpolation. Source values are bound during insertion, but generated identifiers and table names need strict internal generation and quoting.
7. SQL execution creates intermediate tables but query analysis does not preserve structured operation/query IDs, input/output table metadata, or plan rows.
8. SQL execution catches step exceptions and breaks, then returns the last table without raising a failure to the caller. This is a risk for later candidate validation and must be addressed before treating candidate results as successful.
9. `EXPLAIN QUERY PLAN` evidence must be kept distinct from runtime measurements. Scans are not inherently inefficient, especially for small in-memory tables.
10. The executor does not model joins, user filters, or arbitrary query text today; recommendations and parsing must remain limited to the supported structured operations.

## Reusable components

- `performance_profiler.profile_callable` for measuring baseline and candidate SQL execution; benchmark code should call it rather than duplicate instrumentation.
- `execution_engine._execute_sql_plan` and the in-memory SQLite setup as the execution integration point, subject to strict failure propagation and safe operation metadata.
- `sql_generator.generate_sql_steps` as the current supported-operation inventory, while replacing unsafe internal naming with a structured validated representation.
- `workflow_state.record_event`, `save_snapshot`, and existing event history for proposal, approval, validation, and benchmark audit data.
- Existing Streamlit tabs/expander conventions in `app.py` for a later SQL Optimization UI.
- Existing local Ollama service only as an optional advisor integration if configured/identified as Qwen3; deterministic SQL analysis must remain independent.

## Phase 2.1 implementation boundary and current status

The SQL-only work completed the following:

1. Add structured SQL operation/query metadata and strict allowlist validation. Generate deterministic internal table identifiers and collision-resistant internal column mappings while preserving source/output names.
2. Address SQLite type and NULL semantics with schema-aware bindings/inference. Preserve ambiguous mixed columns conservatively and document unsupported conversions. Do not silently cast results.
3. Add SQLite `EXPLAIN QUERY PLAN` capture and structured interpretation (scan/search/index/temp B-tree) with cautious, evidence-backed findings.
4. Add deterministic recommendations and only narrowly provable SQL candidates. Do not auto-create indexes or execute arbitrary LLM-provided SQL.
5. Compare baseline and candidate output with schema/order/row-multiplicity/value/null checks. Reject mismatches and surface execution errors.
6. Benchmark through the existing profiler on identical in-memory data/state; repeat runs where practical and report measurement uncertainty/regressions. Keep setup/index creation costs separate.
7. Integrate the analysis and candidate review into the existing approval workflow only after backend tests establish safe behavior.

Python transformation optimization, joins, and Hugging Face model loading remain out of scope. The implemented Streamlit SQL Optimization Studio is limited to plan analysis, conservative candidates, correctness review, and benchmarking.

## Baseline verification (before Phase 2.1)

- The six existing `unittest` cases pass against the inspected code.
- Existing tests include a representative SQL duplicate-removal execution and verify profiling metadata remains attached.
- These baseline results predate the implementation below and are retained as historical audit context.

## Phase 2.1.0A execution reliability update

- The stale-result path was reproduced: after a successful SQL step, a malformed later statement was swallowed, the loop stopped, and the previous table was returned as the plan result.
- SQL failures now raise a structured `SQLExecutionError` carrying execution ID, failed step ID, operation type, error category/message, timestamp, redacted SQL context, and partial execution audit.
- Completed steps remain in the exception audit; the failed step is marked `execution_status=failed`; later dependent steps do not run.
- The Streamlit caller clears any previous current-run result before execution and records/displays `execution_failed` rather than reporting completion or exposing a stale output.
- Successful SQL steps and source loading now carry explicit `execution_status=succeeded`.
- Subsequent Phase 2.1 work adds safe identifier mapping, metadata validation, SQLite affinity inference, explain-plan analysis, narrowly scoped candidates, equivalence checks, repeated benchmarking, and the Streamlit SQL Optimization Studio. See `SQL_OPTIMIZATION_ARCHITECTURE.md` for current behavior and limitations.
