# ETL Performance Intelligence: Phase 0 Repository Audit

## Scope and method

This audit covers the current application and ETL package in this repository. It is a read-only assessment of the existing implementation for the requested performance-optimization extension. No application code, dependencies, or existing workflow data were changed during Phase 0.

## Repository baseline

| Area | Observed implementation |
|---|---|
| Application | Single Streamlit entry point in `app.py`; upload, dataset selection, ETL plan review, execution, export, and history UI are in one module. |
| Data ingestion | `src/etl/data_loader.py` reads supported files into Pandas DataFrames in memory. It imports all discovered SQLite tables and Excel sheets as separate dataset entries. No chunked or streaming path is present. |
| ETL planning | `src/etl/llm_service.py` makes an HTTP request to a local Ollama endpoint, with Qwen3 model names among its candidates. The request currently selects the first candidate; it does not detect available hardware or models. Any exception falls back to heuristics. |
| LLM integration discrepancy | Despite the project brief describing Hugging Face Qwen3, the checked-in planner is Ollama-only. `transformers` is not declared in `requirements.txt`; the inspected Python environment has Transformers installed but PyTorch is not available. There is no verified Hugging Face inference path in the repository. |
| Execution | `src/etl/execution_engine.py` executes a fixed set of Python operations or SQL emitted for an in-memory SQLite database. It does not execute LLM-produced Python source. There are no operation-level timings, resource samples, timeout controls, or configurable concurrency limits. |
| SQL | `src/etl/sql_generator.py` builds SQL strings for known operations. The executor creates SQLite tables with text columns, inserts the full DataFrame, then reads each step result into Pandas. No query-plan inspection, EXPLAIN integration, SQL benchmark, index analysis, or optimized-vs-original query comparison exists. |
| Validation | `validate_dataframe` checks row-count change, duplicate rows, and empty schema. It does not compare output equivalence, validate operation-specific expectations, or quantify data-quality deltas as correctness constraints. |
| Workflow persistence | `src/etl/workflow_state.py` persists event and snapshot JSON in `workflow_history.db`. A snapshot can retain a serialized last run, but it does not retain or identify the uploaded source bytes through a content fingerprint. The UI restore path restores plan/run fields, not the source dataset itself. |
| Agents/orchestration | No project agent modules or orchestration graph were found. `langgraph` is present in the inspected environment, but is not declared or imported by the project. The current workflow is a direct Streamlit-to-service call path. |
| Optimization features | No profiling, bottleneck diagnosis, transformation DAG/pruning, join analysis, resource allocation, caching/incremental execution, benchmark engine, Pareto analysis, or optimization approval/version workflow was found. |
| Tests | No test files or test configuration were found in the project. |
| Dependencies | `requirements.txt` includes Pandas, DuckDB, SQLAlchemy, Plotly, Streamlit, and file-format libraries, among others. `psutil` is available in the inspected environment but not declared. Polars and PyTorch were unavailable in that environment. |
| Database | Existing SQLite database contains `workflow_events` and `workflow_snapshots` (and SQLite's `sqlite_sequence`). Do not reset or replace it when adding optimization persistence. |

## Reusable functionality

- Keep the existing Streamlit workflow and approval UI; add performance intelligence as a section rather than a replacement application.
- Reuse `summarize_dataframe`, the operation registry embodied by the executor's explicit operation branches, and existing workflow event/snapshot persistence.
- Keep original and transformed DataFrames separate. Use the existing source DataFrame as the correctness reference for optimization validation.
- Reuse SQLite for local persistence and local SQL explain/benchmark operations. DuckDB is already in the dependency manifest and can be evaluated for columnar/out-of-core work with evidence from benchmarks.
- Reuse the local model approach only behind a provider abstraction. Preserve the existing Ollama behavior while adding the requested Transformers/Hugging Face path in a later phase.

## Baseline risks and constraints for later phases

1. Do not equate an LLM proposal or a generated SQL plan with a verified optimization. Recommendations must be evidence-backed, and execution must remain deterministic and allowlisted.
2. The current SQL route changes data representation by inserting values into SQLite `TEXT` columns and converting results back to Pandas. SQL-vs-Python comparisons must account for dtype and null semantics and reject non-equivalent results.
3. Current plan steps do not consistently define stable input/output schemas or dependencies. A dependency graph must be based on explicit operation metadata and conservative rules; visually similar operations cannot be pruned without equivalence proof.
4. Full-DataFrame copies and per-step validation can materially affect measured time and peak memory. Baseline and optimized measurements must use equivalent inputs, instrumentation overhead disclosures, and reproducible repeated runs.
5. Uploaded source data is not rehydrated by snapshots. Resume and cache reuse require a durable, verifiable source reference or a carefully bounded persisted dataset artifact; filename alone is insufficient.
6. The current workflow permits approval of a plan but does not support per-optimization approval, validation gating, rollback/version selection, or execution resource limits.
7. Broad exception fallbacks in the LLM service obscure model availability and response errors. Later integration should report provider status explicitly without allowing model failure to block deterministic profiling.

## Environment observations

At audit time, the environment reported `psutil`, `duckdb`, `transformers`, and `langgraph` as importable; `polars` and `torch` were not importable. These are environment observations, not repository guarantees. Only dependencies committed in the project manifest should be treated as supported.

## Phase 0 completion checklist

- [x] Inspect the existing application, ETL modules, dependency manifest, README, and local workflow database.
- [x] Identify the current model integration and compare it with the requested Hugging Face/Qwen3 requirement.
- [x] Check for existing profiling, agent architecture, optimization code, and tests.
- [x] Document reusable components and key correctness/reliability constraints.
- [x] Produce an integration architecture and phased implementation plan.
- [x] Make no application-code or dependency changes in Phase 0.

## Phase 1 implementation note

Phase 1 profiling has since been added in `src/etl/performance_profiler.py` and wired into extraction, schema profiling, Python transformation steps, and the SQLite source-load/SQL steps. The Streamlit UI reports measured operation durations and evidence-based duration-share bottlenecks. Metrics are retained in existing workflow event/snapshot payloads. RSS peak is sampled; Python allocation tracking may omit native buffers; disk/network snapshots are system-wide; GPU utilization is unavailable. See the Phase 1 checklist in `PERFORMANCE_IMPLEMENTATION_PLAN.md` for implemented scope and remaining regression work.

## Phase 2.1.0A implementation note

The SQL executor now propagates failed SQL steps through a structured `SQLExecutionError` with partial audit history instead of returning the prior intermediate result as a successful output. The Streamlit caller records a failed event and does not display a completed run. This reliability change does not yet solve table/column identifier collisions or SQLite type inference.
