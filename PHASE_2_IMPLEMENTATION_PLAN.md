# Phase 2 SQL Optimization Implementation Plan

The SQL-focused Phase 2.1 scope is implemented. Python transformation and join optimization remain excluded.

## Phase 2.0 — Repository inspection (complete)

- [x] Inspect SQL generator and SQL/Python execution engine.
- [x] Inspect and preserve the existing performance profiler.
- [x] Inspect current LLM provider/model behavior.
- [x] Inspect workflow-event/snapshot persistence and existing frontend integration points.
- [x] Inspect current tests and dependencies.
- [x] Document reusable components, known correctness/security risks, and phased boundaries.
- [x] Make no runtime behavior or dependency changes.

## Phase 2.1 — SQL analysis and deterministic optimization (complete)

### Phase 2.1.0A — SQL execution reliability (complete)

- [x] Add and run a regression test that reproduces a successful SQL step followed by malformed SQL returning a stale previous result.
- [x] Raise a structured execution failure with execution ID, step ID, operation, category, timestamp, redacted SQL context, and partial audit.
- [x] Mark SQL operation statuses explicitly; stop on failure and prevent stale outputs from being treated as completed runs.
- [x] Record and display failed execution events in the existing Streamlit workflow.
- [x] Verify completed steps remain in audit and dependent later steps do not execute.

### 2.1a Safe SQL metadata and execution foundation (complete)

- [x] Define typed metadata for each allowlisted operation/query, its IDs, tables, columns, schema, and dependencies.
- [x] Reject unknown operations, columns, forward/self dependencies, and raw SQL fragments explicitly.
- [x] Generate deterministic unique internal table identifiers independent of user step names.
- [x] Build collision-resistant position-based column mappings and restore user-visible names.
- [x] Infer SQLite affinities from complete source Series; use BLOB affinity for ambiguous mixed values.
- [x] Propagate SQL execution errors; do not return a previous intermediate result as a success-shaped final output.
- [x] Add tests for identifier collisions, invalid operations/columns, nulls, numeric/mixed values, and output schema.

### 2.1b Query-plan analysis (complete for the current operation allowlist)

- [x] Add SQLite `EXPLAIN QUERY PLAN` collection for generated statements.
- [x] Return structured SQLite plan rows tied to operation IDs and generated SQL.
- [x] Keep plan output separate from measured runtime and describe its evidence boundary.
- [x] Treat plan rows as contextual evidence, not automatic bottlenecks.
- [x] Add deterministic in-memory query-plan tests.

### 2.1c Deterministic recommendations and candidate construction (complete for a narrow candidate class)

- [x] Analyze only generated SQL from the allowlisted structured plan; no SQL parser or arbitrary SQL is accepted.
- [x] Provide deterministic conservative index guidance; do not automatically create indexes.
- [x] Support only removal of the generated no-op `validate_dataset` operation; the Qwen3 advisor cannot change executable SQL.
- [x] Validate candidates through the same plan allowlist, schema, and identifier checks before execution.
- [x] Attach candidate ID, rationale, approval state, and benchmark validation results.

### 2.1d Candidate equivalence and benchmark (complete)

- [x] Run baseline and candidate repeatedly against equivalent in-memory source data.
- [x] Compare output column names/order, Pandas dtypes, shape, duplicate multiplicity, nulls, and values.
- [x] Reject detected mismatches; do not rely only on row counts.
- [x] Reuse `performance_profiler.profile_callable`.
- [x] Require three or more runs and report small timing changes as inconclusive.
- [x] Report only actual measured changes; no workload speedup is claimed by this implementation.

### 2.1e Existing workflow integration (complete)

- [x] Persist analysis, recommendations, advisor output, decisions, validation, and benchmark data using existing events/snapshots.
- [x] Add SQL Optimization Studio analysis, candidate review, approval/rejection, and benchmark controls.
- [x] Preserve the approved plan unless a user explicitly selects an equivalent candidate.
- [x] Update README, audit, architecture, and completion report after tests.

## Later subphases (not part of 2.1)

### Phase 2.2 — Python transformation optimization

- [ ] Analyze operation metadata/AST for safe redundancy and Pandas improvement proposals.
- [ ] Validate semantics before proposing any pruning/fusion.

### Phase 2.3 — Join optimization

- [ ] Add structured join support and analyze cardinality, null keys, duplicates, types, and query plans.
- [ ] Preserve Pandas-versus-SQL NULL semantics and join multiplicity.

### Phase 2.4 — Validation and benchmarking hardening

- [ ] Expand output equivalence rules, repeated benchmark tests, and persisted version history.

### Phase 2.5 — Broader Qwen3 advisor integration (not implemented)

- [ ] Reconcile the requested Hugging Face model requirement with the actual Ollama-only existing service.
- [ ] Keep deterministic engine results authoritative and accept only schema-validated proposals.
- [ ] Do not add a Transformers model loader without a separate scoped implementation and dependency review.

### Phase 2.6 — Broader optimization studio (not implemented)

- [ ] Add SQL, transformation, join, benchmarking, and history panels as each backend phase is validated.

### Phase 2.7 — Testing and hardening

- [ ] Add focused tests for each module, full feature regression, and reliability/security checks.

## Verification

- `python -m unittest discover -s tests -v`: 18 tests pass.
- `python -m compileall -q app.py src tests`: pass.
- No dependencies added; no general performance improvement claimed.
- See `SQL_OPTIMIZATION_ARCHITECTURE.md` and `PHASE_2_COMPLETION_REPORT.md` for limits and details.
