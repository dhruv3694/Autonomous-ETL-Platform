# SQL Optimization Architecture

## Scope and trust boundaries

The SQL path accepts only the operation allowlist in `src/etl/sql_metadata.py`. Plan text is not SQL: user/model-provided `sql` or `query` fields are rejected, and SQL is generated from validated operation metadata. This work covers SQLite ETL only; Python transformation optimization and join optimization remain out of scope.

## Metadata, schema, and execution

- Each source field receives a position-based internal name with a digest. The original label is restored by position after reading SQL results; duplicate labels are preserved in output, but a plan cannot unambiguously target a duplicated label.
- Source values are inserted with bound parameters. Internal identifiers are quoted, and intermediate table IDs are deterministic from operation ID, operation type, and ordinal—not from user-facing step names.
- Numeric Series map to SQLite INTEGER/REAL, strings and datetime-like Series to TEXT, and mixed/ambiguous Series to BLOB affinity to avoid affinity-driven text/number coercion. Numeric-looking strings are not coerced.
- SQL failures raise `SQLExecutionError`, preserve completed/failed audit entries, stop later operations, and cannot return a stale result as a completed run.
- SQLite `MEDIAN` and `MODE` aggregates support the current `fill_missing` SQL operation. SQLite and Pandas semantics are not guaranteed identical for every operation or dtype.

## Query-plan analysis

The analyzer loads the same source schema into an in-memory SQLite connection, obtains `EXPLAIN QUERY PLAN` rows for each generated query, then executes each operation to create the next input table. Reports retain operation IDs, SQL, scan/search classification, temporary-B-tree flags, and predicate/group/order metadata. Explain output is not treated as elapsed-time evidence. The current allowlist has no selective predicate, explicit group, or order key, so deterministic index guidance reports that no index is justified; indexes are never created automatically.

## Candidate generation and correctness gate

Candidate generation currently supports one safe class: removing `validate_dataset` steps, which compile to a row-preserving projection. No SQL rewrite is accepted from an LLM. Before timing is used, baseline and candidate outputs are compared for:

- column names and order;
- Pandas dtypes;
- row/column shape;
- an order-independent row multiset including scalar types, nulls, and duplicate counts.

Any detected mismatch rejects the candidate. A user must approve a candidate for benchmarking and separately select an equivalent candidate before it replaces the active plan. The previously approved plan remains unchanged otherwise.

## Benchmarking

Baseline and candidate SQL plans run repeatedly against copies of the same in-memory source data. `profile_callable` supplies elapsed-time measurements, and run order alternates. At least three runs are required; median times and all samples are shown. A result is labeled measured improvement/regression only when the median difference exceeds both five percent of the baseline median and twice the larger median absolute deviation; otherwise it is inconclusive. The benchmark includes in-process profiling overhead and does not imply general workload performance.

## Optional local advisor and persistence

The existing Ollama integration (`qwen3:4b`) may explain the captured plan using a constrained JSON response. Output is bounded and validated against operation IDs. It is display-only and cannot alter SQL, candidates, correctness results, or benchmark evidence. Analysis, candidate decisions, advisor results, and benchmark outcomes are recorded through the existing `workflow_events` system; snapshots can retain SQL Optimization Studio state.

## Known limitations

- The SQL operation allowlist is intentionally narrow. `EXPLAIN QUERY PLAN` for the current simple projections is not a general optimizer or cost model.
- No arbitrary SQL parser, join support, index creation, SQL/Python equivalence guarantee, or Python transformation optimization is included.
- Datetime conversion, text cleaning, missing-value behavior, and unusual object dtypes may not exactly reproduce Pandas semantics.
- In-memory SQLite runs may be too short for reliable timing; small differences are reported as inconclusive.
