# AI-Powered Autonomous ETL Pipeline Generator

This project is a local-first ETL workflow builder that allows users to upload datasets, describe required transformations in natural language, review a generated execution plan, approve or reject each stage, and execute the approved cleaning pipeline with transparent validation.

## Features

- Multi-format data upload support: CSV, TSV, TXT, JSON, JSONL, XML, XLS/XLSX, Parquet, and SQLite files.
- Dataset preview and schema analysis before processing.
- Natural-language ETL instruction input.
- Dataset-aware natural-language pipeline generation using the configured Ollama or Hugging Face model, with an explicit deterministic heuristic fallback.
- Bounded schema context containing inferred types, null/duplicate/unique counts, basic numeric statistics, and at most three short samples per column; full datasets are never included in the model prompt.
- Validated operation allowlist, dependency graph, Python/SQLite previews, per-operation column editing and approval, pipeline-level approval, execution audit, and final output checks.
- ETL plan generation with Python code and generated SQL previews and risk classification.
- Human-in-the-loop approval gates before execution.
- Execution in either Python or SQL mode, with SQLite-backed SQL steps for validation workflow.
- Allowlisted SQL operations with collision-resistant internal identifiers, source-aware SQLite affinities, bound inserts, and restored source column labels.
- SQLite `EXPLAIN QUERY PLAN` evidence and conservative deterministic recommendations; indexes are never created automatically.
- Human-reviewed SQL candidate generation, duplicate-aware output-equivalence checks, repeated benchmarks, and persisted decisions/results.
- Optional local Ollama/Qwen3 query-plan advice with strict response validation; advice cannot generate or change executable SQL.
- Persistent workflow history for approvals, rejections, execution events, and plan generation.
- Data quality validation after each transformation step.
- Before-versus-after dashboard for comparison and review.
- ETL performance intelligence with extraction/schema and operation-level timing, CPU time, sampled process RSS, Python allocation estimates, throughput, and evidence-based bottleneck indicators.

## Run locally

1. Create a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the app:

```bash
streamlit run app.py
```

4. Upload datasets, select one or more in the AI ETL Pipeline Generator, add an instruction, review the generated operations/code, and approve every operation or the complete pipeline before execution.

## Natural-language pipeline generation

The planner uses the provider and model selected in **AI Model Configuration**. It profiles the selected uploaded datasets locally and sends only bounded schema/profile context to the configured model. Uploaded values are treated as untrusted data. The structured response is validated against real dataset/column names and an operation allowlist before code previews are built. Model-produced code or SQL is never executed.

The current deterministic executor supports duplicate removal, missing-value handling (numeric median / categorical mode), date parsing, text whitespace cleanup, column-name normalization, and validation-only steps. Python operations execute through the existing DataFrame engine; SQL mode uses the existing generated SQLite statements and rejects raw model SQL. Selected datasets are processed independently; cross-dataset joins, aggregations, filters, arbitrary derivations, and model-generated category mappings are not currently supported.

Ambiguous categorical corrections require an explicit mapping or user clarification; the planner will not infer business equivalence from category labels. Unsupported or ambiguous requests are shown as clarification items and block execution rather than silently claiming completion. The user can edit target columns, approve operations individually or approve the complete pipeline, reject it, and inspect per-dataset row counts, audit records, and validation issues. Original uploaded DataFrames are never mutated.

If real model inference is unavailable or its structured response is invalid, a deterministic heuristic may generate a limited plan from the instruction and actual profiled schema. The interface labels this as **heuristic fallback**, includes the reason, and does not report it as successful LLM plan generation. Hugging Face inference is local-cache-only; model weights must already be present in the configured cache/path, so planning does not silently download large models.

## Performance measurements

The Performance Intelligence tab reports measurements collected during extraction, schema profiling, and ETL operations. RSS peak is sampled at 20 ms intervals and includes the Streamlit process; Python allocation peaks do not fully include native Pandas/NumPy buffers. System disk and network counters are snapshots, not process-attributed measurements. GPU utilization is currently reported as unavailable. Bottleneck indicators rank measured operation durations and do not claim that a pipeline has been optimized.

## SQL execution failures

SQLite pipeline failures stop execution and raise a structured error containing the execution and failed-step identifiers, operation, error category, timestamp, redacted SQL context, and completed/failed step audit. The UI records a failed workflow event and does not mark a partial execution as complete or show a previous intermediate result as the final output.

## SQL Optimization Studio

Choose SQL execution mode and generate a plan to open the SQL Optimization Studio. The analyzer reports SQLite `EXPLAIN QUERY PLAN` rows separately from measured timings. Its index guidance is advisory and does not create indexes. The only built-in candidate currently removes a generated `validate_dataset` no-op projection. A candidate must be explicitly approved for a three-run benchmark, and output labels/order, dtypes, row values, nulls, and duplicate multiplicities are checked before timing results are considered. A candidate can only become the active plan after equivalence succeeds and the user explicitly selects it.

The optional local advisor uses the existing Ollama `qwen3:4b` endpoint. It is advisory only: only structured, bounded text tied to known operation IDs is accepted, and neither its output nor arbitrary SQL from a generated plan is executed.

SQLite type mapping uses INTEGER/REAL for typed numeric Series, TEXT for string and datetime-like Series, BLOB affinity for mixed/ambiguous values, and parameterized inserts for source data. Numeric-looking strings remain strings. Unsupported Python objects fail explicitly during SQLite loading. SQL-specific behavior can differ from Pandas behavior for operations such as date conversion and text cleaning; Python and SQL execution modes are not claimed to be interchangeable.

## SQL optimization limits

This implementation intentionally does not optimize Python transformations or joins, does not parse or execute arbitrary SQL, does not auto-create indexes, and does not claim general performance improvements. `EXPLAIN QUERY PLAN` is structural evidence, while benchmark results include in-process profiler overhead and can be inconclusive for short runs.

## Local model support

The app attempts to use an Ollama-hosted model when available. If no local model is running, it falls back to a deterministic rule-based planner. This keeps the project functional without requiring a paid API or external inference service.

## Notes

- The pipeline is intentionally designed to run safe, explainable operations on a copy of the data.
- SQL and Python generation are exposed as structured execution steps before approval and execution.
- Unsafe or unsupported imports are shown as explicit errors instead of silently corrupting data.
