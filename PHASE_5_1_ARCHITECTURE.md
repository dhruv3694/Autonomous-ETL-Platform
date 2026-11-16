# Phase 5.1 Technical Architecture: Hardening & Reliability of the Autonomous ETL Optimization Engine

## 1. Overview
Phase 5.1 hardens and expands the Autonomous AI-Powered ETL Optimization Engine, integrating robust Python AST security analysis, scoped runtime execution, multi-statement SQL parsing with adversarial inspection, three-state output equivalence verification, and statistical reliability in performance benchmarking.

---

## 2. Optimization Architecture & Input Modes

The Optimization Studio supports three distinct operational modes:
- **Mode A (Active / AI-Generated Pipelines)**: Direct one-click structural analysis, candidate generation, and 3-run alternating benchmarking on the active pipeline.
- **Mode B (User-Provided Pipelines)**: Input fields for raw Python or SQL transformations with strict AST / regex security validation before analysis.
- **Mode C (Existing / Historical Pipelines)**: Select previously executed or saved workflow snapshots directly from SQLite history (`workflow_history.db`) and load them for optimization.

---

## 3. Security Boundaries

### Python Execution Security
- **AST Structural Check**: Prohibits `Import`, `ImportFrom`, `Global`, `Nonlocal`, `With`, `AsyncWith`, `ClassDef`, `FunctionDef`, `AsyncFunctionDef`, `Lambda`, and `Try` nodes.
- **Forbidden Calls**: Rejects `eval`, `exec`, `compile`, `__import__`, `open`, `read`, `write`, `system`, `popen`, `spawn`, `subprocess`, `socket`, `requests`, `urllib`, `remove`, `rmdir`, `unlink`, `rename`, `shutil`, `os`, `sys`, `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `dir`.
- **Reflection Protection**: Detects and rejects access to attributes such as `__class__`, `__bases__`, `__subclasses__`, `__mro__`, `__globals__`, `__code__`, `__closure__`, `__builtins__`, `__import__`, `__dict__`.
- **Restricted Execution Scope**: Validated code is compiled and executed in a dedicated local namespace containing only allowlisted builtins (`pd`, `np`, `re`, standard math/type primitives) on an isolated copy of the DataFrame (`df`). Execution time is capped by a strict 2.0s timeout.

### SQL Execution Security
- **Comment & String Literal Stripping**: Pre-processes queries to strip block comments (`/* ... */`), line comments (`-- ...`), and single-quoted string literals. Keywords appearing inside string literals or comments are never falsely flagged.
- **Multi-statement Denial**: Rejects queries containing semicolons separating multiple statements.
- **Disallowed Operations**: Blocks administrative and destructive queries (`ATTACH`, `DETACH`, `PRAGMA`, `LOAD_EXTENSION`, `VACUUM`, `ALTER`, `DROP TABLE`, `DROP DATABASE`, `DROP VIEW`, `CREATE TRIGGER`, `GRANT`, `REVOKE`).
- **Read-Only Invariant**: Guarantees that query optimization analysis and EXPLAIN plans execute on isolated SQLite `:memory:` connections without mutating the original user database.

---

## 4. Three-State Equivalence Contract

`validate_pipeline_equivalence()` returns one of three discrete outcomes:
1. `EQUIVALENT`: Complete verification of column names, dtypes, NULL semantics, row count, duplicate multiplicity, and numeric values within configured tolerances (`1e-6`).
2. `NOT EQUIVALENT`: Confirmed discrepancy violating the equivalence contract.
3. `INCONCLUSIVE`: Returned when datasets exceed configured row thresholds (e.g. `max_compare_rows = 500,000`), or when sampling is insufficient to prove full equivalence without risk of memory exhaustion.

An `INCONCLUSIVE` result blocks automatic application and is never treated as a successful optimization.

---

## 5. Candidate Generation & Safety Rules

Every generated candidate includes explicit preconditions and safety check verifications:
- **Pruning No-Op Validations**: Verified that the candidate step is an identity projection with no custom transformation parameters or mutation.
- **Early Deduplication**: Verified that operations preceding deduplication contain no aggregations, groupings, or row-order dependencies.
- **Numeric Float Downcasting**: Checks minimum/maximum ranges and verifies that round-trip casting (`float64` to `float32`) retains precision within `1e-4` on real column data before proposing downcasting.

---

## 6. Benchmarking Methodology

`benchmark_pipeline_candidate()` executes an alternating benchmark protocol:
- **Warm-Up Phase**: Configurable unmeasured warm-up iterations to stabilize JIT/cache states.
- **Alternating Execution**: Alternates execution order between baseline and candidate to avoid cache bias.
- **Statistical Metrics**: Reports baseline and candidate median, mean, standard deviation, and median absolute deviation (MAD).
- **Noise Control**: Employs a noise floor threshold:
  $$\text{Noise Floor} = \max(0.05 \times \text{median}, 2 \times \text{MAD}, 0.001\text{s})$$
- **Classification**:
  - `PERFORMANCE_IMPROVED`
  - `EQUIVALENT_BUT_NO_MEASURABLE_IMPROVEMENT`
  - `PERFORMANCE_REGRESSION`
  - `REJECTED_DUE_TO_INCORRECT_OUTPUT`
  - `INCONCLUSIVE`
- **Overhead Accounting**: Separates preparation and equivalence validation overhead from raw execution timings.

---

## 7. Running Tests

Execute the full suite using pytest:
```bash
pytest -q
```
All 57 unit and integration tests execute in ~10 seconds.
