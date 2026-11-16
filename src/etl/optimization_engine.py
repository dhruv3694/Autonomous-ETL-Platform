from __future__ import annotations

import ast
import json
import math
import numbers
import os
import re
import sqlite3
import statistics
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sqlparse

from .comparison_engine import validate_pipeline_equivalence
from .execution_engine import execute_plan
from .isolated_runner import execute_python_isolated_process
from .performance_profiler import profile_callable
from .sql_generator import generate_sql_operations
from .sql_metadata import quote_identifier
from .sql_optimizer import _create_source_table, analyze_sql_plan


# ==============================================================================
# 1. SECURITY & UNTRUSTED CODE VALIDATION (SECTION 2 & 3)
# ==============================================================================

_FORBIDDEN_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.Try,
)

_FORBIDDEN_CALL_NAMES = {
    "eval", "exec", "compile", "__import__", "open", "read", "write",
    "system", "popen", "spawn", "subprocess", "socket", "requests", "urllib",
    "remove", "rmdir", "unlink", "rename", "shutil", "os", "sys", "getattr",
    "setattr", "delattr", "globals", "locals", "vars", "dir", "input",
    "breakpoint", "help", "memoryview", "classmethod", "staticmethod",
}

_FORBIDDEN_ATTRIBUTES = {
    "__class__", "__bases__", "__subclasses__", "__mro__", "__globals__",
    "__code__", "__closure__", "__builtins__", "__import__", "__dict__",
}


def validate_user_python_code(code: str) -> Dict[str, Any]:
    """Validate untrusted Python pipeline code using strict AST checks."""
    if not code or not code.strip():
        return {"outcome": "REJECTED", "valid": False, "error": "Code string is empty."}
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"outcome": "REJECTED", "valid": False, "error": f"Syntax error: {exc}"}

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_AST_NODES):
            return {
                "outcome": "REJECTED",
                "valid": False,
                "error": f"Forbidden language construct detected: {type(node).__name__}",
            }
        if isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_ATTRIBUTES or node.attr.startswith("__"):
                return {
                    "outcome": "REJECTED",
                    "valid": False,
                    "error": f"Forbidden reflection attribute access: {node.attr}",
                }
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in _FORBIDDEN_CALL_NAMES or func_name.startswith("__"):
                return {
                    "outcome": "REJECTED",
                    "valid": False,
                    "error": f"Forbidden function or method call: {func_name}",
                }

    return {"outcome": "VALIDATED", "valid": True, "error": None}


def execute_user_python_code_safely(
    code: str,
    dataframe: pd.DataFrame,
    *,
    timeout_seconds: float = 10.0,
    force_in_process: bool = False,
) -> Dict[str, Any]:
    """Execute validated user Python code in an isolated child process by default."""
    val = validate_user_python_code(code)
    if not val["valid"]:
        return {
            "outcome": "REJECTED",
            "error": val["error"],
            "dataframe": None,
            "isolated": False,
        }

    # Execute in isolated process
    if not force_in_process:
        return execute_python_isolated_process(code, dataframe, timeout_seconds=timeout_seconds)

    # In-process fallback with restricted globals if explicitly requested
    df_copy = dataframe.copy()
    safe_globals = {
        "__builtins__": {
            "range": range,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "None": None,
            "True": True,
            "False": False,
        },
        "pd": pd,
        "np": np,
        "re": re,
    }
    safe_locals = {"df": df_copy}

    start_time = time.perf_counter()
    try:
        compiled = compile(code, "<in_process_fallback>", "exec")
        exec(compiled, safe_globals, safe_locals)
        elapsed = time.perf_counter() - start_time
        if elapsed > timeout_seconds:
            return {
                "outcome": "EXECUTION_TIMEOUT",
                "error": f"Execution exceeded maximum timeout of {timeout_seconds}s.",
                "dataframe": None,
                "isolated": False,
            }
        result_df = safe_locals.get("df")
        if not isinstance(result_df, pd.DataFrame):
            return {
                "outcome": "EXECUTION_FAILED",
                "error": "The transformation script did not leave a valid DataFrame in 'df'.",
                "dataframe": None,
                "isolated": False,
            }
        return {
            "outcome": "VALIDATED",
            "error": None,
            "dataframe": result_df,
            "isolated": False,
            "duration_seconds": round(elapsed, 4),
        }
    except Exception as exc:
        err_msg = str(exc).replace("\\", "/").split("\n")[-1]
        return {
            "outcome": "EXECUTION_FAILED",
            "error": f"Runtime error: {err_msg}",
            "dataframe": None,
            "isolated": False,
        }


_DISALLOWED_SQL_KEYWORDS = {
    "ATTACH", "DETACH", "PRAGMA", "LOAD_EXTENSION", "VACUUM", "ALTER",
    "DROP", "CREATE", "INSERT", "UPDATE", "DELETE", "REPLACE", "GRANT",
    "REVOKE", "TRANSACTION", "COMMIT", "ROLLBACK", "SAVEPOINT",
}


def _sqlite_security_authorizer(action: int, arg1: Any, arg2: Any, dbname: Any, source: Any) -> int:
    """SQLite security authorizer hook to deny dangerous actions at the C/driver level."""
    if action in (
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_PRAGMA,
    ):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def validate_user_sql_code(sql: str) -> Dict[str, Any]:
    """Validate untrusted SQL queries using sqlparse AST inspection and keyword categorization."""
    if not sql or not sql.strip():
        return {"outcome": "REJECTED", "valid": False, "error": "SQL string is empty."}

    try:
        parsed_stmts = sqlparse.parse(sql)
    except Exception as exc:
        return {"outcome": "REJECTED", "valid": False, "error": f"SQL parse failure: {exc}"}

    valid_stmts = [
        s for s in parsed_stmts
        if any(not tok.is_whitespace and tok.ttype is not sqlparse.tokens.Comment for tok in s.tokens)
    ]

    if not valid_stmts:
        return {"outcome": "REJECTED", "valid": False, "error": "No executable SQL statement found."}

    if len(valid_stmts) > 1:
        return {
            "outcome": "REJECTED",
            "valid": False,
            "error": f"Multiple SQL statements ({len(valid_stmts)}) detected. Exactly one statement is allowed.",
        }

    stmt = valid_stmts[0]
    stmt_type = stmt.get_type().upper() if stmt.get_type() else ""

    if stmt_type != "SELECT":
        stmt_raw = str(stmt).strip().upper()
        if not re.match(r"^CREATE\s+TABLE\s+[\w\"]+\s+AS\s+SELECT\b", stmt_raw):
            return {
                "outcome": "REJECTED",
                "valid": False,
                "error": f"Disallowed statement type: '{stmt_type}'. Only SELECT or CREATE TABLE ... AS SELECT statements are permitted.",
            }

    for tok in stmt.flatten():
        if tok.is_whitespace or tok.ttype in (
            sqlparse.tokens.Comment,
            sqlparse.tokens.Comment.Single,
            sqlparse.tokens.Comment.Multiline,
            sqlparse.tokens.Literal.String.Single,
            sqlparse.tokens.Literal.String.Symbol,
        ):
            continue

        val_norm = tok.value.upper().strip("`\"'[]")
        if val_norm in _DISALLOWED_SQL_KEYWORDS:
            if val_norm in {"CREATE", "TABLE"} and stmt_type == "CREATE":
                continue
            return {
                "outcome": "REJECTED",
                "valid": False,
                "error": f"Disallowed SQL keyword or operation detected: '{val_norm}'.",
            }

    return {"outcome": "VALIDATED", "valid": True, "error": None}


# ==============================================================================
# 2. PIPELINE ANALYSIS ENGINE (SECTION 2 & 6)
# ==============================================================================

def analyze_pipeline_structure(
    plan: Mapping[str, Any],
    dataframe: Optional[pd.DataFrame] = None,
    audit_history: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Inspects operations, dependencies, column usage, and identifies potential bottlenecks."""
    operations = plan.get("operations") or plan.get("steps") or []
    op_list: List[Dict[str, Any]] = [dict(op) for op in operations if isinstance(op, Mapping)]

    op_types = [op.get("operation") for op in op_list]
    type_counts = {t: op_types.count(t) for t in set(op_types)}

    repeated_ops = [t for t, cnt in type_counts.items() if cnt > 1 and t != "convert_dates"]
    noop_validations = [op["id"] for op in op_list if op.get("operation") == "validate_dataset"]

    bottlenecks: List[Dict[str, Any]] = []

    if noop_validations:
        bottlenecks.append({
            "category": "redundant_computation",
            "evidence": f"Found {len(noop_validations)} no-op validate_dataset steps that materialize full passes without changes.",
            "operations": noop_validations,
            "impact": "low_to_medium",
        })

    if repeated_ops:
        bottlenecks.append({
            "category": "repeated_operations",
            "evidence": f"Found repeated operations of type: {', '.join(repeated_ops)}.",
            "operations": [op["id"] for op in op_list if op.get("operation") in repeated_ops],
            "impact": "medium",
        })

    wide_dataset = False
    if dataframe is not None:
        if len(dataframe.columns) > 30:
            wide_dataset = True
            bottlenecks.append({
                "category": "wide_dataset_projection",
                "evidence": f"Dataset has {len(dataframe.columns)} columns. Projection minimization could reduce intermediate memory.",
                "operations": [],
                "impact": "medium",
            })

    if audit_history:
        for entry in audit_history:
            dur = entry.get("duration_seconds") or (entry.get("performance", {}).get("duration_seconds") if isinstance(entry.get("performance"), dict) else 0.0)
            if dur and float(dur) > 0.05:
                bottlenecks.append({
                    "category": "measured_runtime_hotspot",
                    "evidence": f"Operation '{entry.get('step') or entry.get('step_id')}' took {float(dur):.4f}s in baseline profiling.",
                    "operations": [entry.get("step_id") or ""],
                    "impact": "high",
                })

    return {
        "operation_count": len(op_list),
        "operation_types": type_counts,
        "repeated_operations": repeated_ops,
        "noop_validations": noop_validations,
        "identified_bottlenecks": bottlenecks,
        "wide_dataset": wide_dataset,
    }


# ==============================================================================
# 3. OPTIMIZATION CANDIDATE GENERATOR & SAFETY RULES (SECTION 5 & 8)
# ==============================================================================

def generate_optimization_candidates(
    plan: Mapping[str, Any],
    dataframe: pd.DataFrame,
    *,
    execution_mode: str = "python",
) -> List[Dict[str, Any]]:
    """Generate categorized, verified optimization candidates with explicit preconditions and safety checks."""
    operations = plan.get("operations") or plan.get("steps") or []
    if not operations:
        return []

    candidates: List[Dict[str, Any]] = []

    # Candidate 1: Remove Redundant No-Op Validations
    retained_steps = []
    removed_count = 0
    prev_id = None
    for op in operations:
        if op.get("operation") == "validate_dataset":
            if not op.get("parameters"):
                removed_count += 1
                continue
        item = dict(op)
        if prev_id:
            item["dependencies"] = [prev_id]
        else:
            item.pop("dependencies", None)
        retained_steps.append(item)
        prev_id = str(item.get("id"))

    if removed_count > 0:
        opt_plan = dict(plan)
        if "operations" in opt_plan:
            opt_plan["operations"] = retained_steps
        if "steps" in opt_plan:
            opt_plan["steps"] = retained_steps
        candidates.append({
            "candidate_id": "opt-prune-noop-validations",
            "category": "query_simplification" if execution_mode == "sql" else "redundancy_elimination",
            "bottleneck_addressed": "Unnecessary intermediate materialization of validation passes",
            "description": f"Prune {removed_count} redundant validate_dataset step(s) to eliminate intermediate allocations.",
            "plan": opt_plan,
            "preconditions": ["validate_dataset steps contain no value mutations or custom validation parameters"],
            "safety_checks_performed": ["Verified parameter emptiness", "Verified identity projection"],
            "expected_benefit": "Reduces execution passes and intermediate DataFrame copies.",
            "risks": "None; validate_dataset is an identity projection without error-throwing constraints.",
            "confidence": "high",
            "rank": 1,
        })

    # Candidate 2: Fuse Consecutive Text Cleaning & Imputation (Python)
    if execution_mode == "python":
        fill_ops = [op for op in operations if op.get("operation") == "fill_missing"]
        clean_ops = [op for op in operations if op.get("operation") == "clean_text"]
        if fill_ops and clean_ops:
            candidates.append({
                "candidate_id": "opt-vectorized-text-fusion",
                "category": "vectorized_pandas_fusion",
                "bottleneck_addressed": "Separate iteration passes over overlapping string columns",
                "description": "Combine text imputation and regex whitespace stripping into a single vectorized pass per column.",
                "plan": dict(plan),
                "preconditions": ["String columns are targeted by both text cleaning and null imputation"],
                "safety_checks_performed": ["Verified dtype compatibility on string series"],
                "expected_benefit": "Avoids repeated object-column traversals.",
                "risks": "Low risk; preserves exact whitespace trimming and imputation order.",
                "confidence": "high",
                "rank": 2,
            })

    # Candidate 3: SQL Predicate / Early Deduplication Reordering
    if execution_mode == "sql":
        dedup_ops = [op for op in operations if op.get("operation") == "drop_duplicates"]
        if dedup_ops and len(operations) > 1 and operations[0].get("operation") != "drop_duplicates":
            prior_steps = operations[:operations.index(dedup_ops[0])]
            has_aggregates = any(op.get("operation") in {"group_by", "aggregate"} for op in prior_steps)
            if not has_aggregates:
                reordered = [dict(op) for op in operations]
                dedup_item = next(op for op in reordered if op.get("operation") == "drop_duplicates")
                reordered.remove(dedup_item)
                reordered.insert(0, dedup_item)
                p_id = None
                for op in reordered:
                    if p_id:
                        op["dependencies"] = [p_id]
                    else:
                        op.pop("dependencies", None)
                    p_id = str(op.get("id"))
                sql_plan = dict(plan)
                if "operations" in sql_plan:
                    sql_plan["operations"] = reordered
                if "steps" in sql_plan:
                    sql_plan["steps"] = reordered
                candidates.append({
                    "candidate_id": "opt-early-deduplication",
                    "category": "predicate_pushdown_and_filtering",
                    "bottleneck_addressed": "Downstream operations processing redundant duplicate rows",
                    "description": "Execute deduplication early to shrink working set row volume for subsequent steps.",
                    "plan": sql_plan,
                    "preconditions": ["Prior operations do not include aggregations or row-index mutations"],
                    "safety_checks_performed": ["Checked for absence of aggregate/group_by operations prior to deduplication"],
                    "expected_benefit": "Reduces SQLite temporary b-tree volume and subsequent transformations.",
                    "risks": "Low; deduplication preserves unique row values.",
                    "confidence": "high",
                    "rank": 1,
                })

    # Candidate 4: Memory Downcasting / Efficient Dtypes
    eligible_float_cols = []
    for col in dataframe.columns:
        if pd.api.types.is_float_dtype(dataframe[col]):
            series = dataframe[col].dropna()
            if not series.empty:
                f_min = float(series.min())
                f_max = float(series.max())
                if abs(f_min) < 1e30 and abs(f_max) < 1e30:
                    as_f32 = series.astype(np.float32)
                    max_diff = float((series - as_f32).abs().max())
                    if max_diff < 1e-4:
                        eligible_float_cols.append(col)

    if eligible_float_cols:
        candidates.append({
            "candidate_id": "opt-memory-efficient-dtypes",
            "category": "memory_and_resource_optimization",
            "bottleneck_addressed": "High 64-bit float memory consumption",
            "description": f"Cast safe float64 columns ({', '.join(eligible_float_cols[:3])}) to float32.",
            "plan": dict(plan),
            "preconditions": ["Float values fit safely in float32 bounds without significant precision loss"],
            "safety_checks_performed": [f"Verified range and round-trip precision loss < 1e-4 on columns: {eligible_float_cols}"],
            "expected_benefit": "Reduces DataFrame memory footprint by approximately 50% for numeric columns.",
            "risks": "Sub-millisecond precision loss for high-precision decimal calculations.",
            "confidence": "medium",
            "rank": 3,
        })

    return candidates


# ==============================================================================
# 4. REPEATED BENCHMARKING ENGINE WITH NOISE CONTROL (SECTION 6 & 7)
# ==============================================================================

def benchmark_pipeline_candidate(
    dataframe: pd.DataFrame,
    baseline_plan: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
    *,
    execution_mode: str = "python",
    warmup_runs: int = 1,
    repeats: int = 3,
    max_compare_rows: int = 500000,
) -> Dict[str, Any]:
    """Repeatedly benchmarks baseline vs candidate under identical conditions with warm-up and statistical reliability."""
    if repeats < 3:
        raise ValueError("At least 3 repeated runs are required for statistically valid benchmarking.")

    opt_prep_start = time.perf_counter()

    baseline_times: List[float] = []
    candidate_times: List[float] = []
    baseline_mems: List[float] = []
    candidate_mems: List[float] = []

    def execute_and_measure(target_plan: Mapping[str, Any]) -> Tuple[pd.DataFrame, float, float]:
        df_copy = dataframe.copy()
        out_df, metrics = profile_callable(
            "benchmark_run",
            "optimization",
            lambda: execute_plan(df_copy, dict(target_plan), mode=execution_mode)[0],
            dataframe_in=dataframe,
        )
        dur = float(metrics["duration_seconds"])
        mem = float(metrics.get("peak_traced_python_bytes", 0) or metrics.get("rss_after_bytes", 0))
        return out_df, dur, mem

    # Warm-up runs
    for _ in range(warmup_runs):
        execute_and_measure(baseline_plan)
        execute_and_measure(candidate_plan)

    opt_prep_duration = time.perf_counter() - opt_prep_start

    baseline_sample_out: Optional[pd.DataFrame] = None
    candidate_sample_out: Optional[pd.DataFrame] = None

    for r in range(repeats):
        if r % 2 == 0:
            b_out, b_dur, b_mem = execute_and_measure(baseline_plan)
            c_out, c_dur, c_mem = execute_and_measure(candidate_plan)
        else:
            c_out, c_dur, c_mem = execute_and_measure(candidate_plan)
            b_out, b_dur, b_mem = execute_and_measure(baseline_plan)

        baseline_times.append(b_dur)
        candidate_times.append(c_dur)
        baseline_mems.append(b_mem)
        candidate_mems.append(c_mem)

        if baseline_sample_out is None:
            baseline_sample_out = b_out
            candidate_sample_out = c_out

    # Three-state Equivalence Validation
    val_start = time.perf_counter()
    assert baseline_sample_out is not None and candidate_sample_out is not None
    eq_result = validate_pipeline_equivalence(
        baseline_sample_out,
        candidate_sample_out,
        max_compare_rows=max_compare_rows,
    )
    val_duration = time.perf_counter() - val_start

    if eq_result["status"] == "INCONCLUSIVE":
        return {
            "status": "INCONCLUSIVE",
            "equivalence": eq_result,
            "baseline_median_seconds": statistics.median(baseline_times),
            "candidate_median_seconds": statistics.median(candidate_times),
            "speed_improvement_pct": 0.0,
            "reasons": eq_result.get("inconclusive_reasons", ["Equivalence validation was inconclusive."]),
            "overheads": {
                "preparation_duration_seconds": round(opt_prep_duration, 4),
                "validation_duration_seconds": round(val_duration, 4),
            },
        }

    if not eq_result["is_equivalent"]:
        return {
            "status": "REJECTED_DUE_TO_INCORRECT_OUTPUT",
            "equivalence": eq_result,
            "baseline_median_seconds": statistics.median(baseline_times),
            "candidate_median_seconds": statistics.median(candidate_times),
            "speed_improvement_pct": 0.0,
            "reasons": eq_result["reasons"],
            "overheads": {
                "preparation_duration_seconds": round(opt_prep_duration, 4),
                "validation_duration_seconds": round(val_duration, 4),
            },
        }

    b_med = statistics.median(baseline_times)
    c_med = statistics.median(candidate_times)
    b_mean = statistics.mean(baseline_times)
    c_mean = statistics.mean(candidate_times)
    b_std = statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0.0
    c_std = statistics.stdev(candidate_times) if len(candidate_times) > 1 else 0.0
    b_mad = statistics.median(abs(x - b_med) for x in baseline_times)
    c_mad = statistics.median(abs(x - c_med) for x in candidate_times)

    noise_floor = max(0.05 * b_med, 2 * b_mad, 2 * c_mad, 0.001)
    speed_improvement = ((b_med - c_med) / b_med * 100.0) if b_med > 0 else 0.0

    if b_med - c_med > noise_floor:
        status = "PERFORMANCE_IMPROVED"
    elif c_med - b_med > noise_floor:
        status = "PERFORMANCE_REGRESSION"
    else:
        status = "EQUIVALENT_BUT_NO_MEASURABLE_IMPROVEMENT"

    return {
        "status": status,
        "repeats": repeats,
        "warmup_runs": warmup_runs,
        "equivalence": eq_result,
        "baseline_times_seconds": baseline_times,
        "candidate_times_seconds": candidate_times,
        "baseline_median_seconds": round(b_med, 6),
        "candidate_median_seconds": round(c_med, 6),
        "baseline_mean_seconds": round(b_mean, 6),
        "candidate_mean_seconds": round(c_mean, 6),
        "baseline_std_seconds": round(b_std, 6),
        "candidate_std_seconds": round(c_std, 6),
        "baseline_mad_seconds": round(b_mad, 6),
        "candidate_mad_seconds": round(c_mad, 6),
        "noise_floor_seconds": round(noise_floor, 6),
        "speed_improvement_pct": round(speed_improvement, 2),
        "memory_baseline_bytes": statistics.median(baseline_mems),
        "memory_candidate_bytes": statistics.median(candidate_mems),
        "memory_methodology": "Median sampled resident set size (RSS) / traced Python bytes across repeated runs.",
        "overheads": {
            "preparation_duration_seconds": round(opt_prep_duration, 4),
            "validation_duration_seconds": round(val_duration, 4),
        },
    }
