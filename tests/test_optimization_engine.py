from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.etl.comparison_engine import validate_pipeline_equivalence
from src.etl.optimization_engine import (
    analyze_pipeline_structure,
    benchmark_pipeline_candidate,
    execute_user_python_code_safely,
    generate_optimization_candidates,
    validate_user_python_code,
    validate_user_sql_code,
)


def _sample_pipeline():
    return {
        "operations": [
            {
                "id": "op_1",
                "name": "Validation pass 1",
                "operation": "validate_dataset",
                "columns": ["id", "val"],
            },
            {
                "id": "op_2",
                "name": "Drop duplicate records",
                "operation": "drop_duplicates",
                "columns": ["id", "val"],
                "dependencies": ["op_1"],
            },
            {
                "id": "op_3",
                "name": "Validation pass 2",
                "operation": "validate_dataset",
                "columns": ["id", "val"],
                "dependencies": ["op_2"],
            },
        ]
    }


def _sample_df():
    return pd.DataFrame({
        "id": [1, 2, 2, 3],
        "val": [10.0, 20.0, 20.0, 30.0],
    })


# ==============================================================================
# 1. PYTHON SECURITY TESTS
# ==============================================================================

def test_validate_untrusted_python_allowed():
    res = validate_user_python_code("df['val'] = df['val'].fillna(df['val'].median())")
    assert res["valid"] is True
    assert res["outcome"] == "VALIDATED"


def test_validate_untrusted_python_forbidden_imports():
    assert validate_user_python_code("import os")["valid"] is False
    assert validate_user_python_code("from subprocess import Popen")["valid"] is False


def test_validate_untrusted_python_forbidden_calls_and_builtins():
    assert validate_user_python_code("eval('1+1')")["valid"] is False
    assert validate_user_python_code("exec('1+1')")["valid"] is False
    assert validate_user_python_code("__import__('os')")["valid"] is False
    assert validate_user_python_code("open('/etc/passwd', 'r')")["valid"] is False


def test_validate_untrusted_python_reflection():
    assert validate_user_python_code("df.__class__.__bases__")["valid"] is False
    assert validate_user_python_code("getattr(df, '__dict__')")["valid"] is False


def test_execute_user_python_safely():
    df = _sample_df()
    # Safe transformation
    code = "df['val'] = df['val'] * 2"
    exec_res = execute_user_python_code_safely(code, df)
    assert exec_res["outcome"] == "VALIDATED"
    assert list(exec_res["dataframe"]["val"]) == [20.0, 40.0, 40.0, 60.0]

    # Rejection of unsafe code
    bad_code = "import sys; sys.exit(1)"
    bad_res = execute_user_python_code_safely(bad_code, df)
    assert bad_res["outcome"] == "REJECTED"


# ==============================================================================
# 2. SQL SECURITY TESTS
# ==============================================================================

def test_validate_sql_safe_queries():
    assert validate_user_sql_code("SELECT DISTINCT id, val FROM input")["valid"] is True
    assert validate_user_sql_code("SELECT * FROM input WHERE val > 10")["valid"] is True


def test_validate_sql_comments_and_strings_safety():
    # String literal contains 'ATTACH', but it's safe inside quotes
    assert validate_user_sql_code("SELECT * FROM input WHERE val = 'ATTACH'")["valid"] is True
    # Harmless comment containing keyword
    assert validate_user_sql_code("SELECT * FROM input -- check pragma\nWHERE id = 1")["valid"] is True


def test_validate_sql_prohibits_multiple_statements_and_admin():
    # Multiple statements
    assert validate_user_sql_code("SELECT * FROM input; SELECT * FROM other;")["valid"] is False
    # Forbidden administrative/destructive queries
    assert validate_user_sql_code("ATTACH DATABASE 'test.db' AS db2")["valid"] is False
    assert validate_user_sql_code("DROP TABLE input")["valid"] is False
    assert validate_user_sql_code("PRAGMA journal_mode=WAL")["valid"] is False


# ==============================================================================
# 3. THREE-STATE EQUIVALENCE TESTS
# ==============================================================================

def test_three_state_equivalence_equivalent():
    df1 = pd.DataFrame({"a": [1, 2], "b": [10.0, 20.0]})
    res = validate_pipeline_equivalence(df1, df1.copy())
    assert res["status"] == "EQUIVALENT"
    assert res["is_equivalent"] is True


def test_three_state_equivalence_not_equivalent():
    df1 = pd.DataFrame({"a": [1, 2], "b": [10.0, 20.0]})
    df2 = pd.DataFrame({"a": [1, 2], "b": [10.0, 99.0]})
    res = validate_pipeline_equivalence(df1, df2)
    assert res["status"] == "NOT EQUIVALENT"
    assert res["is_equivalent"] is False


def test_three_state_equivalence_inconclusive_on_resource_limits():
    df1 = pd.DataFrame({"a": list(range(100))})
    res = validate_pipeline_equivalence(df1, df1.copy(), max_compare_rows=50)
    assert res["status"] == "INCONCLUSIVE"
    assert res["is_equivalent"] is False
    assert "exceeds configured in-memory comparison limit" in res["inconclusive_reasons"][0]


# ==============================================================================
# 4. OPTIMIZATION CANDIDATE & SAFETY TESTS
# ==============================================================================

def test_analyze_pipeline_structure():
    plan = _sample_pipeline()
    analysis = analyze_pipeline_structure(plan, dataframe=_sample_df())
    assert analysis["operation_count"] == 3
    assert len(analysis["noop_validations"]) == 2
    assert len(analysis["identified_bottlenecks"]) >= 1


def test_generate_optimization_candidates_with_preconditions():
    plan = _sample_pipeline()
    candidates = generate_optimization_candidates(plan, _sample_df(), execution_mode="python")
    assert len(candidates) >= 1
    cand = candidates[0]
    assert "preconditions" in cand
    assert "safety_checks_performed" in cand


def test_benchmark_pipeline_candidate_reports_stats_and_overheads():
    plan = _sample_pipeline()
    df = _sample_df()
    candidates = generate_optimization_candidates(plan, df, execution_mode="python")
    candidate = candidates[0]

    bench = benchmark_pipeline_candidate(
        df,
        plan,
        candidate["plan"],
        execution_mode="python",
        warmup_runs=1,
        repeats=3,
    )
    assert bench["status"] in {"PERFORMANCE_IMPROVED", "EQUIVALENT_BUT_NO_MEASURABLE_IMPROVEMENT"}
    assert bench["equivalence"]["status"] == "EQUIVALENT"
    assert "overheads" in bench
    assert "baseline_std_seconds" in bench
    assert "baseline_mad_seconds" in bench


def test_benchmark_candidate_rejects_non_equivalent():
    baseline_plan = {
        "steps": [{"id": "s1", "name": "noop", "operation": "validate_dataset", "columns": ["id"]}]
    }
    bad_candidate_plan = {
        "steps": [{"id": "s1", "name": "drop", "operation": "drop_duplicates", "columns": ["id"]}]
    }
    df = pd.DataFrame({"id": [1, 1, 2]})

    bench = benchmark_pipeline_candidate(
        df,
        baseline_plan,
        bad_candidate_plan,
        execution_mode="python",
        repeats=3,
    )
    assert bench["status"] == "REJECTED_DUE_TO_INCORRECT_OUTPUT"
    assert bench["equivalence"]["status"] == "NOT EQUIVALENT"
