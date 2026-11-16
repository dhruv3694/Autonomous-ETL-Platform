from __future__ import annotations

import pandas as pd
import pytest

from src.etl.isolated_runner import execute_python_isolated_process
from src.etl.optimization_engine import (
    execute_user_python_code_safely,
    validate_user_python_code,
    validate_user_sql_code,
)


def _sample_df():
    return pd.DataFrame({"id": [1, 2, 3], "val": [10.0, 20.0, 30.0]})


# ==============================================================================
# 1. PROCESS ISOLATION & TIMEOUT TESTS
# ==============================================================================

def test_isolated_process_executes_valid_code():
    df = _sample_df()
    code = "df['val'] = df['val'] + 5"
    res = execute_python_isolated_process(code, df)
    assert res["outcome"] == "VALIDATED"
    assert res["isolated"] is True
    assert list(res["dataframe"]["val"]) == [15.0, 25.0, 35.0]
    # Verify original df in caller memory was preserved
    assert list(df["val"]) == [10.0, 20.0, 30.0]


def test_isolated_process_timeout_enforcement():
    df = _sample_df()
    # Infinite loop
    loop_code = "while True:\n    pass"
    res = execute_python_isolated_process(loop_code, df, timeout_seconds=1.0)
    assert res["outcome"] == "EXECUTION_TIMEOUT"
    assert "timeout" in res["error"].lower()


def test_isolated_process_does_not_inherit_secrets():
    import os
    os.environ["SECRET_APP_KEY"] = "super_secret_12345"
    df = _sample_df()
    # Try to access SECRET_APP_KEY
    code = """
import os
val = os.environ.get('SECRET_APP_KEY', 'not_found')
df['secret'] = val
"""
    # Note that AST will reject 'import os' if run through execute_user_python_code_safely
    # But even if sent directly to isolated runner:
    res = execute_python_isolated_process(code, df)
    if res["outcome"] == "VALIDATED":
        assert list(res["dataframe"]["secret"]) == ["not_found", "not_found", "not_found"]
    else:
        # Either runtime error or not found
        pass


def test_isolated_process_runtime_error_captured_cleanly():
    df = _sample_df()
    code = "df['val'] = 1 / 0"
    res = execute_python_isolated_process(code, df)
    assert res["outcome"] == "EXECUTION_FAILED"
    assert "ZeroDivisionError" in res["error"]


# ==============================================================================
# 2. SQLPARSER ROBUSTNESS & ADVERSARIAL CASES
# ==============================================================================

def test_sqlparse_nested_subqueries_allowed():
    sql = "SELECT id, val FROM (SELECT id, val FROM input WHERE val > 15) sub"
    res = validate_user_sql_code(sql)
    assert res["valid"] is True
    assert res["outcome"] == "VALIDATED"


def test_sqlparse_prohibits_ddl_and_dml_mutations():
    assert validate_user_sql_code("INSERT INTO input VALUES (4, 40)")["valid"] is False
    assert validate_user_sql_code("UPDATE input SET val = 0")["valid"] is False
    assert validate_user_sql_code("DELETE FROM input WHERE id = 1")["valid"] is False
    assert validate_user_sql_code("DROP VIEW my_view")["valid"] is False


def test_sqlparse_rejects_empty_and_multiple_semicolons():
    assert validate_user_sql_code("   -- just comment\n   ")["valid"] is False
    assert validate_user_sql_code("SELECT 1; SELECT 2;")["valid"] is False
