from __future__ import annotations

import re
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd

from .sql_generator import generate_sql_operations
from .performance_profiler import profile_callable
from .sql_metadata import SQLColumnMapping, quote_identifier


class SQLExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        execution_id: str,
        step_id: str,
        operation_type: str,
        error: Exception,
        sql_context: str,
        audit: List[Dict[str, Any]],
    ) -> None:
        self.execution_id = execution_id
        self.step_id = step_id
        self.operation_type = operation_type
        self.error_category = type(error).__name__
        self.error_message = str(error)
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.sql_context = _redact_sql_literals(sql_context)
        self.audit = audit
        self.status = "failed"
        super().__init__(
            f"SQL execution {execution_id} failed at step {step_id} "
            f"({operation_type}): {self.error_message}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "step_id": self.step_id,
            "operation_type": self.operation_type,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "sql_context": self.sql_context,
            "status": self.status,
        }


def _redact_sql_literals(sql: str, limit: int = 1000) -> str:
    redacted = re.sub(r"'(?:''|[^'])*'", "'[REDACTED]'", sql)
    if len(redacted) > limit:
        return redacted[:limit] + "... [truncated]"
    return redacted


class _MedianAggregate:
    def __init__(self) -> None:
        self.values: list[float] = []

    def step(self, value: Any) -> None:
        if value is not None:
            self.values.append(float(value))

    def finalize(self) -> float | None:
        if not self.values:
            return None
        ordered = sorted(self.values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2


class _ModeAggregate:
    def __init__(self) -> None:
        self.values: list[Any] = []

    def step(self, value: Any) -> None:
        if value is not None:
            self.values.append(value)

    def finalize(self) -> Any:
        if not self.values:
            return "unknown"
        counts = Counter(self.values)
        highest = max(counts.values())
        candidates = [value for value, count in counts.items() if count == highest]
        try:
            return sorted(candidates)[0]
        except TypeError:
            return sorted(candidates, key=str)[0]


def _sqlite_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _restore_sql_output(
    result: pd.DataFrame,
    mapping: SQLColumnMapping,
    output_names: List[Any] | None = None,
    output_dtypes: List[Any] | None = None,
) -> pd.DataFrame:
    columns = []
    for position, internal_name in enumerate(mapping.internal_names):
        column = result[internal_name]
        if output_dtypes is not None and output_dtypes[position] is not None:
            column = column.astype(output_dtypes[position])
        columns.append(column.reset_index(drop=True))
    restored = pd.concat(columns, axis=1) if columns else pd.DataFrame()
    restored.columns = output_names or mapping.original_names
    return restored


def _clean_text_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [re.sub(r"\s+", "_", str(col)).lower().strip(" ") for col in renamed.columns]
    return renamed


def _normalized_column_names(names: List[Any]) -> List[str]:
    return [re.sub(r"\s+", "_", str(name)).lower().strip(" ") for name in names]


def validate_dataframe(before: pd.DataFrame, after: pd.DataFrame) -> Dict[str, Any]:
    row_delta = len(after) - len(before)
    row_loss_pct = abs(row_delta) / max(len(before), 1) * 100
    duplicate_rows = int(after.duplicated().sum())
    null_count = int(after.isna().sum().sum())
    issues: List[str] = []

    if row_loss_pct > 25:
        issues.append(f"Large row change detected ({row_loss_pct:.1f}% from original).")
    if duplicate_rows > 0:
        issues.append(f"Duplicate rows remain: {duplicate_rows}.")
    if len(after.columns) == 0:
        issues.append("No columns remain after transformation.")

    status = "passed" if not issues else "failed"
    return {
        "status": status,
        "issues": issues,
        "row_delta": row_delta,
        "row_loss_pct": round(row_loss_pct, 2),
        "duplicate_rows": duplicate_rows,
        "null_count": null_count,
        "row_count": int(len(after)),
    }


def _execute_python_plan(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    output = df.copy()
    audit: List[Dict[str, Any]] = []
    steps = plan.get("steps", [])

    for step in steps:
        operation = step.get("operation")
        columns = step.get("columns") or list(output.columns)
        description = step.get("description", "Apply ETL step")

        before = output

        def apply_operation() -> pd.DataFrame:
            result = before.copy()
            if operation == "drop_duplicates":
                return result.drop_duplicates().reset_index(drop=True)
            if operation == "fill_missing":
                for col in result.columns:
                    if col in columns:
                        if pd.api.types.is_numeric_dtype(result[col]):
                            result[col] = result[col].fillna(result[col].median())
                        else:
                            mode_value = result[col].mode(dropna=True)
                            result[col] = result[col].fillna(
                                mode_value.iloc[0] if not mode_value.empty else "unknown"
                            )
                return result
            if operation == "convert_dates":
                for col in result.columns:
                    if col in columns:
                        result[col] = pd.to_datetime(result[col], errors="coerce", format="mixed")
                return result
            if operation == "normalize_columns":
                return _normalize_columns(result)
            if operation == "clean_text":
                for col in result.columns:
                    if col in columns:
                        result[col] = _clean_text_series(result[col]).str.replace(
                            r"\s+", " ", regex=True
                        )
                return result
            return result

        output, performance = profile_callable(
            name=step.get("name", f"step_{len(audit) + 1}"),
            stage="transformation",
            operation=apply_operation,
            dataframe_in=before,
        )
        validation = validate_dataframe(before, output)
        audit.append(
            {
                "step": step.get("name", "Generic transformation"),
                "description": description,
                "status": validation["status"],
                "details": validation,
                "performance": performance,
            }
        )

    return output, audit


def _execute_sql_plan(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    conn = sqlite3.connect(":memory:")
    conn.create_aggregate("MEDIAN", 1, _MedianAggregate)
    conn.create_aggregate("MODE", 1, _ModeAggregate)
    current_table = "etl_source"
    audit: List[Dict[str, Any]] = []
    execution_id = str(uuid.uuid4())

    try:
        mapping, generated_operations = generate_sql_operations(df, plan)
    except Exception as exc:
        conn.close()
        raise SQLExecutionError(
            execution_id=execution_id,
            step_id="validate-plan",
            operation_type="validate_plan",
            error=exc,
            sql_context="",
            audit=audit,
        ) from exc

    safe_columns = mapping.columns
    current_output_names = mapping.original_names
    current_output_dtypes: List[Any] = list(df.dtypes)
    create_table_sql = (
        f"CREATE TABLE {quote_identifier(current_table)} ("
        + ", ".join(
            f"{quote_identifier(column.internal_name)} {column.sqlite_type}"
            for column in safe_columns
        )
        + ");"
    )

    def load_source() -> None:
        conn.execute(create_table_sql)
        if len(df):
            placeholders = ", ".join("?" for _ in safe_columns)
            insert_sql = (
                f"INSERT INTO {quote_identifier(current_table)} "
                f"({', '.join(quote_identifier(column.internal_name) for column in safe_columns)}) "
                f"VALUES ({placeholders})"
            )
            rows = (
                tuple(_sqlite_value(value) for value in row)
                for row in df.itertuples(index=False, name=None)
            )
            conn.executemany(insert_sql, rows)

    try:
        _, load_performance = profile_callable(
            name="Load source into SQLite",
            stage="loading",
            operation=load_source,
            dataframe_in=df,
        )
    except Exception as exc:
        audit.append(
            {
                "step": "Load source into SQLite",
                "step_id": "load-source",
                "operation": "load_source",
                "status": "failed",
                "execution_status": "failed",
                "details": {"error_category": type(exc).__name__, "error_message": str(exc)},
            }
        )
        conn.close()
        raise SQLExecutionError(
            execution_id=execution_id,
            step_id="load-source",
            operation_type="load_source",
            error=exc,
            sql_context=create_table_sql,
            audit=audit,
        ) from exc
    audit.append(
        {
            "step": "Load source into SQLite",
            "description": "Copy input rows to the in-memory SQLite execution table.",
            "status": "completed",
            "execution_status": "succeeded",
            "step_id": "load-source",
            "operation": "load_source",
            "performance": load_performance,
        }
    )

    for index, (step, operation) in enumerate(
        zip(plan.get("steps", []), generated_operations), start=1
    ):
        sql = operation.sql
        try:
            before = pd.read_sql_query(
                f"SELECT * FROM {quote_identifier(current_table)}", conn
            )
            restored_before = _restore_sql_output(
                before, mapping, current_output_names, current_output_dtypes
            )
            operation_output_names = (
                _normalized_column_names(current_output_names)
                if operation.metadata.operation_type == "normalize_columns"
                else current_output_names
            )
            modified_columns = (
                set(operation.metadata.input_columns)
                if operation.metadata.operation_type
                in {"fill_missing", "convert_dates", "clean_text"}
                else set()
            )
            operation_output_dtypes = [
                dtype if column.internal_name not in modified_columns else None
                for column, dtype in zip(mapping.columns, current_output_dtypes)
            ]

            def execute_sql_step() -> pd.DataFrame:
                nonlocal current_table
                conn.execute(sql)
                current_table = operation.metadata.output_table_id
                internal_result = pd.read_sql_query(
                    f"SELECT * FROM {quote_identifier(current_table)}", conn
                )
                return _restore_sql_output(
                    internal_result,
                    mapping,
                    operation_output_names,
                    operation_output_dtypes,
                )

            result, performance = profile_callable(
                name=operation.display_name,
                stage="transformation",
                operation=execute_sql_step,
                dataframe_in=before,
            )
            validation = validate_dataframe(restored_before, result)
            audit.append({
                "step": operation.display_name,
                "description": step.get("description", "Apply SQL ETL step"),
                "status": validation["status"],
                "execution_status": "succeeded",
                "step_id": operation.metadata.operation_id,
                "operation": operation.metadata.operation_type,
                "details": validation,
                "sql": sql,
                "metadata": operation.metadata.to_dict(),
                "performance": performance,
            })
            current_output_names = operation_output_names
            current_output_dtypes = list(result.dtypes)
        except Exception as exc:
            failed_entry = {
                "step": operation.display_name,
                "step_id": operation.metadata.operation_id,
                "operation": operation.metadata.operation_type,
                "description": step.get("description", "Apply SQL ETL step"),
                "status": "failed",
                "execution_status": "failed",
                "details": {"status": "failed", "issues": [str(exc)]},
                "sql": _redact_sql_literals(sql),
                "performance": {"name": step.get("name", f"SQL step {index}"), "stage": "transformation", "status": "failed"},
                "metadata": operation.metadata.to_dict(),
            }
            audit.append(failed_entry)
            conn.close()
            raise SQLExecutionError(
                execution_id=execution_id,
                step_id=operation.metadata.operation_id,
                operation_type=operation.metadata.operation_type,
                error=exc,
                sql_context=sql,
                audit=audit,
            ) from exc

    try:
        internal_result = pd.read_sql_query(
            f"SELECT * FROM {quote_identifier(current_table)}", conn
        )
        final_result = _restore_sql_output(
            internal_result,
            mapping,
            current_output_names,
            current_output_dtypes,
        )
    except Exception as exc:
        audit.append(
            {
                "step": "Read final SQL result",
                "step_id": "read-final-result",
                "operation": "read_result",
                "status": "failed",
                "execution_status": "failed",
                "details": {"error_category": type(exc).__name__, "error_message": str(exc)},
            }
        )
        conn.close()
        raise SQLExecutionError(
            execution_id=execution_id,
            step_id="read-final-result",
            operation_type="read_result",
            error=exc,
            sql_context=f"SELECT * FROM {current_table}",
            audit=audit,
        ) from exc
    conn.close()
    return final_result, audit


def execute_plan(df: pd.DataFrame, plan: Dict[str, Any], mode: str | None = None) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    selected_mode = mode or plan.get("execution_mode") or "python"
    if selected_mode == "sql":
        return _execute_sql_plan(df, plan)
    return _execute_python_plan(df, plan)
