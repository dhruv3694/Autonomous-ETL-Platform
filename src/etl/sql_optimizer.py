from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List, Mapping

import pandas as pd

from .execution_engine import execute_plan
from .performance_profiler import profile_callable
from .sql_generator import generate_sql_operations
from .sql_metadata import quote_identifier


def _create_source_table(connection: Any, dataframe: pd.DataFrame, mapping: Any) -> None:
    definitions = ", ".join(
        f"{quote_identifier(column.internal_name)} {column.sqlite_type}"
        for column in mapping.columns
    )
    connection.execute(f"CREATE TABLE {quote_identifier('etl_source')} ({definitions})")
    if dataframe.empty:
        return
    placeholders = ", ".join("?" for _ in mapping.columns)
    insert_sql = (
        f"INSERT INTO {quote_identifier('etl_source')} "
        f"({', '.join(quote_identifier(column.internal_name) for column in mapping.columns)}) "
        f"VALUES ({placeholders})"
    )

    def as_sqlite_value(value: Any) -> Any:
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    rows = (
        tuple(as_sqlite_value(value) for value in row)
        for row in dataframe.itertuples(index=False, name=None)
    )
    connection.executemany(insert_sql, rows)


def analyze_sql_plan(dataframe: pd.DataFrame, plan: Mapping[str, Any]) -> Dict[str, Any]:
    import sqlite3

    mapping, operations = generate_sql_operations(dataframe, dict(plan))
    connection = sqlite3.connect(":memory:")
    try:
        _create_source_table(connection, dataframe, mapping)
        operation_plans = []
        for operation in operations:
            prefix = f'CREATE TABLE {quote_identifier(operation.metadata.output_table_id)} AS '
            query = operation.sql.removeprefix(prefix).rstrip(";")
            rows = connection.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
            details = [
                {
                    "id": int(row[0]),
                    "parent": int(row[1]),
                    "detail": str(row[3]),
                    "access_type": (
                        "search"
                        if str(row[3]).upper().startswith("SEARCH ")
                        else "scan"
                        if str(row[3]).upper().startswith("SCAN ")
                        else "other"
                    ),
                    "uses_temporary_btree": (
                        "USE TEMP B-TREE" in str(row[3]).upper()
                    ),
                }
                for row in rows
            ]
            operation_plans.append(
                {
                    "operation_id": operation.metadata.operation_id,
                    "operation": operation.metadata.operation_type,
                    "sql": operation.sql,
                    "plan": details,
                    "filter_conditions": list(operation.metadata.filter_conditions),
                    "grouping_columns": list(operation.metadata.grouping_columns),
                    "ordering_columns": list(operation.metadata.ordering_columns),
                }
            )
            connection.execute(operation.sql)

        recommendations = [
            {
                "kind": "index",
                "status": "not_recommended",
                "recommendation": (
                    "No index is recommended: this plan has no selective filter, "
                    "join, grouping, or ordering key for an index to target."
                ),
                "evidence": operation_plans,
            }
        ]
        return {
            "engine": "SQLite",
            "operations": operation_plans,
            "recommendations": recommendations,
            "evidence_note": (
                "EXPLAIN QUERY PLAN is structural evidence, not a runtime measurement. "
                "No index is created automatically."
            ),
        }
    finally:
        connection.close()


def generate_sql_candidates(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("SQL plan steps must be a list.")
    candidate_steps = []
    previous_id = None
    for step in steps:
        if not isinstance(step, Mapping):
            raise ValueError("Each SQL plan step must be an object.")
        if step.get("operation") == "validate_dataset":
            continue
        retained = dict(step)
        if previous_id is not None:
            retained["dependencies"] = [previous_id]
        else:
            retained.pop("dependencies", None)
        candidate_steps.append(retained)
        previous_id = str(step.get("id") or f"sql-operation-{len(candidate_steps)}")
    removed = len(steps) - len(candidate_steps)
    if not removed:
        return []
    candidate_plan = dict(plan)
    candidate_plan["steps"] = candidate_steps
    candidate_plan["source"] = "deterministic_sql_optimizer"
    return [
        {
            "candidate_id": "remove-sql-noop-validation",
            "description": (
                f"Remove {removed} validate_dataset step(s), which generate a "
                "row-preserving SQL projection without changing values."
            ),
            "plan": candidate_plan,
            "approval_required": True,
            "approved": False,
        }
    ]


def _cell_signature(value: Any) -> tuple[str, str]:
    if pd.api.types.is_scalar(value) and bool(pd.isna(value)):
        return ("null", "")
    if hasattr(value, "item"):
        value = value.item()
    return (type(value).__name__, repr(value))


def compare_sql_outputs(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> Dict[str, Any]:
    reasons = []
    if list(baseline.columns) != list(candidate.columns):
        reasons.append("Column labels or order differ.")
    if list(map(str, baseline.dtypes)) != list(map(str, candidate.dtypes)):
        reasons.append("Column dtypes differ.")
    same_shape = baseline.shape == candidate.shape
    same_columns = list(baseline.columns) == list(candidate.columns)
    same_dtypes = list(map(str, baseline.dtypes)) == list(map(str, candidate.dtypes))
    if not same_shape:
        reasons.append("Row or column counts differ.")
    rows_match = False
    if same_shape and same_columns:
        baseline_rows = Counter(
            tuple(_cell_signature(value) for value in row)
            for row in baseline.itertuples(index=False, name=None)
        )
        candidate_rows = Counter(
            tuple(_cell_signature(value) for value in row)
            for row in candidate.itertuples(index=False, name=None)
        )
        rows_match = baseline_rows == candidate_rows
        if not rows_match:
            reasons.append("Row values, nulls, or duplicate multiplicities differ.")
    return {
        "equivalent": not reasons,
        "checks": {
            "column_order_and_labels": same_columns,
            "dtypes": same_dtypes,
            "shape": same_shape,
            "row_multiset_including_nulls_and_duplicates": rows_match,
        },
        "reasons": reasons,
    }


def benchmark_sql_candidate(
    dataframe: pd.DataFrame,
    baseline_plan: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
    *,
    repeats: int = 3,
) -> Dict[str, Any]:
    if repeats < 3:
        raise ValueError("At least three repeated runs are required for benchmarking.")

    baseline_times: List[float] = []
    candidate_times: List[float] = []

    def measure(label: str, plan: Mapping[str, Any]) -> tuple[pd.DataFrame, float]:
        output, metrics = profile_callable(
            label,
            "benchmark",
            lambda: execute_plan(dataframe.copy(), dict(plan), mode="sql")[0],
            dataframe_in=dataframe,
        )
        return output, float(metrics["duration_seconds"])

    for run_index in range(repeats):
        if run_index % 2:
            candidate, candidate_seconds = measure(
                "SQL candidate run", candidate_plan
            )
            baseline, baseline_seconds = measure("SQL baseline run", baseline_plan)
        else:
            baseline, baseline_seconds = measure("SQL baseline run", baseline_plan)
            candidate, candidate_seconds = measure(
                "SQL candidate run", candidate_plan
            )
        equivalence = compare_sql_outputs(baseline, candidate)
        baseline_times.append(baseline_seconds)
        candidate_times.append(candidate_seconds)
        if not equivalence["equivalent"]:
            return {
                "status": "rejected_incorrect",
                "equivalence": equivalence,
                "baseline_runs_seconds": baseline_times,
                "candidate_runs_seconds": candidate_times,
                "improvement_claim": None,
            }

    baseline_median = statistics.median(baseline_times)
    candidate_median = statistics.median(candidate_times)
    relative_change = (
        (baseline_median - candidate_median) / baseline_median
        if baseline_median
        else 0.0
    )
    baseline_mad = statistics.median(
        abs(value - baseline_median) for value in baseline_times
    )
    candidate_mad = statistics.median(
        abs(value - candidate_median) for value in candidate_times
    )
    noise_floor = max(0.05 * baseline_median, 2 * baseline_mad, 2 * candidate_mad)
    if baseline_median - candidate_median > noise_floor:
        status = "measured_improvement"
    elif candidate_median - baseline_median > noise_floor:
        status = "measured_regression"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "repeats": repeats,
        "baseline_runs_seconds": baseline_times,
        "candidate_runs_seconds": candidate_times,
        "baseline_median_seconds": baseline_median,
        "candidate_median_seconds": candidate_median,
        "baseline_median_absolute_deviation_seconds": baseline_mad,
        "candidate_median_absolute_deviation_seconds": candidate_mad,
        "decision_noise_floor_seconds": noise_floor,
        "relative_time_change": relative_change,
        "equivalence": equivalence,
        "measurement_note": (
            "Repeated in-process measurements include SQL execution and profiler "
            "overhead. Run order alternates; differences must exceed both five "
            "percent of baseline median and twice either run set's median absolute "
            "deviation to be called measured."
        ),
        "improvement_claim": status == "measured_improvement",
    }
