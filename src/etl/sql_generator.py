from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from .sql_metadata import (
    SQLColumnMapping,
    SQLOperationMetadata,
    quote_identifier,
    validate_sql_plan,
)


def sanitize_identifier(name: Any) -> str:
    value = str(name).strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not value:
        value = "column"
    if value[0].isdigit():
        value = "c_" + value
    return value


@dataclass(frozen=True)
class GeneratedSQLOperation:
    metadata: SQLOperationMetadata
    sql: str
    display_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.metadata.to_dict(),
            "sql": self.sql,
            "display_name": self.display_name,
        }


def _column_expr(column: str) -> str:
    quoted = quote_identifier(column)
    return f"{quoted} AS {quoted}"


def _expressions_for_step(
    operation: SQLOperationMetadata,
    mapping: SQLColumnMapping,
) -> list[str]:
    operation_type = operation.operation_type
    selected_internal = set(operation.input_columns)
    expressions: list[str] = []
    for column in mapping.columns:
        quoted = quote_identifier(column.internal_name)
        if operation_type == "fill_missing" and column.internal_name in selected_internal:
            if column.sqlite_type in {"INTEGER", "REAL"}:
                expression = f"COALESCE({quoted}, MEDIAN({quoted}))"
            else:
                expression = f"COALESCE({quoted}, MODE({quoted}))"
            expressions.append(f"{expression} AS {quoted}")
        elif operation_type == "convert_dates" and column.internal_name in selected_internal:
            expressions.append(f"date({quoted}) AS {quoted}")
        elif operation_type == "clean_text" and column.internal_name in selected_internal:
            expressions.append(f"trim(CAST({quoted} AS TEXT)) AS {quoted}")
        else:
            expressions.append(_column_expr(column.internal_name))
    return expressions


def generate_sql_operations(
    dataframe: pd.DataFrame,
    plan: Dict[str, Any],
) -> tuple[SQLColumnMapping, list[GeneratedSQLOperation]]:
    mapping, operations = validate_sql_plan(dataframe, plan)
    generated: list[GeneratedSQLOperation] = []
    for step, metadata in zip(plan.get("steps", []), operations):
        source_table = quote_identifier(metadata.input_table_ids[0])
        target_table = quote_identifier(metadata.output_table_id)
        if metadata.operation_type == "drop_duplicates":
            select_sql = f"SELECT DISTINCT * FROM {source_table}"
        else:
            if metadata.operation_type == "normalize_columns":
                expressions = [_column_expr(column) for column in mapping.internal_names]
            else:
                expressions = _expressions_for_step(metadata, mapping)
            select_sql = f"SELECT {', '.join(expressions)} FROM {source_table}"
        generated.append(
            GeneratedSQLOperation(
                metadata=metadata,
                sql=f"CREATE TABLE {target_table} AS {select_sql};",
                display_name=str(step.get("name") or metadata.operation_id),
            )
        )
    return mapping, generated


def build_sql_from_dataframe(df: pd.DataFrame, table_name: str = "etl_source") -> Tuple[str, List[str]]:
    mapping = SQLColumnMapping.from_dataframe(df)
    quoted_table = quote_identifier(table_name)
    definitions = ", ".join(
        f"{quote_identifier(column.internal_name)} {column.sqlite_type}"
        for column in mapping.columns
    )
    create_sql = f"CREATE TABLE {quoted_table} ({definitions});"
    placeholders = ", ".join("?" for _ in mapping.columns)
    insert_sql = (
        f"INSERT INTO {quoted_table} "
        f"({', '.join(quote_identifier(column.internal_name) for column in mapping.columns)}) "
        f"VALUES ({placeholders});"
    )
    return f"{create_sql}\n{insert_sql}", mapping.internal_names


def generate_sql_steps(df: pd.DataFrame, plan: Dict[str, Any]) -> List[str]:
    _, operations = generate_sql_operations(df, plan)
    return [operation.sql for operation in operations]
