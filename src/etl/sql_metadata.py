from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


SUPPORTED_SQL_OPERATIONS = frozenset(
    {
        "drop_duplicates",
        "fill_missing",
        "convert_dates",
        "normalize_columns",
        "clean_text",
        "validate_dataset",
    }
)


def quote_identifier(identifier: str) -> str:
    if not identifier or "\x00" in identifier:
        raise ValueError("SQL identifiers must be non-empty and cannot contain NUL.")
    return '"' + identifier.replace('"', '""') + '"'


def _safe_prefix(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    if not normalized:
        normalized = fallback
    if normalized[0].isdigit():
        normalized = f"{fallback}_{normalized}"
    return normalized[:40]


def internal_table_identifier(
    operation_id: str,
    operation_type: str,
    ordinal: int,
) -> str:
    stable_source = f"{ordinal}:{operation_id}:{operation_type}"
    digest = hashlib.sha256(stable_source.encode("utf-8")).hexdigest()[:10]
    prefix = _safe_prefix(operation_type, "step")
    return f"etl_{prefix}_{ordinal:04d}_{digest}"


def infer_sqlite_column_type(series: pd.Series) -> tuple[str, str]:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "INTEGER", f"Source dtype {dtype} maps to SQLite INTEGER."
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER", f"Source dtype {dtype} maps to SQLite INTEGER."
    if pd.api.types.is_float_dtype(dtype):
        return "REAL", f"Source dtype {dtype} maps to SQLite REAL."
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
        return "TEXT", f"Source dtype {dtype} is stored as ISO text."

    non_null = series.dropna()
    if len(non_null) and all(isinstance(value, (bytes, bytearray, memoryview)) for value in non_null):
        return "BLOB", "Every non-null value is bytes-like; stored as SQLite BLOB."
    if len(non_null) and all(isinstance(value, str) for value in non_null):
        return "TEXT", "String columns remain TEXT; numeric-looking strings are not coerced."
    return "BLOB", (
        "Mixed or ambiguous scalar values use BLOB affinity to preserve SQLite "
        "storage classes without text or numeric coercion."
    )


@dataclass(frozen=True)
class SQLColumnMetadata:
    position: int
    original_name: Any
    internal_name: str
    source_dtype: str
    sqlite_type: str
    mapping_status: str
    type_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SQLOperationMetadata:
    operation_id: str
    operation_type: str
    input_table_ids: tuple[str, ...]
    output_table_id: str
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    filter_conditions: tuple[str, ...]
    grouping_columns: tuple[str, ...]
    ordering_columns: tuple[str, ...]
    data_type_information: Mapping[str, str]
    dependencies: tuple[str, ...]
    execution_status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in (
            "input_table_ids",
            "input_columns",
            "output_columns",
            "filter_conditions",
            "grouping_columns",
            "ordering_columns",
            "dependencies",
        ):
            result[key] = list(result[key])
        result["data_type_information"] = dict(self.data_type_information)
        return result


@dataclass(frozen=True)
class SQLColumnMapping:
    columns: tuple[SQLColumnMetadata, ...]

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> "SQLColumnMapping":
        columns: List[SQLColumnMetadata] = []
        seen: set[str] = set()
        duplicate_mask = dataframe.columns.duplicated(keep=False)
        for position, name in enumerate(dataframe.columns):
            series = dataframe.iloc[:, position]
            sqlite_type, reason = infer_sqlite_column_type(series)
            digest = hashlib.sha256(
                f"{position}:{type(name).__name__}:{name!s}".encode("utf-8")
            ).hexdigest()[:10]
            prefix = _safe_prefix(str(name), "column")
            internal_name = f"etl_col_{position:04d}_{prefix}_{digest}"
            if internal_name in seen:
                raise ValueError("Internal SQL column identifier collision detected.")
            seen.add(internal_name)
            columns.append(
                SQLColumnMetadata(
                    position=position,
                    original_name=name,
                    internal_name=internal_name,
                    source_dtype=str(series.dtype),
                    sqlite_type=sqlite_type,
                    mapping_status=(
                        "ambiguous_source_label"
                        if duplicate_mask[position]
                        else "unique"
                    ),
                    type_reason=reason,
                )
            )
        return cls(tuple(columns))

    @property
    def original_names(self) -> list[Any]:
        return [column.original_name for column in self.columns]

    @property
    def internal_names(self) -> list[str]:
        return [column.internal_name for column in self.columns]

    def columns_for_labels(self, labels: Iterable[Any]) -> list[SQLColumnMetadata]:
        requested = list(labels)
        missing = [label for label in requested if label not in self.original_names]
        if missing:
            raise ValueError(f"Unknown SQL column(s): {missing!r}")
        return [
            column for column in self.columns
            if any(column.original_name == label for label in requested)
        ]


def validate_sql_plan(
    dataframe: pd.DataFrame,
    plan: Mapping[str, Any],
    mapping: SQLColumnMapping | None = None,
) -> tuple[SQLColumnMapping, list[SQLOperationMetadata]]:
    if len(dataframe.columns) == 0:
        raise ValueError("SQL execution requires at least one source column.")
    steps = plan.get("steps", [])
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise ValueError("SQL plan steps must be a sequence.")
    if any(not isinstance(step, Mapping) for step in steps):
        raise ValueError("Each SQL step must be an object.")
    mapping = mapping or SQLColumnMapping.from_dataframe(dataframe)
    if dataframe.columns.has_duplicates:
        duplicate_names = dataframe.columns[dataframe.columns.duplicated()].tolist()
        if any(
            label in (step.get("columns") or [])
            for step in steps
            for label in duplicate_names
        ):
            raise ValueError(
                "A plan cannot target duplicate source labels unambiguously; "
                "select a unique column name or rename duplicate headers first."
            )

    metadata: list[SQLOperationMetadata] = []
    previous_operation_id: str | None = None
    operation_ids: set[str] = set()
    current_table = "etl_source"
    for ordinal, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            raise ValueError(f"SQL step {ordinal} must be an object.")
        if "sql" in step or "query" in step:
            raise ValueError(
                f"Raw SQL is not accepted in pipeline step {ordinal}; use an allowlisted operation."
            )
        operation_type = step.get("operation")
        if operation_type not in SUPPORTED_SQL_OPERATIONS:
            raise ValueError(f"Unsupported SQL operation at step {ordinal}: {operation_type!r}")
        operation_id = str(step.get("id") or f"sql-operation-{ordinal}")
        if operation_id in operation_ids:
            raise ValueError(f"Duplicate SQL operation ID: {operation_id!r}")

        raw_columns = step.get("columns")
        if raw_columns is not None and (
            not isinstance(raw_columns, Sequence) or isinstance(raw_columns, (str, bytes))
        ):
            raise ValueError(f"SQL step {operation_id!r} columns must be a list.")
        selected = list(raw_columns or mapping.original_names)
        selected_columns = mapping.columns_for_labels(selected)
        if operation_type == "normalize_columns":
            selected_columns = list(mapping.columns)

        explicit_dependencies = step.get("dependencies")
        if explicit_dependencies is None:
            dependencies = (previous_operation_id,) if previous_operation_id else ()
        else:
            if not isinstance(explicit_dependencies, Sequence) or isinstance(explicit_dependencies, (str, bytes)):
                raise ValueError(f"SQL step {operation_id!r} dependencies must be a list.")
            dependencies = tuple(str(dependency) for dependency in explicit_dependencies)
            if any(dependency not in operation_ids for dependency in dependencies):
                raise ValueError(
                    f"SQL step {operation_id!r} references an unknown or forward dependency."
                )
            if previous_operation_id and previous_operation_id not in dependencies:
                raise ValueError(
                    f"SQL step {operation_id!r} must depend on the previous pipeline operation."
                )

        operation_ids.add(operation_id)
        output_table = internal_table_identifier(operation_id, str(operation_type), ordinal)
        metadata.append(
            SQLOperationMetadata(
                operation_id=operation_id,
                operation_type=str(operation_type),
                input_table_ids=(current_table,),
                output_table_id=output_table,
                input_columns=tuple(column.internal_name for column in selected_columns),
                output_columns=tuple(mapping.internal_names),
                filter_conditions=(),
                grouping_columns=(),
                ordering_columns=(),
                data_type_information={
                    column.internal_name: column.sqlite_type for column in mapping.columns
                },
                dependencies=dependencies,
            )
        )
        current_table = output_table
        previous_operation_id = operation_id
    return mapping, metadata
