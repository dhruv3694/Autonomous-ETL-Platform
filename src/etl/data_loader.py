from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .performance_profiler import profile_callable


SUPPORTED_EXTENSIONS = {
    "csv": "csv",
    "tsv": "tsv",
    "txt": "txt",
    "json": "json",
    "jsonl": "jsonl",
    "xml": "xml",
    "xlsx": "xlsx",
    "xls": "xls",
    "parquet": "parquet",
    "db": "sqlite",
    "sqlite": "sqlite",
    "sqlite3": "sqlite",
    "sql": "sql"
}


def _infer_delimiter(name: str, sample: str) -> str:
    delimiters = [",", "\t", ";", "|", ":"]
    for delimiter in delimiters:
        if sample.count(delimiter) > sample.count(" ") * 0.2:
            return delimiter
    return ","


def _safe_json_loads(payload: str) -> Any:
    try:
        return json.loads(payload)
    except Exception:
        return None


def _read_csv_like_bytes(raw_bytes: bytes, file_name: str) -> pd.DataFrame:
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    if "\n" not in text and "\r" not in text:
        raise ValueError("File appears to be empty or unreadable.")

    delimiter = _infer_delimiter(file_name, text[:2000])
    return pd.read_csv(io.StringIO(text), sep=delimiter, engine="python")


def _read_json_like(raw_bytes: bytes, file_name: str) -> pd.DataFrame:
    payload = raw_bytes.decode("utf-8-sig", errors="replace")
    json_data = _safe_json_loads(payload)
    if isinstance(json_data, list):
        return pd.DataFrame(json_data)
    if isinstance(json_data, dict):
        if "records" in json_data and isinstance(json_data["records"], list):
            return pd.DataFrame(json_data["records"])
        return pd.DataFrame([json_data])
    raise ValueError(f"Unsupported JSON structure in {file_name}.")


def _read_jsonl(raw_bytes: bytes) -> pd.DataFrame:
    lines = [line for line in raw_bytes.decode("utf-8-sig", errors="replace").splitlines() if line.strip()]
    records = []
    for line in lines:
        row = _safe_json_loads(line)
        if isinstance(row, dict):
            records.append(row)
    if not records:
        raise ValueError("No valid JSONL records were found.")
    return pd.DataFrame(records)


def _read_xml(raw_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_xml(io.BytesIO(raw_bytes))
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise ValueError(f"Unable to parse XML: {exc}") from exc


def _read_excel_workbook(raw_bytes: bytes) -> List[Dict[str, Any]]:
    buffer = io.BytesIO(raw_bytes)
    excel_file = pd.ExcelFile(buffer)
    datasets: List[Dict[str, Any]] = []
    sheet_names = excel_file.sheet_names
    if not sheet_names:
        raise ValueError("Excel file contains no sheets.")
    for sheet_name in sheet_names:
        buffer.seek(0)
        df = pd.read_excel(buffer, sheet_name=sheet_name)
        datasets.append({"sheet": sheet_name, "dataframe": df})
    return datasets


def _read_sqlite_database(raw_bytes: bytes, file_name: str) -> List[Dict[str, Any]]:
    path = Path(file_name)
    temp_db = Path.cwd() / f"{path.stem}_{abs(hash(file_name))}.db"
    temp_db.write_bytes(raw_bytes)
    try:
        conn = sqlite3.connect(str(temp_db))
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        datasets: List[Dict[str, Any]] = []
        for (table_name,) in tables:
            df = pd.read_sql_query(f"SELECT * FROM \"{table_name}\"", conn)
            datasets.append({"table": table_name, "dataframe": df})
        if not datasets:
            raise ValueError("SQLite database contains no tables.")
        return datasets
    finally:
        try:
            temp_db.unlink(missing_ok=True)
        except Exception:
            pass


def _read_parquet(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(raw_bytes))


def _read_sql_dump(raw_bytes: bytes) -> pd.DataFrame:
    raise ValueError("SQL dump files are supported as metadata-only imports; full execution requires schema import.")


def _read_uploaded_file(file_like: Any) -> List[Dict[str, Any]]:
    file_name = getattr(file_like, "name", "uploaded_file")
    logical_name = file_name.lower()
    extension = logical_name.rsplit(".", 1)[-1] if "." in logical_name else ""
    kind = SUPPORTED_EXTENSIONS.get(extension, "unsupported")
    raw_bytes = file_like.read()

    if kind == "unsupported":
        raise ValueError(f"Unsupported file type '{extension}' for {file_name}.")

    if kind in {"csv", "tsv", "txt"}:
        return [{"file_name": file_name, "source": "file", "dataframe": _read_csv_like_bytes(raw_bytes, file_name)}]
    if kind == "json":
        return [{"file_name": file_name, "source": "file", "dataframe": _read_json_like(raw_bytes, file_name)}]
    if kind == "jsonl":
        return [{"file_name": file_name, "source": "file", "dataframe": _read_jsonl(raw_bytes)}]
    if kind == "xml":
        return [{"file_name": file_name, "source": "file", "dataframe": _read_xml(raw_bytes)}]
    if kind in {"xlsx", "xls"}:
        frames = _read_excel_workbook(raw_bytes)
        result = []
        for item in frames:
            result.append({"file_name": file_name, "source": "file", "sheet": item["sheet"], "dataframe": item["dataframe"]})
        return result
    if kind == "parquet":
        return [{"file_name": file_name, "source": "file", "dataframe": _read_parquet(raw_bytes)}]
    if kind == "sqlite":
        tables = _read_sqlite_database(raw_bytes, file_name)
        result = []
        for table in tables:
            result.append({
                "file_name": file_name,
                "source": "database",
                "table": table["table"],
                "dataframe": table["dataframe"],
            })
        return result
    if kind == "sql":
        _read_sql_dump(raw_bytes)
    raise ValueError(f"Unable to import {file_name}.")


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    numeric_cols = list(df.select_dtypes(include="number").columns)
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

    summary = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "null_count": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric_cols,
        "text_columns": non_numeric_cols,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
        "column_profiles": [
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "nulls": int(df[col].isna().sum()),
                "unique_values": int(df[col].nunique(dropna=True)),
                "sample_values": df[col].dropna().astype(str).head(5).tolist(),
            }
            for col in df.columns
        ],
    }
    return summary


def load_uploaded_files(uploaded_files: Iterable[Any]) -> List[Dict[str, Any]]:
    datasets: List[Dict[str, Any]] = []
    for file_like in uploaded_files:
        try:
            imported_entries, extraction_performance = profile_callable(
                name=f"Extract {getattr(file_like, 'name', 'uploaded file')}",
                stage="extraction",
                operation=lambda: _read_uploaded_file(file_like),
                bytes_in=getattr(file_like, "size", None),
            )
            extraction_performance["records_out"] = sum(
                len(entry.get("dataframe", pd.DataFrame()))
                for entry in imported_entries
                if isinstance(entry.get("dataframe"), pd.DataFrame)
            )
            if extraction_performance["duration_seconds"] > 0:
                extraction_performance["records_per_second"] = (
                    extraction_performance["records_out"]
                    / extraction_performance["duration_seconds"]
                )
            extraction_performance["dataframe_output_bytes"] = sum(
                int(entry["dataframe"].memory_usage(index=True, deep=True).sum())
                for entry in imported_entries
                if isinstance(entry.get("dataframe"), pd.DataFrame)
            )
            for entry in imported_entries:
                frame = entry.get("dataframe")
                if frame is None:
                    continue
                frame = frame.copy()
                frame.columns = [str(col).strip() for col in frame.columns]
                summary, summary_performance = profile_callable(
                    name=f"Profile {entry.get('table') or entry.get('sheet') or entry.get('file_name', 'dataset')}",
                    stage="schema_analysis",
                    operation=lambda frame=frame: summarize_dataframe(frame),
                    dataframe_in=frame,
                )
                datasets.append({
                    "file_name": entry.get("file_name", getattr(file_like, "name", "uploaded_dataset")),
                    "source": entry.get("source", "file"),
                    "sheet": entry.get("sheet"),
                    "table": entry.get("table"),
                    "dataframe": frame,
                    "summary": summary,
                    "performance": {
                        "extraction": extraction_performance,
                        "schema_analysis": summary_performance,
                    },
                })
        except ValueError as exc:
            datasets.append({
                "file_name": getattr(file_like, "name", "uploaded_dataset"),
                "source": "error",
                "error": str(exc),
                "dataframe": pd.DataFrame(),
                "summary": {"row_count": 0, "column_count": 0, "columns": [], "error": str(exc)},
            })
    if not datasets:
        raise ValueError("No datasets were uploaded.")
    return datasets
