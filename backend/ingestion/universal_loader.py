"""
Data Ingestion and Universal Multi-Format Parsing Module.
Supports CSV, TSV, XLSX, XLS, JSON, JSONL, XML, Parquet, SQLite (.db, .sqlite), SQL dumps, TXT, PDF, and DOCX.
Always returns a pandas DataFrame, extracted metadata, and ensures original data is immutably preserved.
"""

import os
import io
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

SUPPORTED_EXTENSIONS = [
    ".csv", ".tsv", ".txt",
    ".xlsx", ".xls",
    ".json", ".jsonl",
    ".parquet",
    ".xml",
    ".sqlite", ".db",
    ".sql",
    ".pdf", ".docx"
]

def calculate_file_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 fingerprint of uploaded file."""
    return hashlib.sha256(file_bytes).hexdigest()

def detect_text_delimiter(sample_text: str) -> str:
    """Heuristic detection of delimiter for plain text / csv."""
    delimiters = [',', '\t', ';', '|']
    lines = [line.strip() for line in sample_text.splitlines()[:15] if line.strip()]
    if not lines:
        return ','
    scores = {}
    for d in delimiters:
        counts = [line.count(d) for line in lines]
        if all(c > 0 for c in counts) and len(set(counts)) == 1:
            scores[d] = 100 + counts[0]
        else:
            scores[d] = sum(counts) / (len(counts) or 1)
    return max(scores, key=scores.get) if scores else ','

def parse_dataset_bytes(file_name: str, file_bytes: bytes, sheet_name: Optional[str] = None, table_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Universally parse incoming bytes into a DataFrame and extraction metadata.
    Raises ValueError with explanatory message on unsupported or malformed data.
    """
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")

    meta: Dict[str, Any] = {
        "file_name": file_name,
        "extension": ext,
        "file_size_bytes": len(file_bytes),
        "file_hash": calculate_file_hash(file_bytes),
        "available_sheets": [],
        "available_tables": [],
        "selected_sheet": None,
        "selected_table": None,
        "encoding": "utf-8",
        "delimiter": None,
        "extraction_method": None
    }

    df = pd.DataFrame()

    try:
        if ext in [".csv", ".tsv", ".txt"]:
            encoding = "utf-8"
            try:
                sample = file_bytes[:16384].decode("utf-8")
            except UnicodeDecodeError:
                encoding = "latin-1"
                sample = file_bytes[:16384].decode("latin-1")
            meta["encoding"] = encoding
            delimiter = "\t" if ext == ".tsv" else detect_text_delimiter(sample)
            meta["delimiter"] = delimiter
            meta["extraction_method"] = "pandas.read_csv"
            df = pd.read_csv(io.BytesIO(file_bytes), sep=delimiter, encoding=encoding)

        elif ext in [".xlsx", ".xls"]:
            meta["extraction_method"] = "pandas.read_excel"
            excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
            meta["available_sheets"] = excel_file.sheet_names
            target_sheet = sheet_name if (sheet_name and sheet_name in excel_file.sheet_names) else excel_file.sheet_names[0]
            meta["selected_sheet"] = target_sheet
            df = pd.read_excel(excel_file, sheet_name=target_sheet)

        elif ext == ".json":
            meta["extraction_method"] = "pandas.read_json"
            try:
                df = pd.read_json(io.BytesIO(file_bytes))
            except ValueError:
                # If nested object structure
                data = json.loads(file_bytes.decode("utf-8", errors="replace"))
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                elif isinstance(data, dict):
                    # Find candidate list in keys
                    candidate_key = next((k for k, v in data.items() if isinstance(v, list)), None)
                    if candidate_key:
                        df = pd.json_normalize(data[candidate_key])
                    else:
                        df = pd.DataFrame([data])
                else:
                    raise ValueError("JSON file does not contain a valid record array or object.")

        elif ext == ".jsonl":
            meta["extraction_method"] = "pandas.read_json(lines=True)"
            df = pd.read_json(io.BytesIO(file_bytes), lines=True)

        elif ext == ".parquet":
            meta["extraction_method"] = "pandas.read_parquet"
            df = pd.read_parquet(io.BytesIO(file_bytes))

        elif ext == ".xml":
            meta["extraction_method"] = "pandas.read_xml"
            try:
                df = pd.read_xml(io.BytesIO(file_bytes))
            except Exception as e:
                # Fallback to python xml parser
                import xml.etree.ElementTree as ET
                root = ET.fromstring(file_bytes.decode("utf-8", errors="replace"))
                records = []
                for child in root:
                    row = {sub.tag: sub.text for sub in child}
                    if row:
                        records.append(row)
                if records:
                    df = pd.DataFrame(records)
                else:
                    raise ValueError(f"Could not extract tabular nodes from XML: {e}")

        elif ext in [".sqlite", ".db"]:
            meta["extraction_method"] = "sqlite3.connect"
            # Read in-memory sqlite
            temp_db = sqlite3.connect(":memory:")
            # Write bytes to temp sqlite memory
            source_con = sqlite3.connect(io.BytesIO(file_bytes).getvalue()) if hasattr(io.BytesIO, "getvalue") else None
            # Standard approach: write to temp memory using sqlite3 backup or script
            # Since sqlite expects a file path, we inspect master table
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                tf.write(file_bytes)
                temp_path = tf.name
            try:
                conn = sqlite3.connect(temp_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
                tables = [r[0] for r in cursor.fetchall()]
                meta["available_tables"] = tables
                if not tables:
                    raise ValueError("SQLite database contains no user tables.")
                target_table = table_name if (table_name and table_name in tables) else tables[0]
                meta["selected_table"] = target_table
                df = pd.read_sql_query(f"SELECT * FROM `{target_table}`", conn)
                conn.close()
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        elif ext == ".sql":
            meta["extraction_method"] = "sqlite_script_replay"
            # Execute dump into in-memory SQLite and select first resulting table
            conn = sqlite3.connect(":memory:")
            sql_text = file_bytes.decode("utf-8", errors="replace")
            conn.executescript(sql_text)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [r[0] for r in cursor.fetchall()]
            meta["available_tables"] = tables
            if not tables:
                raise ValueError("SQL dump did not create any detectable tables.")
            target_table = table_name if (table_name and table_name in tables) else tables[0]
            meta["selected_table"] = target_table
            df = pd.read_sql_query(f"SELECT * FROM `{target_table}`", conn)
            conn.close()

        elif ext == ".pdf":
            meta["extraction_method"] = "pymupdf_table_extraction"
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            all_tables = []
            for page in doc:
                tables = page.find_tables()
                for table in tables:
                    table_df = table.to_pandas()
                    if not table_df.empty:
                        all_tables.append(table_df)
            if all_tables:
                df = pd.concat(all_tables, ignore_index=True)
            else:
                raise ValueError("No tabular data detected in PDF document.")

        elif ext == ".docx":
            meta["extraction_method"] = "docx_table_extraction"
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            all_rows = []
            for table in doc.tables:
                headers = [cell.text.strip() for cell in table.rows[0].cells] if table.rows else []
                for row in table.rows[1:]:
                    values = [cell.text.strip() for cell in row.cells]
                    if len(values) == len(headers):
                        all_rows.append(dict(zip(headers, values)))
            if all_rows:
                df = pd.DataFrame(all_rows)
            else:
                raise ValueError("No structured tables detected in DOCX document.")

    except Exception as e:
        raise ValueError(f"Failed to parse {ext} file '{file_name}': {str(e)}")

    # Clean empty rows / sanitize columns
    df.columns = [str(col).strip() for col in df.columns]
    meta["rows"] = len(df)
    meta["columns"] = len(df.columns)
    meta["column_names"] = list(df.columns)
    return df, meta
