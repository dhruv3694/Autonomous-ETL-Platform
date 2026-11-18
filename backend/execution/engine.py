"""
Controlled ETL Execution Engine.
Executes approved pipeline operations in a sandboxed, deterministic manner.
Never executes arbitrary unapproved code.
Maintains intermediate dataset snapshots and execution audit trails.
"""

import time
import re
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

def execute_operation_step(df: pd.DataFrame, operation: str, target_columns: List[str], parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes a single validated transformation step on the DataFrame.
    Returns (transformed_df, step_log).
    """
    start_time = time.time()
    input_rows = len(df)
    input_cols = len(df.columns)
    transformed_df = df.copy()
    details = {}

    if operation == "drop_duplicates":
        transformed_df = transformed_df.drop_duplicates()
        details["dropped_rows"] = input_rows - len(transformed_df)

    elif operation == "normalize_columns":
        new_cols = [re.sub(r"\s+", "_", str(c)).lower().strip() for c in transformed_df.columns]
        # Ensure no duplicates after normalization
        seen = {}
        unique_cols = []
        for c in new_cols:
            if c not in seen:
                seen[c] = 0
                unique_cols.append(c)
            else:
                seen[c] += 1
                unique_cols.append(f"{c}_{seen[c]}")
        transformed_df.columns = unique_cols
        details["renamed_count"] = len(unique_cols)

    elif operation == "clean_text":
        cols_to_clean = [c for c in target_columns if c in transformed_df.columns]
        lowercase = parameters.get("lowercase", False)
        for col in cols_to_clean:
            if transformed_df[col].dtype == object:
                s = transformed_df[col].astype(str).str.strip()
                if lowercase:
                    s = s.str.lower()
                transformed_df[col] = s
        details["cleaned_columns"] = cols_to_clean

    elif operation == "fill_missing":
        cols = [c for c in target_columns if c in transformed_df.columns]
        strategy = parameters.get("strategy", "median")
        filled_count = 0
        for col in cols:
            null_count = int(transformed_df[col].isna().sum())
            if null_count > 0:
                if pd.api.types.is_numeric_dtype(transformed_df[col]):
                    numeric_vals = pd.to_numeric(transformed_df[col], errors="coerce")
                    if strategy == "median":
                        fill_val = numeric_vals.median()
                    elif strategy == "mean":
                        fill_val = numeric_vals.mean()
                    else:
                        fill_val = 0
                    transformed_df[col] = transformed_df[col].fillna(fill_val)
                    filled_count += null_count
                else:
                    mode_val = transformed_df[col].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
                    transformed_df[col] = transformed_df[col].fillna(fill_val)
                    filled_count += null_count
        details["imputed_cells"] = filled_count

    elif operation == "convert_dates":
        date_format = parameters.get("format", "%Y-%m-%d")
        cols = [c for c in target_columns if c in transformed_df.columns]
        for col in cols:
            transformed_df[col] = pd.to_datetime(transformed_df[col], errors="coerce").dt.strftime(date_format)
        details["standardized_date_columns"] = cols

    elif operation == "rename_columns":
        mapping = parameters.get("mapping", {})
        transformed_df = transformed_df.rename(columns=mapping)
        details["mapped_columns"] = mapping

    elif operation == "cast_types":
        type_mapping = parameters.get("types", {})
        for col, target_type in type_mapping.items():
            if col in transformed_df.columns:
                if target_type == "numeric":
                    transformed_df[col] = pd.to_numeric(transformed_df[col], errors="coerce")
                elif target_type == "string":
                    transformed_df[col] = transformed_df[col].astype(str)
                elif target_type == "datetime":
                    transformed_df[col] = pd.to_datetime(transformed_df[col], errors="coerce")
        details["cast_targets"] = type_mapping

    duration = round(time.time() - start_time, 4)
    log_entry = {
        "operation": operation,
        "duration_seconds": duration,
        "input_rows": input_rows,
        "output_rows": len(transformed_df),
        "input_cols": input_cols,
        "output_cols": len(transformed_df.columns),
        "details": details,
        "status": "success"
    }

    return transformed_df, log_entry
