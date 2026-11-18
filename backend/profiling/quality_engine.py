"""
Deterministic Data Quality Metrics and Profiling Engine.
Evaluates dataset completeness, uniqueness, validity, consistency, integrity proxies,
statistical profiles, and computes an overall 0-100 weighted quality score.
Does NOT rely on LLM for numerical metrics.
"""

import re
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Configurable default metric weights
DEFAULT_WEIGHTS = {
    "completeness": 0.25,
    "uniqueness": 0.20,
    "validity": 0.20,
    "consistency": 0.15,
    "integrity": 0.20
}

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
DATE_REGEX = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d{1,2}[-/]\d{1,2}[-/]\d{4}$")

def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates column-level statistical and structural metadata."""
    col_profiles = {}
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        null_count = int(series.isna().sum())
        total_count = len(series)
        unique_count = int(series.nunique(dropna=True))

        dtype_str = str(series.dtype)
        is_numeric = pd.api.types.is_numeric_dtype(series)

        profile: Dict[str, Any] = {
            "name": col,
            "dtype": dtype_str,
            "total_count": total_count,
            "null_count": null_count,
            "null_percentage": round((null_count / total_count * 100), 2) if total_count > 0 else 0.0,
            "unique_count": unique_count,
            "unique_percentage": round((unique_count / total_count * 100), 2) if total_count > 0 else 0.0,
            "sample_values": [str(v) for v in non_null.head(5).tolist()]
        }

        if is_numeric and len(non_null) > 0:
            numeric_vals = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(numeric_vals) > 0:
                profile.update({
                    "mean": round(float(numeric_vals.mean()), 3),
                    "std": round(float(numeric_vals.std()), 3) if len(numeric_vals) > 1 else 0.0,
                    "median": round(float(numeric_vals.median()), 3),
                    "min": round(float(numeric_vals.min()), 3),
                    "max": round(float(numeric_vals.max()), 3),
                    "q25": round(float(numeric_vals.quantile(0.25)), 3),
                    "q75": round(float(numeric_vals.quantile(0.75)), 3),
                    "zero_count": int((numeric_vals == 0).sum()),
                    "negative_count": int((numeric_vals < 0).sum())
                })
        col_profiles[col] = profile

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": col_profiles
    }

def calculate_quality_metrics(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Computes deterministic data quality metrics across:
    1. Completeness
    2. Uniqueness
    3. Validity
    4. Consistency
    5. Integrity Proxy
    Returns individual component scores (0-100) and weighted overall score.
    """
    w = weights or DEFAULT_WEIGHTS
    total_cells = df.size
    total_rows = len(df)
    total_cols = len(df.columns)

    if total_rows == 0 or total_cols == 0:
        return {
            "overall_score": 0.0,
            "completeness": {"score": 0.0, "total_missing": 0, "missing_percent": 100.0, "complete_rows": 0},
            "uniqueness": {"score": 0.0, "duplicate_rows": 0, "duplicate_percent": 0.0},
            "validity": {"score": 0.0, "invalid_records": 0, "checked_fields": 0},
            "consistency": {"score": 0.0, "inconsistent_formats": 0},
            "integrity": {"score": 0.0, "null_propagation_risk": 0},
            "weights": w
        }

    # 1. Completeness
    total_missing = int(df.isna().sum().sum())
    missing_percent = (total_missing / total_cells) * 100 if total_cells > 0 else 0.0
    complete_rows = int(df.dropna().shape[0])
    completeness_score = max(0.0, round(100.0 - missing_percent, 2))

    # 2. Uniqueness
    duplicate_rows = int(df.duplicated().sum())
    duplicate_percent = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0.0
    uniqueness_score = max(0.0, round(100.0 - duplicate_percent, 2))

    # 3. Validity (Checks email syntax, negative amounts where prohibited, date format plausibility)
    validity_checks = 0
    validity_violations = 0
    for col in df.columns:
        series = df[col].dropna()
        col_lower = col.lower()
        if "email" in col_lower:
            validity_checks += len(series)
            invalid_emails = sum(1 for val in series if not EMAIL_REGEX.match(str(val).strip()))
            validity_violations += invalid_emails
        elif "date" in col_lower:
            validity_checks += len(series)
            invalid_dates = 0
            for val in series:
                try:
                    pd.to_datetime(val, errors="raise")
                except Exception:
                    invalid_dates += 1
            validity_violations += invalid_dates
        elif any(term in col_lower for term in ["price", "amount", "salary", "age", "quantity"]):
            numeric_series = pd.to_numeric(series, errors="coerce")
            validity_checks += len(series)
            invalid_numeric = int(numeric_series.isna().sum()) + int((numeric_series < 0).sum())
            validity_violations += invalid_numeric

    if validity_checks > 0:
        validity_score = max(0.0, round(100.0 - (validity_violations / validity_checks * 100.0), 2))
    else:
        validity_score = 100.0

    # 4. Consistency (Data type uniformity and column naming consistency)
    consistency_violations = 0
    for col in df.columns:
        # Check mixed types in object column
        if df[col].dtype == object:
            types_in_col = set(type(x) for x in df[col].dropna())
            if len(types_in_col) > 1:
                consistency_violations += 1
        # Check whitespace pollution
        if df[col].dtype == object:
            has_edge_spaces = df[col].dropna().apply(lambda s: str(s) != str(s).strip()).any()
            if has_edge_spaces:
                consistency_violations += 1

    consistency_score = max(0.0, round(100.0 - (consistency_violations / max(1, total_cols * 2) * 100.0), 2))

    # 5. Integrity (Preservation of schema identifiers and null propagation risks)
    candidate_id_cols = [c for c in df.columns if any(k in c.lower() for k in ["id", "code", "key", "num"])]
    integrity_penalties = 0
    if candidate_id_cols:
        for id_col in candidate_id_cols:
            if df[id_col].isna().any():
                integrity_penalties += 20
    integrity_score = max(0.0, round(100.0 - integrity_penalties, 2))

    # Overall Weighted Score
    overall_score = round(
        (completeness_score * w.get("completeness", 0.25)) +
        (uniqueness_score * w.get("uniqueness", 0.20)) +
        (validity_score * w.get("validity", 0.20)) +
        (consistency_score * w.get("consistency", 0.15)) +
        (integrity_score * w.get("integrity", 0.20)),
        2
    )

    return {
        "overall_score": overall_score,
        "completeness": {
            "score": completeness_score,
            "total_missing": total_missing,
            "missing_percent": round(missing_percent, 2),
            "complete_rows": complete_rows,
            "weight": w.get("completeness", 0.25)
        },
        "uniqueness": {
            "score": uniqueness_score,
            "duplicate_rows": duplicate_rows,
            "duplicate_percent": round(duplicate_percent, 2),
            "weight": w.get("uniqueness", 0.20)
        },
        "validity": {
            "score": validity_score,
            "checks_evaluated": validity_checks,
            "violations_found": validity_violations,
            "weight": w.get("validity", 0.20)
        },
        "consistency": {
            "score": consistency_score,
            "inconsistencies_detected": consistency_violations,
            "weight": w.get("consistency", 0.15)
        },
        "integrity": {
            "score": integrity_score,
            "candidate_key_columns": candidate_id_cols,
            "weight": w.get("integrity", 0.20)
        },
        "weights": w
    }
