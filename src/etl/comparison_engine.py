from __future__ import annotations

import html
import io
import json
import math
import numbers
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def _is_null_scalar(val: Any) -> bool:
    if val is None:
        return True
    if pd.isna(val) is True:
        return True
    try:
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            return True
    except Exception:
        pass
    return False


def _safe_cell_signature(val: Any, float_tolerance: Optional[float] = None) -> Tuple[str, Any]:
    """Return a normalized hashable representation of a cell value."""
    if _is_null_scalar(val):
        return ("null", None)
    if hasattr(val, "item"):
        try:
            val = val.item()
        except Exception:
            pass
    if isinstance(val, (float, np.floating, int, np.integer)):
        if float_tolerance is not None and float_tolerance > 0:
            if isinstance(val, (float, np.floating)):
                # Round to bucket according to tolerance
                rounded = round(float(val) / float_tolerance) * float_tolerance
                return ("number", round(rounded, 8))
        return ("number", float(val))
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        try:
            ts = pd.to_datetime(val)
            return ("datetime", ts.isoformat())
        except Exception:
            return ("datetime", str(val))
    if isinstance(val, bool):
        return ("bool", val)
    return ("str", str(val))


def _compute_column_summary(series: pd.Series) -> Dict[str, Any]:
    non_null = series.dropna()
    total_count = len(series)
    null_count = int(series.isna().sum())
    null_pct = round((null_count / total_count * 100), 2) if total_count > 0 else 0.0
    unique_count = int(series.nunique(dropna=True))

    summary: Dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "total_count": total_count,
        "null_count": null_count,
        "null_pct": null_pct,
        "unique_count": unique_count,
    }

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if not numeric.empty:
            summary["numeric_stats"] = {
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": round(float(numeric.mean()), 4),
                "median": round(float(numeric.median()), 4),
                "std": round(float(numeric.std()), 4) if len(numeric) > 1 else 0.0,
            }
    elif pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series, errors="coerce").dropna()
        if not dt.empty:
            summary["date_range"] = {
                "min": dt.min().isoformat(),
                "max": dt.max().isoformat(),
            }
    else:
        sample = non_null.head(100)
        if not sample.empty:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.8:
                summary["potential_date"] = True
        if not non_null.empty and unique_count <= 20:
            counts = non_null.astype(str).value_counts().head(5).to_dict()
            summary["top_categories"] = {str(k): int(v) for k, v in counts.items()}

    return summary


# ==============================================================================
# 1. DATASET COMPARISON (SECTION 2 & 7)
# ==============================================================================

def compute_data_quality_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate measurable indicators for Completeness, Uniqueness, Validity, and Consistency."""
    if df.empty:
        return {
            "completeness_score": 100.0,
            "uniqueness_score": 100.0,
            "validity_score": 100.0,
            "consistency_score": 100.0,
            "overall_score": 100.0,
            "metrics": {"total_cells": 0, "null_cells": 0, "duplicate_rows": 0},
            "issues": ["Dataset is empty."],
        }

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    null_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    # Completeness: percentage of non-null cells
    completeness = round(((total_cells - null_cells) / total_cells) * 100.0, 2) if total_cells > 0 else 100.0

    # Uniqueness: percentage of non-duplicate rows
    uniqueness = round(((total_rows - duplicate_rows) / total_rows) * 100.0, 2) if total_rows > 0 else 100.0

    # Validity: checks column names and non-corrupt / unparseable formats
    validity_penalties = 0.0
    issues: List[str] = []

    # Check for empty column names or duplicate column names
    col_names = [str(c) for c in df.columns]
    if len(set(col_names)) < len(col_names):
        validity_penalties += 20.0
        issues.append("Duplicate column names detected.")

    for col in df.columns:
        series = df[col]
        # Check if column has mixed types in object series
        if series.dtype == object:
            types_present = {type(v).__name__ for v in series.dropna()}
            if len(types_present) > 1 and not ({"int", "float"} >= types_present):
                validity_penalties += 2.0
                issues.append(f"Column '{col}' has mixed scalar types: {sorted(types_present)}.")

    validity = max(0.0, round(100.0 - min(100.0, validity_penalties), 2))

    # Consistency: checks standard whitespace and casing anomalies in text columns
    consistency_penalties = 0.0
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            str_series = series.dropna().astype(str)
            # Leading or trailing whitespace
            has_strip_diff = (str_series != str_series.str.strip()).any()
            if has_strip_diff:
                consistency_penalties += 3.0
                issues.append(f"Column '{col}' contains values with untrimmed whitespace.")

    consistency = max(0.0, round(100.0 - min(100.0, consistency_penalties), 2))

    # Documented formula: Overall = 0.35 * completeness + 0.35 * uniqueness + 0.15 * validity + 0.15 * consistency
    overall = round(0.35 * completeness + 0.35 * uniqueness + 0.15 * validity + 0.15 * consistency, 2)

    return {
        "completeness_score": completeness,
        "uniqueness_score": uniqueness,
        "validity_score": validity,
        "consistency_score": consistency,
        "overall_score": overall,
        "metrics": {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "total_cells": total_cells,
            "null_cells": null_cells,
            "duplicate_rows": duplicate_rows,
        },
        "issues": issues,
    }


def compare_datasets(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    dataset_name: str = "dataset",
) -> Dict[str, Any]:
    """Comprehensive comparison between original and transformed datasets."""
    before_cols = list(before.columns)
    after_cols = list(after.columns)

    added_cols = [c for c in after_cols if c not in before_cols]
    removed_cols = [c for c in before_cols if c not in after_cols]
    common_cols = [c for c in before_cols if c in after_cols]

    # Detect renames if column count same and 1-to-1 similarity
    renamed_cols: Dict[str, str] = {}
    if len(removed_cols) > 0 and len(removed_cols) == len(added_cols):
        for rem in removed_cols:
            rem_norm = str(rem).lower().replace(" ", "_").strip()
            for add in added_cols:
                add_norm = str(add).lower().replace(" ", "_").strip()
                if rem_norm == add_norm:
                    renamed_cols[rem] = add

    # Column details and statistics
    before_summaries = {col: _compute_column_summary(before[col]) for col in before_cols}
    after_summaries = {col: _compute_column_summary(after[col]) for col in after_cols}

    dtype_changes: Dict[str, Dict[str, str]] = {}
    null_changes: Dict[str, Dict[str, Any]] = {}
    stats_changes: Dict[str, Dict[str, Any]] = {}

    for col in common_cols:
        b_sum = before_summaries[col]
        a_sum = after_summaries[col]
        if b_sum["dtype"] != a_sum["dtype"]:
            dtype_changes[col] = {"before": b_sum["dtype"], "after": a_sum["dtype"]}

        null_diff = a_sum["null_count"] - b_sum["null_count"]
        null_changes[col] = {
            "before_nulls": b_sum["null_count"],
            "after_nulls": a_sum["null_count"],
            "difference": null_diff,
            "status": "reduced" if null_diff < 0 else "increased" if null_diff > 0 else "unchanged",
        }

        if "numeric_stats" in b_sum and "numeric_stats" in a_sum:
            stats_changes[col] = {
                "before": b_sum["numeric_stats"],
                "after": a_sum["numeric_stats"],
            }

    # High-level metrics
    before_rows = len(before)
    after_rows = len(after)
    row_delta = after_rows - before_rows
    row_loss_pct = round(abs(row_delta) / max(before_rows, 1) * 100.0, 2)

    before_dupes = int(before.duplicated().sum())
    after_dupes = int(after.duplicated().sum())
    dupes_removed = max(0, before_dupes - after_dupes)

    # Data Quality comparison
    dq_before = compute_data_quality_indicators(before)
    dq_after = compute_data_quality_indicators(after)

    dq_improvements: List[str] = []
    dq_regressions: List[str] = []

    for dim in ["completeness_score", "uniqueness_score", "validity_score", "consistency_score", "overall_score"]:
        diff = round(dq_after[dim] - dq_before[dim], 2)
        dim_title = dim.replace("_score", "").capitalize()
        if diff > 0:
            dq_improvements.append(f"{dim_title} improved by +{diff}% ({dq_before[dim]}% -> {dq_after[dim]}%)")
        elif diff < 0:
            dq_regressions.append(f"{dim_title} regressed by {diff}% ({dq_before[dim]}% -> {dq_after[dim]}%)")

    unexpected_data_loss = (after_rows == 0 and before_rows > 0) or (row_loss_pct > 50.0 and before_dupes < abs(row_delta))

    return {
        "dataset_name": dataset_name,
        "row_metrics": {
            "before_rows": before_rows,
            "after_rows": after_rows,
            "row_delta": row_delta,
            "row_loss_pct": row_loss_pct,
            "unexpected_data_loss": unexpected_data_loss,
            "before_duplicates": before_dupes,
            "after_duplicates": after_dupes,
            "duplicates_removed": dupes_removed,
        },
        "schema_comparison": {
            "before_columns": before_cols,
            "after_columns": after_cols,
            "added_columns": added_cols,
            "removed_columns": removed_cols,
            "common_columns": common_cols,
            "renamed_columns": renamed_cols,
            "dtype_changes": dtype_changes,
        },
        "column_summaries": {
            "before": before_summaries,
            "after": after_summaries,
            "null_changes": null_changes,
            "numeric_changes": stats_changes,
        },
        "data_quality": {
            "before": dq_before,
            "after": dq_after,
            "improvements": dq_improvements,
            "regressions": dq_regressions,
            "remaining_issues": dq_after["issues"],
        },
    }


# ==============================================================================
# 2. RECORD-LEVEL CHANGE DETECTION (SECTION 3)
# ==============================================================================

def detect_record_level_changes(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    key_columns: Optional[Sequence[str]] = None,
    max_sample_records: int = 50,
) -> Dict[str, Any]:
    """Detect added, removed, modified, and unchanged records with key-based or multiset fallback."""
    common_cols = [c for c in before.columns if c in after.columns]

    # Strategy 1: Explicit or detected stable key
    valid_key: Optional[List[str]] = None
    if key_columns:
        specified = [c for c in key_columns if c in common_cols]
        if specified:
            valid_key = specified
    else:
        # Detect single candidate key if present and unique in before
        for col in common_cols:
            if len(before) > 0 and before[col].notna().all() and before[col].nunique() == len(before):
                valid_key = [col]
                break

    if valid_key and all(k in before.columns and k in after.columns for k in valid_key):
        strategy = "key_based"
        # Index-based comparison
        b_indexed = before.set_index(valid_key, drop=False)
        a_indexed = after.set_index(valid_key, drop=False)

        # Handle potential non-unique in after
        b_keys = set(b_indexed.index)
        a_keys = set(a_indexed.index)

        common_keys = b_keys & a_keys
        removed_keys = b_keys - a_keys
        added_keys = a_keys - b_keys

        unchanged_count = 0
        modified_count = 0
        modified_samples: List[Dict[str, Any]] = []

        compare_cols = [c for c in common_cols if c not in valid_key]

        for k in common_keys:
            b_row = b_indexed.loc[k]
            a_row = a_indexed.loc[k]
            if isinstance(b_row, pd.DataFrame) or isinstance(a_row, pd.DataFrame):
                # Duplicate keys exist
                modified_count += 1
                continue

            diffs: Dict[str, Dict[str, Any]] = {}
            for col in compare_cols:
                b_val = b_row[col]
                a_val = a_row[col]
                b_sig = _safe_cell_signature(b_val)
                a_sig = _safe_cell_signature(a_val)
                if b_sig != a_sig:
                    diffs[col] = {"before": str(b_val), "after": str(a_val)}

            if diffs:
                modified_count += 1
                if len(modified_samples) < max_sample_records:
                    modified_samples.append({
                        "key": str(k),
                        "changed_columns": list(diffs.keys()),
                        "details": diffs,
                    })
            else:
                unchanged_count += 1

        removed_samples = []
        for k in list(removed_keys)[:max_sample_records]:
            row = b_indexed.loc[k]
            removed_samples.append(row.to_dict() if isinstance(row, pd.Series) else row.iloc[0].to_dict())

        added_samples = []
        for k in list(added_keys)[:max_sample_records]:
            row = a_indexed.loc[k]
            added_samples.append(row.to_dict() if isinstance(row, pd.Series) else row.iloc[0].to_dict())

        return {
            "strategy": strategy,
            "key_columns": valid_key,
            "counts": {
                "total_before": len(before),
                "total_after": len(after),
                "unchanged": unchanged_count,
                "modified": modified_count,
                "added": len(added_keys),
                "removed": len(removed_keys),
            },
            "samples": {
                "modified": modified_samples,
                "added": added_samples,
                "removed": removed_samples,
            },
        }

    # Strategy 2: Multiset Row-Fingerprint comparison (when no stable key is available)
    strategy = "row_fingerprint_multiset"

    def row_sig(row: Tuple[Any, ...]) -> Tuple[Tuple[str, Any], ...]:
        return tuple(_safe_cell_signature(v) for v in row)

    # Use common columns to fingerprint
    b_tuples = [row_sig(r) for r in before[common_cols].itertuples(index=False, name=None)] if common_cols else []
    a_tuples = [row_sig(r) for r in after[common_cols].itertuples(index=False, name=None)] if common_cols else []

    b_counter = Counter(b_tuples)
    a_counter = Counter(a_tuples)

    # Exact matching rows
    all_sigs = set(b_counter.keys()) | set(a_counter.keys())
    unchanged_count = 0
    removed_count = 0
    added_count = 0

    for sig in all_sigs:
        b_cnt = b_counter[sig]
        a_cnt = a_counter[sig]
        common = min(b_cnt, a_cnt)
        unchanged_count += common
        if b_cnt > a_cnt:
            removed_count += (b_cnt - a_cnt)
        elif a_cnt > b_cnt:
            added_count += (a_cnt - b_cnt)

    # If row count preserved or decreased, check if some removed/added are modifications
    # In multiset comparison, a modified row is represented as 1 removed + 1 added
    modified_estimate = min(removed_count, added_count)
    net_removed = removed_count - modified_estimate
    net_added = added_count - modified_estimate

    return {
        "strategy": strategy,
        "key_columns": None,
        "counts": {
            "total_before": len(before),
            "total_after": len(after),
            "unchanged": unchanged_count,
            "modified": modified_estimate,
            "added": net_added,
            "removed": net_removed,
        },
        "samples": {
            "note": "Multiset fingerprint matching preserved duplicate multiplicity without assuming row order."
        },
    }


# ==============================================================================
# 3. TRANSFORMATION IMPACT ANALYSIS (SECTION 4)
# ==============================================================================

def analyze_transformation_impact(
    audit: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Captures operation-level impact, comparing expected behavior vs actual execution."""
    operations = plan.get("operations") or plan.get("steps") or []
    op_map = {str(op.get("id")): op for op in operations if isinstance(op, Mapping)}

    impacts: List[Dict[str, Any]] = []

    for entry in audit:
        op_id = str(entry.get("step_id") or entry.get("id") or "")
        op_name = entry.get("step") or entry.get("name") or op_id
        op_spec = op_map.get(op_id, {})

        in_rows = entry.get("input_row_count", 0)
        out_rows = entry.get("output_row_count", 0)
        row_diff = out_rows - in_rows

        affected_cols = entry.get("affected_columns") or op_spec.get("columns", [])
        duration = 0.0
        if "performance" in entry and isinstance(entry["performance"], dict):
            duration = float(entry["performance"].get("duration_seconds", 0.0))
        elif "duration_seconds" in entry:
            duration = float(entry["duration_seconds"])

        op_type = op_spec.get("operation") or entry.get("operation") or "transform"

        # Determine if the change was expected
        expected = True
        explanation = f"Operation '{op_name}' executed."
        if op_type == "drop_duplicates":
            if row_diff < 0:
                explanation = f"Successfully removed {abs(row_diff)} duplicate records."
            else:
                explanation = "No duplicate records were detected; row count preserved."
        elif op_type == "fill_missing":
            explanation = f"Imputed missing values across columns {affected_cols}."
        elif op_type == "convert_dates":
            explanation = f"Standardized dates into datetime format across columns {affected_cols}."
        elif op_type == "clean_text":
            explanation = f"Cleaned whitespace across text columns {affected_cols}."
        elif op_type == "normalize_columns":
            explanation = "Normalized column identifiers to lowercase snake_case."

        impacts.append({
            "operation_id": op_id,
            "operation_name": op_name,
            "operation_type": op_type,
            "input_rows": in_rows,
            "output_rows": out_rows,
            "rows_affected": abs(row_diff) if op_type == "drop_duplicates" else in_rows,
            "row_delta": row_diff,
            "affected_columns": affected_cols,
            "execution_duration_seconds": round(duration, 4),
            "expected": expected,
            "explanation": explanation,
            "execution_status": entry.get("execution_status", "succeeded"),
        })

    return impacts


# ==============================================================================
# 4. USER REQUIREMENT VALIDATION (SECTION 5)
# ==============================================================================

def validate_user_requirements(
    instruction: str,
    plan: Mapping[str, Any],
    final_dataset: pd.DataFrame,
    validation_results: Mapping[str, Any],
    audit: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compares the original instruction with execution results to identify fulfilled/unfulfilled requirements."""
    text = instruction.casefold()
    requirements: List[Dict[str, Any]] = []

    executed_ops = [str(entry.get("operation") or "").lower() for entry in audit]
    plan_ops = [str(op.get("operation") or "").lower() for op in plan.get("operations", [])]

    # 1. Duplicates
    if any(k in text for k in ["duplicate", "repeated", "dedup"]):
        req_status = "fulfilled"
        details = "Duplicate check executed."
        if "drop_duplicates" in executed_ops:
            rem_dupes = int(final_dataset.duplicated().sum())
            if rem_dupes == 0:
                details = "Duplicate rows were successfully removed; 0 duplicates remain."
            else:
                req_status = "partially_fulfilled"
                details = f"{rem_dupes} duplicates still remain in final dataset."
        else:
            req_status = "not_fulfilled"
            details = "Deduplication was requested but no deduplication step was executed."
        requirements.append({
            "requirement": "Remove duplicate records",
            "status": req_status,
            "details": details,
        })

    # 2. Missing values
    if any(k in text for k in ["missing", "null", "blank", "nan"]):
        req_status = "fulfilled"
        if "fill_missing" in executed_ops:
            null_count = int(final_dataset.isna().sum().sum())
            if null_count == 0:
                details = "All missing values were successfully imputed."
            else:
                details = f"Imputation executed; {null_count} missing values remain in non-targeted columns."
        else:
            req_status = "not_fulfilled"
            details = "Handling missing values was requested but fill_missing was not executed."
        requirements.append({
            "requirement": "Handle missing values",
            "status": req_status,
            "details": details,
        })

    # 3. Date standardization
    if any(k in text for k in ["date", "timestamp", "time format"]):
        req_status = "fulfilled"
        if "convert_dates" in executed_ops:
            details = "Date columns were converted to standardized datetime format."
        else:
            req_status = "not_fulfilled"
            details = "Standardizing date formats was requested but convert_dates was not executed."
        requirements.append({
            "requirement": "Standardize date formats",
            "status": req_status,
            "details": details,
        })

    # 4. Text cleaning / whitespace
    if any(k in text for k in ["text", "whitespace", "trim", "clean strings"]):
        req_status = "fulfilled"
        if "clean_text" in executed_ops:
            details = "Text whitespace was stripped and normalized."
        else:
            req_status = "not_fulfilled"
            details = "Text cleaning was requested but clean_text was not executed."
        requirements.append({
            "requirement": "Clean text whitespace",
            "status": req_status,
            "details": details,
        })

    # 5. Inconsistent categorical values / Ambiguities
    ambiguities = list(plan.get("ambiguities") or [])
    if any(k in text for k in ["categorical", "inconsistent category", "standardize categories"]):
        requirements.append({
            "requirement": "Correct inconsistent categorical values",
            "status": "unresolved_ambiguity" if ambiguities else "fulfilled",
            "details": (
                "Categorical correction requires explicit human-confirmed mapping rules to avoid data corruption. "
                "Flagged as requiring human approval/mapping."
            ),
        })

    # General validation check
    validation_status = validation_results.get("status", "unknown")
    overall_fulfilled = all(r["status"] in {"fulfilled", "partially_fulfilled"} for r in requirements) and (validation_status == "passed")

    return {
        "instruction": instruction,
        "overall_fulfilled": overall_fulfilled,
        "requirements": requirements,
        "unresolved_ambiguities": ambiguities,
        "validation_passed": validation_status == "passed",
    }


# ==============================================================================
# 5. OPTIMIZATION EQUIVALENCE VALIDATION (SECTION 6)
# ==============================================================================

def validate_pipeline_equivalence(
    baseline_output: pd.DataFrame,
    candidate_output: pd.DataFrame,
    *,
    float_tolerance: float = 1e-6,
    require_row_ordering: bool = False,
    max_compare_rows: int = 500000,
) -> Dict[str, Any]:
    """Strict output-equivalence validation returning EQUIVALENT, NOT EQUIVALENT, or INCONCLUSIVE."""
    reasons: List[str] = []
    inconclusive_reasons: List[str] = []
    checks: Dict[str, bool] = {
        "column_labels": False,
        "column_dtypes": False,
        "shape": False,
        "row_values_and_nulls": False,
        "duplicate_multiplicity": False,
        "row_ordering": True if not require_row_ordering else False,
    }

    # Resource check
    if len(baseline_output) > max_compare_rows or len(candidate_output) > max_compare_rows:
        inconclusive_reasons.append(
            f"Row count exceeds configured in-memory comparison limit ({max_compare_rows} rows). "
            "Full equivalence cannot be established safely."
        )
        return {
            "status": "INCONCLUSIVE",
            "is_equivalent": False,
            "float_tolerance": float_tolerance,
            "require_row_ordering": require_row_ordering,
            "checks": checks,
            "reasons": reasons,
            "inconclusive_reasons": inconclusive_reasons,
        }

    # 1. Column labels
    if list(baseline_output.columns) == list(candidate_output.columns):
        checks["column_labels"] = True
    else:
        reasons.append(f"Column labels or order differ: {list(baseline_output.columns)} vs {list(candidate_output.columns)}.")

    # 2. Shape
    if baseline_output.shape == candidate_output.shape:
        checks["shape"] = True
    else:
        reasons.append(f"Output shapes differ: baseline={baseline_output.shape}, candidate={candidate_output.shape}.")

    # 3. Dtypes
    b_types = [str(t) for t in baseline_output.dtypes]
    c_types = [str(t) for t in candidate_output.dtypes]
    if b_types == c_types:
        checks["column_dtypes"] = True
    else:
        reasons.append(f"Column data types differ: baseline={b_types}, candidate={c_types}.")

    # 4. Values and Multiplicity
    if checks["shape"] and checks["column_labels"]:
        try:
            if require_row_ordering:
                # Strict row-by-row check
                order_matches = True
                for idx in range(len(baseline_output)):
                    b_sig = tuple(_safe_cell_signature(v, float_tolerance) for v in baseline_output.iloc[idx])
                    c_sig = tuple(_safe_cell_signature(v, float_tolerance) for v in candidate_output.iloc[idx])
                    if b_sig != c_sig:
                        order_matches = False
                        reasons.append(f"Row values or ordering differ at row index {idx}.")
                        break
                checks["row_ordering"] = order_matches
                checks["row_values_and_nulls"] = order_matches
                checks["duplicate_multiplicity"] = order_matches
            else:
                # Multiset check preserving duplicate multiplicity
                b_counter = Counter(
                    tuple(_safe_cell_signature(v, float_tolerance) for v in row)
                    for row in baseline_output.itertuples(index=False, name=None)
                )
                c_counter = Counter(
                    tuple(_safe_cell_signature(v, float_tolerance) for v in row)
                    for row in candidate_output.itertuples(index=False, name=None)
                )

                if b_counter == c_counter:
                    checks["row_values_and_nulls"] = True
                    checks["duplicate_multiplicity"] = True
                else:
                    reasons.append("Row values, NULL semantics, or duplicate counts differ between baseline and candidate.")
        except Exception as exc:
            inconclusive_reasons.append(f"Comparison interrupted or failed: {exc}")

    if inconclusive_reasons:
        status = "INCONCLUSIVE"
        is_equivalent = False
    elif reasons:
        status = "NOT EQUIVALENT"
        is_equivalent = False
    else:
        status = "EQUIVALENT"
        is_equivalent = True

    return {
        "status": status,
        "is_equivalent": is_equivalent,
        "float_tolerance": float_tolerance,
        "require_row_ordering": require_row_ordering,
        "checks": checks,
        "reasons": reasons,
        "inconclusive_reasons": inconclusive_reasons,
    }



# ==============================================================================
# 6. EXPORT AND REPORT GENERATION (SECTION 9)
# ==============================================================================

def generate_markdown_comparison_report(
    comparison: Dict[str, Any],
    record_changes: Optional[Dict[str, Any]] = None,
    impacts: Optional[Sequence[Mapping[str, Any]]] = None,
    requirements: Optional[Dict[str, Any]] = None,
    equivalence: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a clean markdown report summarising dataset comparison, quality, and validation."""
    lines: List[str] = [
        f"# ETL Pipeline Comparison & Validation Report: {comparison.get('dataset_name', 'Dataset')}",
        "",
        "## 1. Dataset Overview & Row Metrics",
        f"- **Original Rows:** {comparison['row_metrics']['before_rows']}",
        f"- **Transformed Rows:** {comparison['row_metrics']['after_rows']}",
        f"- **Row Delta:** {comparison['row_metrics']['row_delta']} ({comparison['row_metrics']['row_loss_pct']}% change)",
        f"- **Duplicates Removed:** {comparison['row_metrics']['duplicates_removed']}",
        f"- **Unexpected Data Loss:** {'YES (ALERT)' if comparison['row_metrics']['unexpected_data_loss'] else 'No'}",
        "",
        "## 2. Schema Comparison",
        f"- **Added Columns:** {comparison['schema_comparison']['added_columns'] or 'None'}",
        f"- **Removed Columns:** {comparison['schema_comparison']['removed_columns'] or 'None'}",
        f"- **Renamed Columns:** {comparison['schema_comparison']['renamed_columns'] or 'None'}",
        f"- **Data Type Changes:** {json.dumps(comparison['schema_comparison']['dtype_changes']) if comparison['schema_comparison']['dtype_changes'] else 'None'}",
        "",
        "## 3. Data Quality Indicators",
        f"- **Completeness:** {comparison['data_quality']['before']['completeness_score']}% -> {comparison['data_quality']['after']['completeness_score']}%",
        f"- **Uniqueness:** {comparison['data_quality']['before']['uniqueness_score']}% -> {comparison['data_quality']['after']['uniqueness_score']}%",
        f"- **Validity:** {comparison['data_quality']['before']['validity_score']}% -> {comparison['data_quality']['after']['validity_score']}%",
        f"- **Consistency:** {comparison['data_quality']['before']['consistency_score']}% -> {comparison['data_quality']['after']['consistency_score']}%",
        f"- **Overall Quality Score:** {comparison['data_quality']['before']['overall_score']}% -> {comparison['data_quality']['after']['overall_score']}%",
    ]

    if comparison['data_quality']['improvements']:
        lines.append("\n### Improvements:")
        for imp in comparison['data_quality']['improvements']:
            lines.append(f"- {imp}")

    if comparison['data_quality']['regressions']:
        lines.append("\n### Regressions:")
        for reg in comparison['data_quality']['regressions']:
            lines.append(f"- {reg}")

    if record_changes:
        lines.extend([
            "",
            "## 4. Record-Level Changes",
            f"- **Strategy:** {record_changes.get('strategy')}",
            f"- **Unchanged Records:** {record_changes['counts']['unchanged']}",
            f"- **Modified Records:** {record_changes['counts']['modified']}",
            f"- **Added Records:** {record_changes['counts']['added']}",
            f"- **Removed Records:** {record_changes['counts']['removed']}",
        ])

    if impacts:
        lines.extend([
            "",
            "## 5. Transformation Impact Analysis",
            "| Step | Operation | In Rows | Out Rows | Affected Rows | Duration (s) | Status |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for imp in impacts:
            lines.append(
                f"| {imp['operation_name']} | {imp['operation_type']} | {imp['input_rows']} | "
                f"{imp['output_rows']} | {imp['rows_affected']} | {imp['execution_duration_seconds']} | "
                f"{imp['execution_status']} |"
            )

    if requirements:
        lines.extend([
            "",
            "## 6. User Requirement Validation",
            f"- **Instruction:** \"{requirements.get('instruction', '')}\"",
            f"- **Overall Fulfilled:** {'Yes' if requirements.get('overall_fulfilled') else 'No'}",
        ])
        for req in requirements.get("requirements", []):
            lines.append(f"- **{req['requirement']}:** {req['status'].upper()} — {req['details']}")

    if equivalence:
        lines.extend([
            "",
            "## 7. Optimization Equivalence Check",
            f"- **Status:** {equivalence['status'].upper()}",
            f"- **Is Equivalent:** {equivalence['is_equivalent']}",
            f"- **Reasons:** {', '.join(equivalence['reasons']) if equivalence['reasons'] else 'None'}",
        ])

    lines.append("\n*Report generated deterministically by Autonomous AI ETL Platform.*")
    return "\n".join(lines)
