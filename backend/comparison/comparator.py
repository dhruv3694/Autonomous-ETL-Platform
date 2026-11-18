"""
Comprehensive Before-versus-After Data Comparison Engine.
Calculates dataset dimension differences, missing value reductions, duplicate removal rates,
schema evolutions, statistical shift, and quality score improvements.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

def compare_datasets(original_df: pd.DataFrame, transformed_df: pd.DataFrame, orig_metrics: Dict[str, Any], trans_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs multi-dimensional comparison between original and transformed datasets.
    """
    orig_rows = len(original_df)
    trans_rows = len(transformed_df)
    orig_cols = len(original_df.columns)
    trans_cols = len(transformed_df.columns)

    # 1. Dataset Overview
    overview = {
        "original_rows": orig_rows,
        "transformed_rows": trans_rows,
        "row_difference": trans_rows - orig_rows,
        "rows_removed": max(0, orig_rows - trans_rows),
        "original_columns": orig_cols,
        "transformed_columns": trans_cols,
        "column_difference": trans_cols - orig_cols,
        "columns_added": [c for c in transformed_df.columns if c not in original_df.columns],
        "columns_removed": [c for c in original_df.columns if c not in transformed_df.columns]
    }

    # 2. Quality Score Improvement
    orig_score = orig_metrics.get("overall_score", 0.0)
    trans_score = trans_metrics.get("overall_score", 0.0)
    score_diff = round(trans_score - orig_score, 2)

    quality_comparison = {
        "original_score": orig_score,
        "transformed_score": trans_score,
        "improvement": score_diff,
        "verdict": "improved" if score_diff > 0 else ("deteriorated" if score_diff < 0 else "neutral"),
        "completeness_delta": round(trans_metrics["completeness"]["score"] - orig_metrics["completeness"]["score"], 2),
        "uniqueness_delta": round(trans_metrics["uniqueness"]["score"] - orig_metrics["uniqueness"]["score"], 2),
        "validity_delta": round(trans_metrics["validity"]["score"] - orig_metrics["validity"]["score"], 2),
        "consistency_delta": round(trans_metrics["consistency"]["score"] - orig_metrics["consistency"]["score"], 2),
        "integrity_delta": round(trans_metrics["integrity"]["score"] - orig_metrics["integrity"]["score"], 2)
    }

    # 3. Missing Value Analysis
    missing_analysis = {
        "original_missing": orig_metrics["completeness"]["total_missing"],
        "transformed_missing": trans_metrics["completeness"]["total_missing"],
        "missing_values_resolved": max(0, orig_metrics["completeness"]["total_missing"] - trans_metrics["completeness"]["total_missing"])
    }

    # 4. Duplicate Analysis
    duplicate_analysis = {
        "original_duplicates": orig_metrics["uniqueness"]["duplicate_rows"],
        "transformed_duplicates": trans_metrics["uniqueness"]["duplicate_rows"],
        "duplicates_removed": max(0, orig_metrics["uniqueness"]["duplicate_rows"] - trans_metrics["uniqueness"]["duplicate_rows"])
    }

    # 5. Potential Data Loss Warnings
    warnings = []
    if orig_rows > 0 and (orig_rows - trans_rows) / orig_rows > 0.3:
        warnings.append(f"Significant row reduction: {orig_rows - trans_rows} rows ({(orig_rows - trans_rows)/orig_rows*100:.1f}%) were dropped.")
    if len(overview["columns_removed"]) > 0:
        warnings.append(f"Columns were eliminated: {', '.join(overview['columns_removed'])}")
    if score_diff < 0:
        warnings.append("Transformed dataset has a lower overall data quality score than the input.")

    return {
        "overview": overview,
        "quality_comparison": quality_comparison,
        "missing_analysis": missing_analysis,
        "duplicate_analysis": duplicate_analysis,
        "warnings": warnings
    }
