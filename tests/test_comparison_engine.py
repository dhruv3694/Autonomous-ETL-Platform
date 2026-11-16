from __future__ import annotations

import pandas as pd
import pytest

from src.etl.comparison_engine import (
    compare_datasets,
    compute_data_quality_indicators,
    detect_record_level_changes,
    analyze_transformation_impact,
    validate_user_requirements,
    validate_pipeline_equivalence,
    generate_markdown_comparison_report,
)


def _sample_datasets():
    before = pd.DataFrame({
        "id": [1, 2, 2, 3, 4],
        "name": [" Alice ", "Bob", "Bob", "Charlie", None],
        "score": [10.5, 20.0, 20.0, None, 40.0],
        "created_at": ["2025/01/01", "2025/01/02", "2025/01/02", "2025-01-03", "2025-01-04"],
    })
    after = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "unknown"],
        "score": [10.5, 20.0, 20.0, 40.0],
        "created_at": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
    })
    return before, after


def test_compare_datasets_detects_row_col_null_dupe_changes():
    before, after = _sample_datasets()
    comp = compare_datasets(before, after, dataset_name="users")

    assert comp["dataset_name"] == "users"
    assert comp["row_metrics"]["before_rows"] == 5
    assert comp["row_metrics"]["after_rows"] == 4
    assert comp["row_metrics"]["duplicates_removed"] == 1
    assert comp["row_metrics"]["before_duplicates"] == 1
    assert comp["row_metrics"]["after_duplicates"] == 0

    # Data quality improvements
    dq = comp["data_quality"]
    assert dq["after"]["uniqueness_score"] == 100.0
    assert dq["after"]["completeness_score"] == 100.0
    assert dq["before"]["completeness_score"] < 100.0
    assert len(dq["improvements"]) > 0


def test_detect_record_level_changes_with_key():
    before, after = _sample_datasets()
    # Deduplicate before to have a clean key for testing key-based strategy
    before_dedup = before.drop_duplicates(subset=["id"]).reset_index(drop=True)
    res = detect_record_level_changes(before_dedup, after, key_columns=["id"])

    assert res["strategy"] == "key_based"
    assert res["key_columns"] == ["id"]
    assert res["counts"]["total_before"] == 4
    assert res["counts"]["total_after"] == 4
    # Names ' Alice ' -> 'Alice' and None -> 'unknown' are modified
    assert res["counts"]["modified"] >= 2


def test_detect_record_level_changes_multiset_fallback():
    # Dataset without unique identifiers
    df1 = pd.DataFrame({"item": ["apple", "banana", "banana", "cherry"]})
    # Reordered and one duplicate removed
    df2 = pd.DataFrame({"item": ["cherry", "banana", "apple"]})

    res = detect_record_level_changes(df1, df2)
    assert res["strategy"] == "row_fingerprint_multiset"
    assert res["counts"]["unchanged"] == 3
    assert res["counts"]["removed"] == 1
    assert res["counts"]["added"] == 0


def test_analyze_transformation_impact():
    audit = [
        {
            "step_id": "op_1",
            "step": "Remove duplicates",
            "operation": "drop_duplicates",
            "input_row_count": 5,
            "output_row_count": 4,
            "affected_columns": ["id", "name"],
            "performance": {"duration_seconds": 0.005},
            "execution_status": "succeeded",
        }
    ]
    plan = {
        "operations": [
            {"id": "op_1", "operation": "drop_duplicates", "columns": ["id", "name"]}
        ]
    }
    impacts = analyze_transformation_impact(audit, plan)
    assert len(impacts) == 1
    assert impacts[0]["operation_id"] == "op_1"
    assert impacts[0]["rows_affected"] == 1
    assert impacts[0]["expected"] is True
    assert "removed 1 duplicate records" in impacts[0]["explanation"]


def test_validate_user_requirements():
    instruction = "Remove duplicate records, handle missing values, and standardize date formats."
    plan = {"operations": [{"id": "1", "operation": "drop_duplicates"}, {"id": "2", "operation": "fill_missing"}]}
    audit = [
        {"operation": "drop_duplicates"},
        {"operation": "fill_missing"},
    ]
    final_df = pd.DataFrame({"a": [1, 2], "b": [10.0, 20.0]})
    validation_results = {"status": "passed"}

    res = validate_user_requirements(instruction, plan, final_df, validation_results, audit)
    reqs = {r["requirement"]: r["status"] for r in res["requirements"]}
    assert reqs["Remove duplicate records"] == "fulfilled"
    assert reqs["Handle missing values"] == "fulfilled"
    # Date standardization was requested in text but not in audit
    assert reqs["Standardize date formats"] == "not_fulfilled"


def test_validate_pipeline_equivalence_identical_and_reordered():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    # Identical
    eq1 = validate_pipeline_equivalence(df1, df1.copy())
    assert eq1["is_equivalent"] is True
    assert eq1["status"] in {"EQUIVALENT", "equivalent"}

    # Reordered rows (relational equivalence without required ordering)
    df2 = pd.DataFrame({"a": [3, 1, 2], "b": [30.0, 10.0, 20.0]})
    eq2 = validate_pipeline_equivalence(df1, df2, require_row_ordering=False)
    assert eq2["is_equivalent"] is True

    # When row ordering is strictly required
    eq3 = validate_pipeline_equivalence(df1, df2, require_row_ordering=True)
    assert eq3["is_equivalent"] is False
    assert "ordering differ" in eq3["reasons"][0]


def test_validate_pipeline_equivalence_detects_discrepancies():
    df1 = pd.DataFrame({"a": [1, 2], "b": [10.0, 20.0]})

    # Different column name
    df2 = pd.DataFrame({"a": [1, 2], "c": [10.0, 20.0]})
    assert validate_pipeline_equivalence(df1, df2)["is_equivalent"] is False

    # Different values
    df3 = pd.DataFrame({"a": [1, 2], "b": [10.0, 25.0]})
    assert validate_pipeline_equivalence(df1, df3)["is_equivalent"] is False

    # Duplicate multiplicity difference
    df4 = pd.DataFrame({"a": [1, 1, 2], "b": [10.0, 10.0, 20.0]})
    assert validate_pipeline_equivalence(df1, df4)["is_equivalent"] is False


def test_generate_markdown_comparison_report():
    before, after = _sample_datasets()
    comp = compare_datasets(before, after, dataset_name="orders.csv")
    rec = detect_record_level_changes(before, after)
    report = generate_markdown_comparison_report(comp, record_changes=rec)
    assert "# ETL Pipeline Comparison & Validation Report: orders.csv" in report
    assert "Original Rows" in report
    assert "Data Quality Indicators" in report
