from __future__ import annotations

import unittest

import pandas as pd

from src.etl.execution_engine import execute_plan
from src.etl.sql_metadata import SQLColumnMapping, validate_sql_plan
from src.etl.sql_optimizer import (
    analyze_sql_plan,
    benchmark_sql_candidate,
    compare_sql_outputs,
    generate_sql_candidates,
)


class SqlOptimizerTests(unittest.TestCase):
    def test_internal_names_are_unique_for_duplicate_and_reserved_headers(self) -> None:
        frame = pd.DataFrame([[1, 2, 3]], columns=["select", "a b", "a b"])
        mapping = SQLColumnMapping.from_dataframe(frame)
        self.assertEqual(len(set(mapping.internal_names)), 3)
        self.assertTrue(all(name.startswith("etl_col_") for name in mapping.internal_names))
        self.assertEqual(
            [column.mapping_status for column in mapping.columns],
            ["unique", "ambiguous_source_label", "ambiguous_source_label"],
        )
        with self.assertRaisesRegex(ValueError, "duplicate source labels"):
            validate_sql_plan(
                frame,
                {"steps": [{"operation": "clean_text", "columns": ["a b"]}]},
            )

    def test_rejects_raw_sql_and_self_dependency(self) -> None:
        frame = pd.DataFrame({"value": [1]})
        with self.assertRaisesRegex(ValueError, "Raw SQL"):
            validate_sql_plan(frame, {"steps": [{"operation": "validate_dataset", "sql": "DROP TABLE x"}]})
        with self.assertRaisesRegex(ValueError, "unknown or forward"):
            validate_sql_plan(
                frame,
                {
                    "steps": [
                        {
                            "id": "self",
                            "operation": "validate_dataset",
                            "dependencies": ["self"],
                        }
                    ]
                },
            )

    def test_sql_mapping_preserves_types_and_restores_original_labels(self) -> None:
        frame = pd.DataFrame({"Odd Name": [1, 2], "select": [1.5, 2.5]})
        output, _ = execute_plan(
            frame,
            {"steps": [{"operation": "validate_dataset"}]},
            mode="sql",
        )
        self.assertEqual(list(output.columns), ["Odd Name", "select"])
        self.assertEqual(str(output["Odd Name"].dtype), "int64")
        self.assertEqual(str(output["select"].dtype), "float64")

    def test_mixed_scalar_column_uses_no_coercion_affinity(self) -> None:
        frame = pd.DataFrame({"mixed": pd.Series([7, "7"], dtype="object")})
        mapping = SQLColumnMapping.from_dataframe(frame)
        self.assertEqual(mapping.columns[0].sqlite_type, "BLOB")
        output, _ = execute_plan(
            frame,
            {"steps": [{"operation": "validate_dataset"}]},
            mode="sql",
        )
        self.assertEqual(type(output.iloc[0, 0]), int)
        self.assertEqual(type(output.iloc[1, 0]), str)

    def test_sql_round_trip_preserves_nullable_integer_dtype_and_nulls(self) -> None:
        frame = pd.DataFrame(
            {"value": pd.Series([1, pd.NA], dtype="Int64")}
        )
        output, _ = execute_plan(
            frame,
            {"steps": [{"operation": "validate_dataset"}]},
            mode="sql",
        )
        self.assertEqual(str(output["value"].dtype), "Int64")
        self.assertEqual(output["value"].iloc[0], 1)
        self.assertTrue(pd.isna(output["value"].iloc[1]))

    def test_normalize_columns_changes_labels_not_values(self) -> None:
        frame = pd.DataFrame({" Customer Name ": ["  Ada  "]})
        output, _ = execute_plan(
            frame,
            {"steps": [{"operation": "normalize_columns"}]},
            mode="sql",
        )
        self.assertEqual(list(output.columns), ["_customer_name_"])
        self.assertEqual(output.iloc[0, 0], "  Ada  ")

    def test_explain_plan_reports_evidence_without_creating_indexes(self) -> None:
        frame = pd.DataFrame({"value": [1, 1, 2]})
        analysis = analyze_sql_plan(
            frame,
            {"steps": [{"operation": "drop_duplicates", "columns": ["value"]}]},
        )
        self.assertEqual(analysis["engine"], "SQLite")
        self.assertTrue(analysis["operations"][0]["plan"])
        self.assertEqual(analysis["recommendations"][0]["status"], "not_recommended")
        self.assertIn("not a runtime measurement", analysis["evidence_note"])

    def test_candidate_removes_only_known_validation_noop_and_benchmarks_equivalently(self) -> None:
        frame = pd.DataFrame({"value": [1, 2, 2, None]})
        baseline = {
            "steps": [
                {
                    "id": "validation",
                    "operation": "validate_dataset",
                    "columns": ["value"],
                },
                {
                    "id": "clean",
                    "operation": "clean_text",
                    "columns": ["value"],
                    "dependencies": ["validation"],
                },
            ]
        }
        candidates = generate_sql_candidates(baseline)
        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0]["approved"])
        candidate_plan = candidates[0]["plan"]
        self.assertEqual(len(candidate_plan["steps"]), 1)
        self.assertNotIn("dependencies", candidate_plan["steps"][0])
        report = benchmark_sql_candidate(frame, baseline, candidate_plan, repeats=3)
        self.assertIn(
            report["status"],
            {
                "measured_improvement",
                "measured_regression",
                "inconclusive",
                "rejected_incorrect",
            },
        )
        self.assertTrue(report["equivalence"]["equivalent"])
        self.assertEqual(report["repeats"], 3)

    def test_equivalence_preserves_duplicate_multiplicity_nulls_order_and_types(self) -> None:
        baseline = pd.DataFrame({"value": pd.Series([1, 1, None], dtype="float64")})
        reordered = pd.DataFrame({"value": pd.Series([None, 1, 1], dtype="float64")})
        fewer_duplicates = pd.DataFrame({"value": pd.Series([None, 1], dtype="float64")})
        self.assertTrue(compare_sql_outputs(baseline, reordered)["equivalent"])
        self.assertFalse(compare_sql_outputs(baseline, fewer_duplicates)["equivalent"])
        wrong_type = pd.DataFrame({"value": pd.Series(["1", "1", None], dtype="object")})
        self.assertFalse(compare_sql_outputs(baseline, wrong_type)["equivalent"])


if __name__ == "__main__":
    unittest.main()
