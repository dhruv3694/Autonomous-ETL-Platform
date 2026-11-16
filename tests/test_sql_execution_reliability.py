from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import pandas as pd

from src.etl.execution_engine import SQLExecutionError, execute_plan
from src.etl.sql_generator import generate_sql_operations as real_generate_sql_operations


class SqlExecutionReliabilityTests(unittest.TestCase):
    def test_failed_sql_step_does_not_return_previous_step_as_success(self) -> None:
        frame = pd.DataFrame({"value": [1, 1, 2]})
        plan = {
            "steps": [
                {
                    "id": "remove-duplicates",
                    "name": "remove duplicates",
                    "operation": "drop_duplicates",
                    "columns": ["value"],
                },
                {"id": "malformed-step", "name": "malformed", "operation": "validate_dataset", "columns": ["value"]},
                {
                    "id": "dependent-step",
                    "name": "dependent copy",
                    "operation": "validate_dataset",
                    "columns": ["value"],
                },
            ]
        }

        mapping, generated = real_generate_sql_operations(frame, plan)
        generated[1] = replace(
            generated[1],
            sql='CREATE TABLE "etl_failure" AS SELECT "missing_column" FROM "etl_source";',
        )
        with patch(
            "src.etl.execution_engine.generate_sql_operations",
            return_value=(mapping, generated),
        ):
            with self.assertRaises(SQLExecutionError) as raised:
                execute_plan(frame, plan, mode="sql")

        error = raised.exception
        self.assertEqual(error.status, "failed")
        self.assertEqual(error.step_id, "malformed-step")
        self.assertEqual(error.operation_type, "validate_dataset")
        self.assertTrue(error.execution_id)
        self.assertTrue(error.timestamp)
        self.assertIn("missing_column", error.sql_context)
        self.assertEqual(len(error.audit), 3)
        self.assertEqual(error.audit[1]["execution_status"], "succeeded")
        self.assertEqual(error.audit[2]["execution_status"], "failed")
        self.assertEqual(error.audit[2]["step_id"], "malformed-step")
        self.assertNotIn("dependent-step", [item.get("step_id") for item in error.audit])

    def test_successful_sql_pipeline_marks_step_execution_succeeded(self) -> None:
        frame = pd.DataFrame({"value": [1, 1, 2]})
        result, audit = execute_plan(
            frame,
            {
                "steps": [
                    {
                        "id": "remove-duplicates",
                        "name": "remove duplicates",
                        "operation": "drop_duplicates",
                        "columns": ["value"],
                    }
                ]
            },
            mode="sql",
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(audit[0]["execution_status"], "succeeded")
        self.assertEqual(audit[1]["execution_status"], "succeeded")

    def test_sql_failure_context_redacts_string_literals(self) -> None:
        from sqlite3 import OperationalError

        error = SQLExecutionError(
            execution_id="run-id",
            step_id="step-1",
            operation_type="query",
            error=OperationalError("query failed"),
            sql_context="SELECT * FROM users WHERE email = 'person@example.com'",
            audit=[],
        )

        details = error.to_dict()
        self.assertIn("'[REDACTED]'", details["sql_context"])
        self.assertNotIn("person@example.com", details["sql_context"])

if __name__ == "__main__":
    unittest.main()
