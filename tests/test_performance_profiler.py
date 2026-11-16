from __future__ import annotations

import io
import unittest

import pandas as pd

from src.etl.data_loader import load_uploaded_files
from src.etl.execution_engine import execute_plan
from src.etl.performance_profiler import (
    identify_bottlenecks,
    profile_callable,
    system_resource_snapshot,
)


class UploadedCsv(io.BytesIO):
    name = "customers.csv"

    @property
    def size(self) -> int:
        return len(self.getvalue())


class PerformanceProfilerTests(unittest.TestCase):
    def test_profile_callable_records_duration_and_dataframe_metrics(self) -> None:
        frame = pd.DataFrame({"value": [1, 2, 3]})

        result, metrics = profile_callable(
            "copy",
            "transformation",
            lambda: frame.copy(),
            dataframe_in=frame,
        )

        self.assertTrue(result.equals(frame))
        self.assertEqual(metrics["status"], "completed")
        self.assertGreaterEqual(metrics["duration_seconds"], 0)
        self.assertEqual(metrics["records_in"], 3)
        self.assertEqual(metrics["records_out"], 3)
        self.assertGreater(metrics["dataframe_input_bytes"], 0)
        self.assertGreater(metrics["sampled_peak_rss_bytes"], 0)
        self.assertIsNotNone(metrics["python_peak_allocation_bytes"])

    def test_identifies_only_measured_operations_over_threshold(self) -> None:
        bottlenecks = identify_bottlenecks(
            [
                {"name": "dominant", "stage": "transformation", "status": "completed", "duration_seconds": 4.0},
                {"name": "small", "stage": "transformation", "status": "completed", "duration_seconds": 1.0},
                {"name": "failed", "stage": "transformation", "status": "failed", "duration_seconds": 20.0},
            ]
        )

        self.assertEqual([item["name"] for item in bottlenecks], ["dominant"])
        self.assertAlmostEqual(bottlenecks[0]["share_of_measured_time"], 0.8)
        self.assertIn("Measured elapsed time", bottlenecks[0]["evidence"])

    def test_does_not_infer_bottleneck_from_single_operation(self) -> None:
        self.assertEqual(
            identify_bottlenecks(
                [
                    {
                        "name": "only operation",
                        "stage": "transformation",
                        "status": "completed",
                        "duration_seconds": 1.0,
                    }
                ]
            ),
            [],
        )

    def test_system_snapshot_marks_unmeasured_gpu(self) -> None:
        snapshot = system_resource_snapshot()

        self.assertGreater(snapshot["cpu_count_logical"], 0)
        self.assertGreater(snapshot["memory_total_bytes"], 0)
        self.assertIsNone(snapshot["gpu_utilization_percent"])

    def test_upload_and_etl_audit_include_stage_and_operation_metrics(self) -> None:
        uploaded = UploadedCsv(b"name,age\nAda,36\nAda,36\nLin,28\n")
        datasets = load_uploaded_files([uploaded])
        dataset = datasets[0]

        self.assertEqual(dataset["performance"]["extraction"]["stage"], "extraction")
        self.assertEqual(dataset["performance"]["extraction"]["records_out"], 3)
        self.assertEqual(dataset["performance"]["schema_analysis"]["stage"], "schema_analysis")

        output, audit = execute_plan(
            dataset["dataframe"],
            {
                "steps": [
                    {
                        "name": "Drop duplicates",
                        "operation": "drop_duplicates",
                        "columns": list(dataset["dataframe"].columns),
                    }
                ]
            },
            mode="python",
        )

        self.assertEqual(len(output), 2)
        self.assertEqual(audit[0]["performance"]["stage"], "transformation")
        self.assertEqual(audit[0]["performance"]["records_in"], 3)
        self.assertEqual(audit[0]["performance"]["records_out"], 2)

    def test_sql_execution_retains_load_and_operation_profiles(self) -> None:
        frame = pd.DataFrame({"name": ["Ada", "Ada"], "age": [36, 36]})
        output, audit = execute_plan(
            frame,
            {
                "steps": [
                    {
                        "name": "Drop duplicates",
                        "operation": "drop_duplicates",
                        "columns": ["name", "age"],
                    }
                ]
            },
            mode="sql",
        )

        self.assertEqual(len(output), 1)
        self.assertEqual(output.iloc[0].to_dict(), {"name": "Ada", "age": 36})
        self.assertEqual(audit[0]["performance"]["stage"], "loading")
        self.assertEqual(audit[1]["performance"]["stage"], "transformation")
        self.assertEqual(audit[1]["performance"]["records_out"], 1)


if __name__ == "__main__":
    unittest.main()
