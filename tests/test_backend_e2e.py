"""End-to-end integration and unit tests for the AI ETL Platform."""
import io
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.ingestion.universal_loader import parse_dataset_bytes
from backend.profiling.quality_engine import calculate_quality_metrics, profile_dataset
from backend.execution.engine import execute_operation_step
from backend.comparison.comparator import compare_datasets

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_csv_ingestion_and_quality():
    csv_content = b"customer_id,name,email,age,amount\n1,Alice,alice@example.com,25,100.5\n2,Bob,bob@invalid,30,200.0\n1,Alice,alice@example.com,25,100.5\n4,Charlie,,45,-50\n"
    df, meta = parse_dataset_bytes("test_customers.csv", csv_content)
    assert len(df) == 4
    assert "email" in df.columns

    quality = calculate_quality_metrics(df)
    assert "overall_score" in quality
    assert quality["uniqueness"]["duplicate_rows"] == 1
    assert quality["completeness"]["total_missing"] == 1

def test_execution_engine_deduplication():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": ["A", "B", "B", "C"]
    })
    transformed, log = execute_operation_step(df, "drop_duplicates", ["id", "name"], {})
    assert len(transformed) == 3
    assert log["status"] == "success"
    assert log["details"]["dropped_rows"] == 1

def test_comparison_engine():
    orig_df = pd.DataFrame({"id": [1, 1], "name": ["a", "a"]})
    trans_df = pd.DataFrame({"id": [1], "name": ["a"]})
    orig_q = calculate_quality_metrics(orig_df)
    trans_q = calculate_quality_metrics(trans_df)

    cmp = compare_datasets(orig_df, trans_df, orig_q, trans_q)
    assert cmp["overview"]["rows_removed"] == 1
    assert cmp["quality_comparison"]["uniqueness_delta"] > 0

def test_api_full_workflow():
    csv_bytes = b"id,email,amount\n1,user1@test.com,100\n2,invalid_email,200\n1,user1@test.com,100\n3,,300\n"
    files = {"file": ("test.csv", csv_bytes, "text/csv")}
    
    # 1. Upload
    up_res = client.post("/api/datasets/upload", files=files)
    assert up_res.status_code == 200
    up_data = up_res.json()
    workflow_id = up_data["workflow_id"]
    assert "quality" in up_data
    assert up_data["quality"]["uniqueness"]["duplicate_rows"] == 1

    # 2. Generate Plan
    plan_res = client.post("/api/etl/generate-plan", json={
        "workflow_id": workflow_id,
        "instruction": "Remove duplicate customer records and handle missing values."
    })
    assert plan_res.status_code == 200
    plan_data = plan_res.json()
    steps = plan_data["plan"]["steps"]
    assert len(steps) > 0
    step_ids = [s["step_id"] for s in steps]

    # 3. Approve Steps (Human-in-the-loop)
    app_res = client.post("/api/etl/approve-step", json={
        "workflow_id": workflow_id,
        "approved_steps": step_ids,
        "rejected_steps": []
    })
    assert app_res.status_code == 200
    assert app_res.json()["status"] == "approved"

    # 4. Execute Approved Pipeline
    exec_res = client.post("/api/etl/execute", json={"workflow_id": workflow_id})
    assert exec_res.status_code == 200
    exec_data = exec_res.json()
    assert "comparison" in exec_data
    assert exec_data["comparison"]["duplicate_analysis"]["duplicates_removed"] >= 1
    assert len(exec_data["execution_logs"]) == len(step_ids)

    # 5. Export
    exp_res = client.get(f"/api/export/{workflow_id}?format=csv")
    assert exp_res.status_code == 200
    assert len(exp_res.content) > 0
