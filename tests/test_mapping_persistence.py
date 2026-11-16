import json
import os
import sqlite3
import uuid
from datetime import datetime

import pytest

# Import the functions from the project
from src.etl import workflow_state as ws

# Helper to reset the mapping table before each test
@pytest.fixture(autouse=True)
def clean_mapping_table():
    # Ensure the table exists
    ws._ensure_schema_mapping_table()
    # Delete all rows
    conn = ws._connect()
    conn.execute("DELETE FROM schema_mapping_plans")
    conn.commit()
    conn.close()
    yield
    # Cleanup after test (in case any leftover)
    conn = ws._connect()
    conn.execute("DELETE FROM schema_mapping_plans")
    conn.commit()
    conn.close()


def test_save_and_load_mapping_plan():
    plan = {
        "plan_id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_dataset_id": "src-123",
        "target_dataset_id": "tgt-456",
        "mappings": [{"source": "name", "target": "full_name", "transformation": "rename"}],
    }
    plan_id = ws.save_mapping_plan(plan)
    assert plan_id == plan["plan_id"]
    loaded = ws.load_mapping_plan(plan_id)
    assert loaded == plan


def test_list_mapping_plans():
    # Insert two plans
    plans = []
    for i in range(2):
        p = {
            "plan_id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_dataset_id": f"src-{i}",
            "target_dataset_id": f"tgt-{i}",
            "approved": i % 2,
        }
        ws.save_mapping_plan(p)
        plans.append(p)
    lst = ws.list_mapping_plans(limit=10)
    # Should return both plans ordered by created_at desc
    assert len(lst) == 2
    returned_ids = {entry["plan_id"] for entry in lst}
    assert {p["plan_id"] for p in plans} == returned_ids
    # Verify approval flag is boolean
    for entry in lst:
        assert isinstance(entry["approved"], bool)


def test_update_mapping_approval():
    plan = {
        "plan_id": str(uuid.uuid4()),
        "source_dataset_id": "src",
        "target_dataset_id": "tgt",
    }
    ws.save_mapping_plan(plan)
    ws.update_mapping_approval(plan["plan_id"], approved=True, reviewer="alice", comments="looks good")
    loaded = ws.load_mapping_plan(plan["plan_id"])
    assert loaded["approved"] == 1 or loaded["approved"] is True
    # Reviewer metadata should be present
    assert loaded.get("reviewer") == "alice"
    assert loaded.get("reviewer_comments") == "looks good"


def test_update_nonexistent_plan_raises():
    with pytest.raises(ValueError):
        ws.update_mapping_approval("nonexistent", approved=True)


def test_record_mapping_execution_and_history():
    plan = {
        "plan_id": str(uuid.uuid4()),
        "source_dataset_id": "src",
        "target_dataset_id": "tgt",
    }
    ws.save_mapping_plan(plan)
    ws.record_mapping_execution(plan["plan_id"], status="executed", results={"rows": 10})
    loaded = ws.load_mapping_plan(plan["plan_id"])
    exec_info = loaded.get("execution")
    assert exec_info is not None
    assert exec_info["status"] == "executed"
    assert exec_info["results"] == {"rows": 10}
    # execution_timestamp should be stored (optional)
    assert "timestamp" in exec_info or "execution_timestamp" in exec_info
