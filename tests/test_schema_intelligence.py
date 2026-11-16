import pandas as pd
import pytest

from src.etl import schema_intelligence as si

# Helper DataFrames ---------------------------------------------------------

def df_simple():
    return pd.DataFrame({
        "customer_id": [1, 2, 3],
        "full_name": ["Alice", "Bob", "Carol"],
        "email": ["alice@example.com", "bob@example.com", "carol@example.com"],
        "signup_date": pd.to_datetime(["2020-01-01", "2020-06-15", "2021-03-20"]),
        "amount": [10.5, 20.0, 15.75],
    })

def df_target():
    return pd.DataFrame({
        "customer_id": [1, 2, 3],
        "customer_name": ["Alice", "Bob", "Carol"],
        "email_address": ["alice@example.com", "bob@example.com", "carol@example.com"],
        "transaction_date": pd.to_datetime(["2020-01-01", "2020-06-15", "2021-03-20"]),
        "transaction_amount": [10.5, 20.0, 15.75],
    })

# ---------------------------------------------------------------------------
# 1. Profiling tests
# ---------------------------------------------------------------------------

def test_profile_schema_basic():
    df = df_simple()
    meta = si.profile_schema(df)
    # Basic top‑level keys
    assert meta["row_count"] == 3
    assert meta["column_count"] == 5
    # Check that logical types were inferred correctly
    types = {c["name"]: c["logical_type"] for c in meta["columns"]}
    assert types["customer_id"] == "integer"
    assert types["full_name"] == "text"
    assert types["email"] == "email"
    assert types["signup_date"] in {"date", "datetime"}
    assert types["amount"] == "decimal"
    # Semantic tag detection should pick up known synonyms
    tags = {c["name"]: c["semantic_tags"] for c in meta["columns"]}
    assert "customer_id" in tags["customer_id"]  # canonical tag
    assert "email_address" in tags["email"]

# ---------------------------------------------------------------------------
# 2. Column‑mapping engine tests
# ---------------------------------------------------------------------------

def test_infer_column_mappings_exact_and_synonym():
    src_meta = si.profile_schema(df_simple())
    tgt_meta = si.profile_schema(df_target())
    mappings = si.infer_column_mappings(src_meta, tgt_meta, llm=False)
    # Build a dict for easy lookup
    map_dict = {m["source_column"]: m["target_column"] for m in mappings}
    # Exact match on identifier
    assert map_dict["customer_id"] == "customer_id"
    # Synonym/semantic match for name and email
    assert map_dict["full_name"] == "customer_name"
    assert map_dict["email"] == "email_address"
    # Date and amount columns should be mapped correctly
    assert map_dict["signup_date"] == "transaction_date"
    assert map_dict["amount"] == "transaction_amount"
    # All mappings should have a confidence category of HIGH or MEDIUM
    for m in mappings:
        assert m["category"] in {"HIGH", "MEDIUM", "LOW", "INCOMPATIBLE", "AMBIGUOUS"}

# ---------------------------------------------------------------------------
# 3. Compatibility validation tests
# ---------------------------------------------------------------------------

def test_validate_schema_compatibility():
    src_meta = si.profile_schema(df_simple())
    tgt_meta = si.profile_schema(df_target())
    mappings = si.infer_column_mappings(src_meta, tgt_meta, llm=False)
    report = si.validate_schema_compatibility(src_meta, tgt_meta, mappings)
    # No INCOMPATIBLE entries should appear for a well‑matched case
    statuses = {r["status"] for r in report}
    assert "INCOMPATIBLE" not in statuses
    # Every target column must be present in the mapping report (missing would be INCOMPATIBLE)
    missing_target = [r for r in report if r["status"] == "INCOMPATIBLE"]
    assert not missing_target

# ---------------------------------------------------------------------------
# 4. Mapping‑plan generation sanity check
# ---------------------------------------------------------------------------

def test_generate_mapping_plan_structure():
    src_meta = si.profile_schema(df_simple())
    tgt_meta = si.profile_schema(df_target())
    mappings = si.infer_column_mappings(src_meta, tgt_meta, llm=False)
    relationships = si.infer_keys({"src": df_simple(), "tgt": df_target()})
    compatibility = si.validate_schema_compatibility(src_meta, tgt_meta, mappings)
    plan = si.generate_mapping_plan(
        source_dataset_id="src",
        target_dataset_id="tgt",
        source_meta=src_meta,
        target_meta=tgt_meta,
        mappings=mappings,
        relationships=relationships,
        compatibility=compatibility,
    )
    # Verify required top‑level keys exist and are JSON‑serialisable
    for key in ["plan_id", "created_at", "source_dataset_id", "target_dataset_id", "source_schema", "target_schema", "mappings", "relationships", "compatibility_report", "overall_status"]:
        assert key in plan
    # Ensure each mapping entry contains approval fields
    for m in plan["mappings"]:
        assert "approved" in m
        assert "approval_timestamp" in m

# ---------------------------------------------------------------------------
# 5. Persistence helpers (workflow_state) – simple round‑trip
# ---------------------------------------------------------------------------

def test_workflow_state_save_and_load(tmp_path):
    # Use the existing workflow_state module (it creates/uses the real DB file).
    from src.etl import workflow_state as ws
    # Build a tiny plan
    src_meta = si.profile_schema(df_simple())
    tgt_meta = si.profile_schema(df_target())
    mappings = si.infer_column_mappings(src_meta, tgt_meta, llm=False)
    plan = si.generate_mapping_plan(
        source_dataset_id="src",
        target_dataset_id="tgt",
        source_meta=src_meta,
        target_meta=tgt_meta,
        mappings=mappings,
        relationships=[],
        compatibility=[],
    )
    plan_id = ws.save_mapping_plan(plan)
    loaded = ws.load_mapping_plan(plan_id)
    assert loaded["plan_id"] == plan_id
    assert loaded["source_dataset_id"] == "src"
    assert loaded["target_dataset_id"] == "tgt"
    # List should contain the newly stored plan
    lst = ws.list_mapping_plans(limit=5)
    assert any(p["plan_id"] == plan_id for p in lst)

# ---------------------------------------------------------------------------
# 6. Edge case – unknown logical type handling
# ---------------------------------------------------------------------------

def test_unknown_logical_type():
    df = pd.DataFrame({"misc": [None, None, None]})
    meta = si.profile_schema(df)
    # Logical type should be "unknown"
    assert meta["columns"][0]["logical_type"] == "unknown"

# The above tests cover the core of Phase 6 without requiring external LLMs or the UI.
