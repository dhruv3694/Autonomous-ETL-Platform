"""
FastAPI REST API Routes.
Exposes endpoints for:
- /api/datasets/upload (universal multi-format upload, extraction, profiling, quality calculation)
- /api/etl/generate-plan (Qwen natural language instruction processing)
- /api/etl/approve-step (Human-in-the-loop granular step approval/rejection)
- /api/etl/execute (Safe step-by-step pipeline execution)
- /api/etl/comparison (Original vs Transformed comparison dashboard metrics)
- /api/workflows (History, status, and persisted checkpoints)
- /api/export (Cleaned dataset exports in CSV, JSON, Parquet, SQLite)
"""

import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd

from backend.ingestion.universal_loader import parse_dataset_bytes
from backend.profiling.quality_engine import profile_dataset, calculate_quality_metrics
from backend.llm.qwen_service import request_qwen_plan, ETLPlan
from backend.execution.engine import execute_operation_step
from backend.comparison.comparator import compare_datasets
from backend.database.workflow_store import (
    save_workflow, update_workflow_approval, update_workflow_execution,
    get_workflow, list_workflows
)

router = APIRouter(prefix="/api")
STORAGE_ROOT = Path(__file__).resolve().parent.parent.parent / "storage"

# In-memory DataFrame cache mapping workflow_id to { "original": df, "current": df }
DATASET_CACHE: Dict[str, Dict[str, pd.DataFrame]] = {}

class PlanRequest(BaseModel):
    workflow_id: str
    instruction: str
    ollama_url: Optional[str] = "http://localhost:11434"

class ApprovalRequest(BaseModel):
    workflow_id: str
    approved_steps: List[str]
    rejected_steps: List[str]
    comments: Optional[str] = ""

class ExecuteRequest(BaseModel):
    workflow_id: str

@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), sheet_name: Optional[str] = Form(None), table_name: Optional[str] = Form(None)):
    try:
        content = await file.read()
        workflow_id = str(uuid.uuid4())
        
        # Save raw original immutably to storage/uploads
        upload_dir = STORAGE_ROOT / "uploads" / workflow_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        raw_path = upload_dir / file.filename
        with open(raw_path, "wb") as f:
            f.write(content)

        # Parse universal bytes
        df, meta = parse_dataset_bytes(file.filename, content, sheet_name=sheet_name, table_name=table_name)
        
        # Profile & Quality
        profile = profile_dataset(df)
        quality = calculate_quality_metrics(df)

        # Cache in memory
        DATASET_CACHE[workflow_id] = {
            "original": df.copy(),
            "current": df.copy()
        }

        # Persist workflow record
        save_workflow(workflow_id=workflow_id, dataset_name=file.filename, status="uploaded")

        return {
            "workflow_id": workflow_id,
            "metadata": meta,
            "profile": profile,
            "quality": quality,
            "preview": df.head(15).fillna("").to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/etl/generate-plan")
def generate_plan(req: PlanRequest):
    if req.workflow_id not in DATASET_CACHE:
        # Try loading from storage if cache cold
        raise HTTPException(status_code=404, detail="Dataset not found or session expired.")

    df = DATASET_CACHE[req.workflow_id]["original"]
    profile = profile_dataset(df)
    
    plan, is_qwen = request_qwen_plan(
        pipeline_id=f"plan_{req.workflow_id[:8]}",
        dataset_name=req.workflow_id,
        schema_columns=list(df.columns),
        sample_profile=profile["columns"],
        instruction=req.instruction,
        ollama_url=req.ollama_url or "http://localhost:11434"
    )

    save_workflow(
        workflow_id=req.workflow_id,
        dataset_name=req.workflow_id,
        status="plan_generated",
        instruction=req.instruction,
        plan=plan.model_dump()
    )

    return {
        "workflow_id": req.workflow_id,
        "plan": plan.model_dump(),
        "is_live_qwen": is_qwen
    }

@router.post("/etl/approve-step")
def approve_step(req: ApprovalRequest):
    wf = get_workflow(req.workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found.")

    approval_state = {
        "approved_steps": req.approved_steps,
        "rejected_steps": req.rejected_steps,
        "comments": req.comments
    }

    status = "approved" if req.approved_steps else "rejected"
    update_workflow_approval(req.workflow_id, approval_state, status)

    return {
        "workflow_id": req.workflow_id,
        "status": status,
        "approval_state": approval_state
    }

@router.post("/etl/execute")
def execute_pipeline(req: ExecuteRequest):
    if req.workflow_id not in DATASET_CACHE:
        raise HTTPException(status_code=404, detail="Dataset not loaded in memory.")

    wf = get_workflow(req.workflow_id)
    if not wf or not wf.get("plan"):
        raise HTTPException(status_code=400, detail="No generated plan found.")

    plan = wf["plan"]
    approval_state = wf.get("approval_state", {})
    approved_ids = set(approval_state.get("approved_steps", []))

    if not approved_ids:
        raise HTTPException(status_code=400, detail="Cannot execute: No steps have been explicitly approved.")

    df = DATASET_CACHE[req.workflow_id]["original"].copy()
    execution_logs = []

    for step in plan.get("steps", []):
        if step["step_id"] in approved_ids:
            df, log_entry = execute_operation_step(
                df=df,
                operation=step["operation"],
                target_columns=step.get("target_columns", []),
                parameters=step.get("parameters", {})
            )
            log_entry["step_id"] = step["step_id"]
            log_entry["step_name"] = step["name"]
            execution_logs.append(log_entry)

    DATASET_CACHE[req.workflow_id]["current"] = df.copy()

    # Calculate comparison & transformed quality
    orig_df = DATASET_CACHE[req.workflow_id]["original"]
    orig_quality = calculate_quality_metrics(orig_df)
    trans_quality = calculate_quality_metrics(df)
    comparison = compare_datasets(orig_df, df, orig_quality, trans_quality)

    update_workflow_execution(req.workflow_id, execution_logs, comparison, status="completed")

    return {
        "workflow_id": req.workflow_id,
        "execution_logs": execution_logs,
        "comparison": comparison,
        "transformed_quality": trans_quality,
        "transformed_preview": df.head(15).fillna("").to_dict(orient="records")
    }

@router.get("/workflows")
def get_workflow_history():
    return list_workflows()

@router.get("/workflows/{workflow_id}")
def get_single_workflow(workflow_id: str):
    wf = get_workflow(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    return wf

@router.get("/export/{workflow_id}")
def export_dataset(workflow_id: str, format: str = "csv"):
    if workflow_id not in DATASET_CACHE:
        raise HTTPException(status_code=404, detail="Dataset not found in memory.")

    df = DATASET_CACHE[workflow_id]["current"]
    out_dir = STORAGE_ROOT / "outputs" / workflow_id
    out_dir.mkdir(parents=True, exist_ok=True)

    file_ext = format.lower()
    out_file = out_dir / f"cleaned_dataset.{file_ext}"

    if file_ext == "csv":
        df.to_csv(out_file, index=False)
        return FileResponse(out_file, media_type="text/csv", filename="cleaned_dataset.csv")
    elif file_ext == "json":
        df.to_json(out_file, orient="records", indent=2)
        return FileResponse(out_file, media_type="application/json", filename="cleaned_dataset.json")
    elif file_ext == "parquet":
        df.to_parquet(out_file, index=False)
        return FileResponse(out_file, media_type="application/octet-stream", filename="cleaned_dataset.parquet")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported export format '{format}'. Supported: csv, json, parquet.")
