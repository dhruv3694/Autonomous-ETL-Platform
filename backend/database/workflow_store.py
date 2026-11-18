"""
Persistent Workflow and Approval State Store.
Stores workflows, uploaded datasets, generated plans, approval decisions, intermediate results,
and comparison metrics in SQLite so they survive restarts.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

DB_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "workflows" / "workflow_state.db"

def init_workflow_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                dataset_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'uploaded',
                instruction TEXT,
                plan_json TEXT,
                approval_state_json TEXT,
                execution_log_json TEXT,
                comparison_json TEXT
            )
        """)
        conn.commit()

def save_workflow(workflow_id: str, dataset_name: str, status: str = "uploaded", instruction: str = "", plan: Optional[Dict] = None) -> None:
    init_workflow_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO workflows (workflow_id, dataset_name, status, instruction, plan_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            workflow_id,
            dataset_name,
            status,
            instruction,
            json.dumps(plan or {}, ensure_ascii=False)
        ))
        conn.commit()

def update_workflow_approval(workflow_id: str, approval_state: Dict[str, Any], status: str) -> None:
    init_workflow_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            UPDATE workflows
            SET approval_state_json = ?, status = ?
            WHERE workflow_id = ?
        """, (json.dumps(approval_state, ensure_ascii=False), status, workflow_id))
        conn.commit()

def update_workflow_execution(workflow_id: str, execution_log: List[Dict[str, Any]], comparison: Dict[str, Any], status: str = "completed") -> None:
    init_workflow_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            UPDATE workflows
            SET execution_log_json = ?, comparison_json = ?, status = ?
            WHERE workflow_id = ?
        """, (
            json.dumps(execution_log, ensure_ascii=False),
            json.dumps(comparison, ensure_ascii=False),
            status,
            workflow_id
        ))
        conn.commit()

def get_workflow(workflow_id: str) -> Optional[Dict[str, Any]]:
    init_workflow_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "workflow_id": row["workflow_id"],
            "created_at": row["created_at"],
            "dataset_name": row["dataset_name"],
            "status": row["status"],
            "instruction": row["instruction"],
            "plan": json.loads(row["plan_json"] or "{}"),
            "approval_state": json.loads(row["approval_state_json"] or "{}"),
            "execution_log": json.loads(row["execution_log_json"] or "[]"),
            "comparison": json.loads(row["comparison_json"] or "{}")
        }

def list_workflows(limit: int = 50) -> List[Dict[str, Any]]:
    init_workflow_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT workflow_id, created_at, dataset_name, status, instruction FROM workflows ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]
