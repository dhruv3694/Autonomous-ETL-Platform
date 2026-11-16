from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

# ---------------------------------------------------------------------------
# Phase 6 state‑machine utilities
# ---------------------------------------------------------------------------

# Allowed status transitions for a schema‑mapping plan.
_VALID_TRANSITIONS: dict[str, list[str]] = {
    "draft": ["pending_approval", "rejected"],
    "pending_approval": ["approved", "rejected"],
    "approved": ["executing"],
    "executing": ["executed", "failed"],
    "executed": [],
    "rejected": [],
}

def _validate_transition(current: str, target: str) -> None:
    """Validate that a plan may move from *current* status to *target*.

    Raises:
        ValueError: If the transition is not permitted.
    """
    if target not in _VALID_TRANSITIONS.get(current, []):
        raise ValueError(
            f"Invalid status transition from {current!r} to {target!r}. "
            f"Allowed targets: {_VALID_TRANSITIONS.get(current, [])}"
        )

def _record_event(
    plan_id: str,
    version_number: int,
    event_type: str,
    previous_status: str | None = None,
    new_status: str | None = None,
    reviewer: str | None = None,
    reviewer_comments: str | None = None,
    execution_results: str | None = None,
) -> None:
    """Insert an event record for *plan_id*.

    The function writes to ``schema_mapping_plan_events``. It is a thin wrapper
    around the INSERT statement and is used by the higher‑level helpers.
    """
    _ensure_schema_mapping_table()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO schema_mapping_plan_events (
                plan_id,
                version_number,
                event_type,
                previous_status,
                new_status,
                reviewer,
                reviewer_comments,
                execution_results
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan_id,
                version_number,
                event_type,
                previous_status,
                new_status,
                reviewer,
                reviewer_comments,
                execution_results,
            ),
        )

from datetime import datetime

DB_PATH = Path(__file__).resolve().parents[2] / "workflow_history.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                dataset_name TEXT,
                instruction TEXT,
                plan_json TEXT,
                details_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                dataset_name TEXT,
                dataset_label TEXT,
                instruction TEXT,
                execution_mode TEXT,
                plan_json TEXT,
                last_run_json TEXT,
                state_json TEXT
            )
            """
        )


def record_event(
    event_type: str,
    dataset_name: str = "",
    instruction: str = "",
    plan: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO workflow_events (event_type, dataset_name, instruction, plan_json, details_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                event_type,
                dataset_name,
                instruction,
                json.dumps(plan or {}, ensure_ascii=False),
                json.dumps(details or {}, ensure_ascii=False),
            ),
        )


def save_snapshot(
    dataset_name: str,
    dataset_label: str,
    instruction: str,
    execution_mode: str,
    plan: Optional[Dict[str, Any]] = None,
    last_run: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    init_db()
    payload = {
        "dataset_name": dataset_name,
        "dataset_label": dataset_label,
        "instruction": instruction,
        "execution_mode": execution_mode,
        "plan": plan or {},
        "last_run": last_run or {},
        "state": state or {},
    }
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO workflow_snapshots (dataset_name, dataset_label, instruction, execution_mode, plan_json, last_run_json, state_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_name,
                dataset_label,
                instruction,
                execution_mode,
                json.dumps(plan or {}, ensure_ascii=False),
                json.dumps(last_run or {}, ensure_ascii=False),
                json.dumps(state or {}, ensure_ascii=False),
            ),
        )
    payload["id"] = cursor.lastrowid
    return payload


def list_snapshots(limit: int = 10) -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, dataset_name, dataset_label, instruction, execution_mode, plan_json, last_run_json, state_json
            FROM workflow_snapshots
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    snapshots: List[Dict[str, Any]] = []
    for row in rows:
        snapshots.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "dataset_name": row["dataset_name"],
                "dataset_label": row["dataset_label"],
                "instruction": row["instruction"],
                "execution_mode": row["execution_mode"],
                "plan": json.loads(row["plan_json"] or "{}"),
                "last_run": json.loads(row["last_run_json"] or "{}"),
                "state": json.loads(row["state_json"] or "{}"),
            }
        )
    return snapshots


def list_history(limit: int = 20) -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT event_type, created_at, dataset_name, instruction, plan_json, details_json
            FROM workflow_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    history: List[Dict[str, Any]] = []
    for row in rows:
        history.append(
            {
                "event_type": row["event_type"],
                "created_at": row["created_at"],
                "dataset_name": row["dataset_name"],
                "instruction": row["instruction"],
                "plan": json.loads(row["plan_json"] or "{}"),
                "details": json.loads(row["details_json"] or "{}"),
            }
        )
    return history

# ---------------------------------------------------------------------------
# Schema Mapping Plan persistence helpers (Phase 6)
# ---------------------------------------------------------------------------
import uuid

# ---------------------------------------------------------------------------
# Phase 6 state‑machine utilities
# ---------------------------------------------------------------------------

# Allowed status transitions for a schema‑mapping plan.
_VALID_TRANSITIONS: dict[str, list[str]] = {
    "draft": ["pending_approval", "rejected"],
    "pending_approval": ["approved", "rejected"],
    "approved": ["executing"],
    "executing": ["executed", "failed"],
    "executed": [],
    "rejected": [],
}

def _validate_transition(current: str, target: str) -> None:
    """Validate that a plan may move from *current* status to *target*.

    Raises:
        ValueError: If the transition is not permitted.
    """
    if target not in _VALID_TRANSITIONS.get(current, []):
        raise ValueError(
            f"Invalid status transition from {current!r} to {target!r}. "
            f"Allowed targets: {_VALID_TRANSITIONS.get(current, [])}"
        )

def _record_event(
    plan_id: str,
    version_number: int,
    event_type: str,
    previous_status: str | None = None,
    new_status: str | None = None,
    reviewer: str | None = None,
    reviewer_comments: str | None = None,
    execution_results: str | None = None,
) -> None:
    """Insert an event record for *plan_id*.

    The function writes to ``schema_mapping_plan_events``. It is a thin wrapper
    around the INSERT statement and is used by the higher‑level helpers.
    """
    _ensure_schema_mapping_table()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO schema_mapping_plan_events (
                plan_id,
                version_number,
                event_type,
                previous_status,
                new_status,
                reviewer,
                reviewer_comments,
                execution_results
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan_id,
                version_number,
                event_type,
                previous_status,
                new_status,
                reviewer,
                reviewer_comments,
                execution_results,
            ),
        )


def _ensure_schema_mapping_table() -> None:
    """Create or migrate the schema‑mapping tables.

    - ``schema_mapping_plans`` stores the stable plan identifier, the
      current version number and the overall lifecycle status.
    - ``schema_mapping_plan_versions`` stores immutable snapshots of each
      revision.
    - ``schema_mapping_plan_events`` records state‑machine transitions and
      execution outcomes.
    The function is idempotent – it will add missing columns or tables
    without destroying existing data.
    """
    with _connect() as conn:
        # Main plan table – add columns if this is a migration from the old schema.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_mapping_plans (
                plan_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source_dataset_id TEXT NOT NULL,
                target_dataset_id TEXT NOT NULL,
                current_version INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'draft',
                plan_json TEXT NOT NULL,
                approved INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        # Add missing columns for legacy databases.
        # SQLite does not support ALTER COLUMN, but we can add new columns safely.
        existing = conn.execute("PRAGMA table_info(schema_mapping_plans)").fetchall()
        cols = {row["name"] for row in existing}
        if "current_version" not in cols:
            conn.execute("ALTER TABLE schema_mapping_plans ADD COLUMN current_version INTEGER NOT NULL DEFAULT 1")
        if "status" not in cols:
            conn.execute("ALTER TABLE schema_mapping_plans ADD COLUMN status TEXT NOT NULL DEFAULT 'draft'")
        # Version snapshots table.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_mapping_plan_versions (
                plan_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                PRIMARY KEY (plan_id, version_number),
                FOREIGN KEY (plan_id) REFERENCES schema_mapping_plans(plan_id) ON DELETE CASCADE
            )
            """
        )
        # Event history table.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_mapping_plan_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                previous_status TEXT,
                new_status TEXT NOT NULL,
                reviewer TEXT,
                reviewer_comments TEXT,
                execution_results TEXT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (plan_id) REFERENCES schema_mapping_plans(plan_id) ON DELETE CASCADE
            )
            """
        )

    """Create the `schema_mapping_plans` table if it does not exist.
    This is called lazily by the mapping‑plan helpers so we do not need to
    modify the original `init_db` implementation.
    """
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_mapping_plans (
                plan_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source_dataset_id TEXT NOT NULL,
                target_dataset_id TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                approved INTEGER NOT NULL DEFAULT 0
            )
            """
        )

def save_mapping_plan(plan: dict) -> str:
    """Persist a mapping plan and return its `plan_id`.
    The plan dict must contain the keys ``plan_id``, ``source_dataset_id`` and
    ``target_dataset_id`` as created by ``generate_mapping_plan``.
    """
    _ensure_schema_mapping_table()
    plan_id = plan.get("plan_id") or str(uuid.uuid4())
    created_at = plan.get("created_at") or datetime.utcnow().isoformat() + "Z"
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO schema_mapping_plans (plan_id, created_at, source_dataset_id, target_dataset_id, plan_json, approved) VALUES (?, ?, ?, ?, ?, ?)",
            (
                plan_id,
                created_at,
                plan.get("source_dataset_id"),
                plan.get("target_dataset_id"),
                json.dumps(plan, ensure_ascii=False),
                int(plan.get("approved", 0)),
            ),
        )
    return plan_id

def load_mapping_plan(plan_id: str) -> dict | None:
    """Retrieve a stored mapping plan by its ``plan_id``.
    Returns ``None`` if the plan does not exist.
    """
    _ensure_schema_mapping_table()
    with _connect() as conn:
        row = conn.execute(
            "SELECT plan_json FROM schema_mapping_plans WHERE plan_id = ?", (plan_id,)
        ).fetchone()
    return json.loads(row["plan_json"]) if row else None

def list_mapping_plans(limit: int = 20) -> list[dict]:
    """Return a list of recent mapping‑plan summaries.
    Each entry contains ``plan_id``, ``created_at``, ``source_dataset_id``,
    ``target_dataset_id`` and ``approved`` flag.
    """
    _ensure_schema_mapping_table()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT plan_id, created_at, source_dataset_id, target_dataset_id, approved FROM schema_mapping_plans ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "plan_id": r["plan_id"],
            "created_at": r["created_at"],
            "source_dataset_id": r["source_dataset_id"],
            "target_dataset_id": r["target_dataset_id"],
            "approved": bool(r["approved"]),
        }
        for r in rows
    ]

# ---------------------------------------------------------------------------
# Mapping plan approval and execution tracking (Phase 6 extensions)
# ---------------------------------------------------------------------------

def update_mapping_approval(plan_id: str, approved: bool, reviewer: str = "", comments: str = "") -> None:
    """Update the approval status of a stored mapping plan.

    The ``approved`` flag is stored in the ``approved`` column. Optional
    reviewer information and free‑form comments are merged into the stored
    ``plan_json`` payload under the keys ``reviewer`` and ``reviewer_comments``.
    """
    _ensure_schema_mapping_table()
    with _connect() as conn:
        # Retrieve existing JSON payload
        row = conn.execute(
            "SELECT plan_json FROM schema_mapping_plans WHERE plan_id = ?",
            (plan_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Mapping plan {plan_id!r} does not exist")
        plan = json.loads(row["plan_json"])
        plan["approved"] = bool(approved)
        # Merge reviewer metadata
        if reviewer:
            plan["reviewer"] = reviewer
        if comments:
            plan["reviewer_comments"] = comments
        # Update approval flag and JSON payload
        conn.execute(
            "UPDATE schema_mapping_plans SET approved = ?, plan_json = ? WHERE plan_id = ?",
            (int(approved), json.dumps(plan, ensure_ascii=False), plan_id),
        )

def record_mapping_execution(
    plan_id: str,
    status: str,
    results: dict | None = None,
    execution_timestamp: str | None = None,
) -> None:
    """Record the execution outcome of a mapping plan.

    ``status`` should be one of ``"pending"``, ``"executed"`` or ``"failed"``.
    ``results`` is an optional dictionary containing validation or runtime
    diagnostics; it will be stored inside the plan JSON under the ``execution``
    key.
    """
    _ensure_schema_mapping_table()
    with _connect() as conn:
        row = conn.execute(
            "SELECT plan_json FROM schema_mapping_plans WHERE plan_id = ?",
            (plan_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Mapping plan {plan_id!r} does not exist")
        plan = json.loads(row["plan_json"])
        exec_info: dict = {
            "status": status,
            "timestamp": execution_timestamp or datetime.utcnow().isoformat() + "Z",
        }
        if results:
            exec_info["results"] = results
        plan["execution"] = exec_info
        conn.execute(
            "UPDATE schema_mapping_plans SET plan_json = ? WHERE plan_id = ?",
            (json.dumps(plan, ensure_ascii=False), plan_id),
        )

def get_mapping_approval_history(limit: int = 20) -> List[dict]:
    """Return a list of mapping‑plan records with approval metadata.

    Includes ``reviewer`` and ``reviewer_comments`` when present in the
    stored ``plan_json``.
    """
    _ensure_schema_mapping_table()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT plan_id, created_at, source_dataset_id, target_dataset_id, approved, plan_json "
            "FROM schema_mapping_plans ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    history: List[dict] = []
    for row in rows:
        base = {
            "plan_id": row["plan_id"],
            "created_at": row["created_at"],
            "source_dataset_id": row["source_dataset_id"],
            "target_dataset_id": row["target_dataset_id"],
            "approved": bool(row["approved"]),
        }
        payload = json.loads(row["plan_json"] or "{}")
        if "reviewer" in payload:
            base["reviewer"] = payload["reviewer"]
        if "reviewer_comments" in payload:
            base["reviewer_comments"] = payload["reviewer_comments"]
        history.append(base)
    return history
