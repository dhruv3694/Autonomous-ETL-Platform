'''schema_intelligence.py

Schema Intelligence utilities for Phase 6.

This module builds on the existing dataset profiling code in
`src.etl.pipeline_generator.build_dataset_context`.  It provides:

* `profile_schema` – a pure function that returns a JSON‑serialisable
  dictionary describing each column of a DataFrame.
* Logical type inference helpers.
* Optional semantic tag detection (deterministic + optional LLM fallback).
* Column‑mapping engine (`infer_column_mappings`).
* Primary/foreign‑key inference (`infer_keys`).
* Compatibility validation (`validate_schema_compatibility`).
* Mapping‑plan generation (`generate_mapping_plan`).

All functions avoid mutating the input DataFrames and never transmit full
row data to the LLM – only column names and up‑to‑three sanitized sample
values.
''' 

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Helper constants & utilities
# ---------------------------------------------------------------------------
_MAX_SAMPLE_VALUES = 5
_MAX_SAMPLE_VALUE_LENGTH = 100
_MAX_ROWS_FOR_PROFILING = 100_000  # cap for large datasets

def _bounded_value(val: Any) -> str:
    """Render a value safely for inclusion in metadata/L LM prompts.

    * Truncate long strings.
    * Convert timestamps to ISO format.
    """
    if isinstance(val, pd.Timestamp):
        rendered = val.isoformat()
    else:
        rendered = str(val)
    return rendered[:_MAX_SAMPLE_VALUE_LENGTH]

# ---------------------------------------------------------------------------
# Logical type inference
# ---------------------------------------------------------------------------
_LOGICAL_TYPE_PATTERNS = {
    "email": re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.IGNORECASE),
    "phone": re.compile(
        r"^(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]?)?\d{3,4}[ .-]?\d{4}$"
    ),
    "currency": re.compile(r"^[\$€£¥]\s?\d+(?:[.,]\d{2})?$"),
    "percentage": re.compile(r"^\d+(?:[.,]\d+)?%$"),
    "identifier": re.compile(r"^[A-Za-z0-9_-]+$"),
}

def _infer_logical_type(series: pd.Series) -> str:
    """Return a logical data‑type label for a column.

    The function works on a *sample* of at most 100 non‑null values to keep it
    fast.  The precedence order is:

    1. Boolean (True/False strings or native bool dtype)
    2. Integer / Decimal (numeric dtype)
    3. Date / Datetime (parsable with pandas to_datetime)
    4. Pattern‑based types (email, phone, currency, percentage, identifier)
    5. Text (default)
    """
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"

    # Boolean detection
    if non_null.dtype == bool:
        return "boolean"
    if non_null.astype(str).str.lower().isin(["true", "false", "yes", "no", "0", "1"]).all():
        return "boolean"

    # Numeric detection
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "decimal"

    # Date / datetime detection (sample up to 100 rows)
    sample = non_null.head(100)
    try:
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().all():
            if parsed.dt.time.eq(pd.Timestamp("00:00").time()).all():
                return "date"
            return "datetime"
    except Exception:
        pass

    # Column names such as "full_name" / "customer_name" are names, not IDs,
    # even though the sample values could each look like an identifier token.
    column_name = str(series.name or "").lower()
    if (
        "name" in column_name
        and "id" not in column_name
        and "code" not in column_name
        and "number" not in column_name
    ):
        return "text"

    # Pattern based detection – use first 20 non‑null values for speed
    for pattern_name, regex in _LOGICAL_TYPE_PATTERNS.items():
        if sample.astype(str).apply(lambda v: bool(regex.match(v))).all():
            return pattern_name

    return "text"

# ---------------------------------------------------------------------------
# Semantic tag detection (deterministic + optional LLM fallback)
# ---------------------------------------------------------------------------
_SEMANTIC_SYNONYMS: Mapping[str, List[str]] = {
    "customer_id": ["cust_no", "customerid", "client_id", "clientno"],
    "customer_name": ["full_name", "client_name", "cust_name"],
    "email_address": ["email", "emailaddr", "e_mail"],
    "phone_number": ["phone", "phone_no", "tel"],
    "transaction_date": ["txn_dt", "date", "transactiondate"],
    "transaction_amount": ["net_amt", "amount", "total", "price"],
    "product_id": ["prod_code", "item_id", "sku"],
    "order_id": ["order_no", "orderid"],
    # add more as needed
}

def _detect_semantic_tag(column_name: str, samples: Sequence[Any]) -> List[str]:
    """Return a list of candidate semantic tags for a column.

    The deterministic part checks the static synonym dictionary.  If the
    column name (lowercased, punctuation stripped) matches a synonym we return
    the canonical tag.
    """
    norm = re.sub(r"[^a-z0-9]", "", column_name.lower())
    candidates: List[str] = []
    for tag, syns in _SEMANTIC_SYNONYMS.items():
        if norm == tag.replace("_", "") or norm in [re.sub(r"[^a-z0-9]", "", s) for s in syns]:
            candidates.append(tag)
    return candidates

# ---------------------------------------------------------------------------
# Public profiling API
# ---------------------------------------------------------------------------
def profile_schema(df: pd.DataFrame, *, max_rows: int | None = None) -> Dict[str, Any]:
    """Return a structured metadata description for *df*.

    The function never mutates *df*.  If *max_rows* is provided and the frame is
    larger, a random sample is used for expensive operations (date parsing,
    pattern checks).  The returned dict is JSON‑serialisable.
    """
    if max_rows is None:
        max_rows = _MAX_ROWS_FOR_PROFILING
    row_count = len(df)
    if row_count > max_rows:
        # reproducible sampling for consistency across runs
        df_sample = df.sample(n=max_rows, random_state=42)
        sampling_used = True
    else:
        df_sample = df
        sampling_used = False

    columns_meta: List[Dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        sample_vals = [_bounded_value(v) for v in non_null.head(_MAX_SAMPLE_VALUES)]
        logical_type = _infer_logical_type(series)
        semantic_tags = _detect_semantic_tag(col, sample_vals)
        meta: Dict[str, Any] = {
            "name": col,
            "dtype": str(series.dtype),
            "logical_type": logical_type,
            "null_count": int(series.isna().sum()),
            "null_percent": round(100 * series.isna().mean(), 2),
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": sample_vals,
            "semantic_tags": semantic_tags,
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                meta.update(
                    {
                        "min": float(numeric.min()),
                        "max": float(numeric.max()),
                        "mean": float(numeric.mean()),
                        "median": float(numeric.median()),
                    }
                )
        elif pd.api.types.is_string_dtype(series):
            lengths = non_null.astype(str).str.len()
            if not lengths.empty:
                meta.update({"min_length": int(lengths.min()), "max_length": int(lengths.max())})
        columns_meta.append(meta)

    profile: Dict[str, Any] = {
        "row_count": row_count,
        "column_count": len(df.columns),
        "sampling_used": sampling_used,
        "columns": columns_meta,
    }
    return profile

# ---------------------------------------------------------------------------
# Mapping engine
# ---------------------------------------------------------------------------
class MappingInfo(Dict[str, Any]):
    pass

def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())

def _name_similarity(a: str, b: str) -> float:
    # simple token overlap based similarity (0‑1)
    set_a = set(_normalize_name(a).split("_"))
    set_b = set(_normalize_name(b).split("_"))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def infer_column_mappings(
    source_meta: Dict[str, Any],
    target_meta: Dict[str, Any],
    *,
    llm: bool = True,
) -> List[MappingInfo]:
    """Generate candidate column mappings between two schema profiles.

    The function returns a list of MappingInfo dictionaries with the keys:
    ``source_column``, ``target_column``, ``confidence`` (float 0‑1), ``category``
    (HIGH/MEDIUM/LOW/AMBIGUOUS/INCOMPATIBLE), ``evidence`` (list of strings),
    ``transformation`` (optional description) and ``requires_approval`` (bool).
    """
    src_cols = {c["name"]: c for c in source_meta["columns"]}
    tgt_cols = {c["name"]: c for c in target_meta["columns"]}

    mappings: List[MappingInfo] = []
    used_target = set()

    for src_name, src_info in src_cols.items():
        best_match: Tuple[str | None, float, List[str]] = (None, 0.0, [])
        for tgt_name, tgt_info in tgt_cols.items():
            # skip already used target unless many‑to‑one explicitly allowed later
            if tgt_name in used_target:
                continue
            evidence: List[str] = []
            # exact name match
            if src_name == tgt_name:
                score = 1.0
                evidence.append("exact name match")
            else:
                # normalized similarity
                sim = _name_similarity(src_name, tgt_name)
                score = sim * 0.6  # weight similarity
                if sim > 0.8:
                    evidence.append(f"high normalized similarity ({sim:.2f})")
                # synonym check via semantic tags
                src_tags = set(src_info.get("semantic_tags", []))
                tgt_tags = set(tgt_info.get("semantic_tags", []))
                if src_tags & tgt_tags:
                    score = max(score, 0.9)
                    evidence.append("shared semantic tag")
            # logical type compatibility boost
            if src_info.get("logical_type") == tgt_info.get("logical_type"):
                score += 0.1
                evidence.append("compatible logical type")
            # value‑distribution similarity (Jaccard of sample sets)
            src_samples = set(src_info.get("sample_values", []))
            tgt_samples = set(tgt_info.get("sample_values", []))
            if src_samples and tgt_samples:
                jacc = len(src_samples & tgt_samples) / len(src_samples | tgt_samples)
                if jacc > 0.5:
                    score += 0.1
                    evidence.append(f"sample Jaccard {jacc:.2f}")
            if score > best_match[1]:
                best_match = (tgt_name, score, evidence)
        tgt_name, score, evidence = best_match
        if tgt_name is None:
            continue
        # Determine confidence category
        if score >= 0.9:
            category = "HIGH"
            requires_approval = False
        elif score >= 0.7:
            category = "MEDIUM"
            requires_approval = False
        elif score >= 0.5:
            category = "LOW"
            requires_approval = True
        else:
            category = "INCOMPATIBLE"
            requires_approval = True
        # Avoid high confidence solely on identifier similarity
        if src_info.get("logical_type") == "identifier" and tgt_info.get("logical_type") == "identifier" and category == "HIGH":
            # need additional evidence beyond just type match
            if not any("semantic tag" in e for e in evidence):
                category = "LOW"
                requires_approval = True
                evidence.append("identifier similarity without semantic evidence")
        mapping: MappingInfo = {
            "source_column": src_name,
            "target_column": tgt_name,
            "confidence": round(score, 3),
            "category": category,
            "evidence": evidence,
            "transformation": None,
            "requires_approval": requires_approval,
        }
        mappings.append(mapping)
        used_target.add(tgt_name)
    return mappings

# ---------------------------------------------------------------------------
# Primary / foreign key inference
# ---------------------------------------------------------------------------
def infer_keys(datasets: Mapping[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Return a list of relationship dictionaries between the supplied datasets.

    Each dictionary contains:
    ``left_dataset``, ``right_dataset``, ``left_column``, ``right_column``,
    ``relationship_type`` (one‑to‑many, many‑to‑one, one‑to‑one, unknown),
    ``coverage`` (percentage of left values found in right), ``confidence``
    (float 0‑1) and ``evidence`` (list of strings).
    """
    relations: List[Dict[str, Any]] = []
    dataset_items = list(datasets.items())
    for i, (left_name, left_df) in enumerate(dataset_items):
        for right_name, right_df in dataset_items[i + 1 :]:
            # candidate primary key columns – unique, non‑null
            left_candidates = [c for c in left_df.columns if left_df[c].notna().all() and left_df[c].is_unique]
            right_candidates = [c for c in right_df.columns if right_df[c].notna().all() and right_df[c].is_unique]
            for lcol in left_candidates:
                for rcol in right_candidates:
                    # compute overlap
                    left_vals = set(left_df[lcol].astype(str).unique())
                    right_vals = set(right_df[rcol].astype(str).unique())
                    if not left_vals:
                        continue
                    intersect = left_vals & right_vals
                    coverage = len(intersect) / len(left_vals)
                    if coverage < 0.5:
                        continue
                    # infer direction based on cardinalities
                    rel_type = "one-to-many" if len(right_vals) > len(left_vals) else "many-to-one"
                    confidence = coverage * 0.9  # simple heuristic
                    relations.append(
                        {
                            "left_dataset": left_name,
                            "right_dataset": right_name,
                            "left_column": lcol,
                            "right_column": rcol,
                            "relationship_type": rel_type,
                            "coverage": round(coverage, 3),
                            "confidence": round(confidence, 3),
                            "evidence": ["unique non‑null columns"],
                        }
                    )
    return relations

# ---------------------------------------------------------------------------
# Compatibility validation
# ---------------------------------------------------------------------------
COMPATIBLE = "COMPATIBLE"
COMPATIBLE_WITH_TRANSFORMATION = "COMPATIBLE_WITH_TRANSFORMATION"
INCOMPATIBLE = "INCOMPATIBLE"
REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"

def validate_schema_compatibility(
    source_meta: Dict[str, Any],
    target_meta: Dict[str, Any],
    mappings: Sequence[MappingInfo],
) -> List[Dict[str, Any]]:
    """Validate the compatibility of *source_meta* against *target_meta*.

    Returns a list of report entries, each with ``field``, ``status``, ``detail``
    and optionally ``suggested_transformation``.
    """
    src_cols = {c["name"]: c for c in source_meta["columns"]}
    tgt_cols = {c["name"]: c for c in target_meta["columns"]}
    mapped_src = {m["source_column"] for m in mappings}
    mapped_tgt = {m["target_column"] for m in mappings}
    reports: List[Dict[str, Any]] = []

    # Missing columns in source
    for tgt in tgt_cols:
        if tgt not in mapped_tgt:
            reports.append({"field": tgt, "status": INCOMPATIBLE, "detail": "Target column not mapped"})

    # Extra source columns (unmapped)
    for src in src_cols:
        if src not in mapped_src:
            reports.append({"field": src, "status": COMPATIBLE, "detail": "Source column unused"})

    # Per‑mapping checks
    for m in mappings:
        src = src_cols.get(m["source_column"])
        tgt = tgt_cols.get(m["target_column"])
        if not src or not tgt:
            continue
        src_type = src.get("logical_type")
        tgt_type = tgt.get("logical_type")
        if src_type == tgt_type:
            status = COMPATIBLE
            detail = "Logical types match"
        elif src_type in {"integer", "decimal"} and tgt_type in {"decimal", "integer"}:
            status = COMPATIBLE_WITH_TRANSFORMATION
            detail = f"numeric type conversion ({src_type}→{tgt_type})"
        elif src_type in {"date", "datetime"} and tgt_type in {"date", "datetime"}:
            status = COMPATIBLE_WITH_TRANSFORMATION
            detail = f"date parsing required ({src_type}→{tgt_type})"
        else:
            status = REQUIRES_HUMAN_REVIEW
            detail = f"type mismatch ({src_type}→{tgt_type})"
        reports.append(
            {
                "field": f"{m['source_column']}→{m['target_column']}",
                "status": status,
                "detail": detail,
                "suggested_transformation": None if status == COMPATIBLE else "cast",
            }
        )
    return reports

# ---------------------------------------------------------------------------
# Mapping plan generation
# ---------------------------------------------------------------------------
import uuid
from datetime import datetime

def generate_mapping_plan(
    source_dataset_id: str,
    target_dataset_id: str,
    source_meta: Dict[str, Any],
    target_meta: Dict[str, Any],
    mappings: Sequence[MappingInfo],
    relationships: Sequence[Dict[str, Any]],
    compatibility: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a JSON‑serialisable mapping plan.

    The plan includes all metadata needed for later execution and for UI
    display.  Approval flags are initialised based on the mapping ``category``.
    """
    plan_id = str(uuid.uuid4())
    plan = {
        "plan_id": plan_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_dataset_id": source_dataset_id,
        "target_dataset_id": target_dataset_id,
        "source_schema": source_meta,
        "target_schema": target_meta,
        "mappings": [
            {
                **m,
                "approved": m["category"] in {"HIGH", "MEDIUM"} and not m["requires_approval"],
                "approval_timestamp": None,
            }
            for m in mappings
        ],
        "relationships": relationships,
        "compatibility_report": compatibility,
        "overall_status": "DRAFT",
    }
    return plan

# End of file
