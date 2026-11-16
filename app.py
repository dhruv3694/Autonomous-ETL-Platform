from __future__ import annotations

import io
import json
import sqlite3
import time
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

import os

from src.etl.data_loader import load_uploaded_files, summarize_dataframe
from src.etl.comparison_engine import (
    analyze_transformation_impact,
    compare_datasets,
    detect_record_level_changes,
    generate_markdown_comparison_report,
    validate_pipeline_equivalence,
    validate_user_requirements,
)
from src.etl.optimization_engine import (
    analyze_pipeline_structure,
    benchmark_pipeline_candidate,
    execute_user_python_code_safely,
    generate_optimization_candidates,
    validate_user_python_code,
    validate_user_sql_code,
)
from src.etl.isolated_runner import execute_python_isolated_process
from src.etl.execution_engine import SQLExecutionError, execute_plan, validate_dataframe
from src.etl.llm_service import (
    explain_comparison_results,
    generate_sql_optimization_advice,
    get_llm_status,
    resolve_model_config,
    run_model_inference,
)
from src.etl.pipeline_generator import (
    build_dataset_context,
    dataset_id_for_index,
    execute_pipeline,
    generate_pipeline_specification,
    update_pipeline_operation_columns,
    validate_pipeline,
)
from src.etl.performance_profiler import identify_bottlenecks, system_resource_snapshot
from src.etl.sql_generator import generate_sql_operations
from src.etl.sql_optimizer import (
    analyze_sql_plan,
    benchmark_sql_candidate,
    generate_sql_candidates,
)
from src.etl.workflow_state import list_history, list_snapshots, record_event, save_snapshot


st.set_page_config(page_title="AI ETL Generator", page_icon="🧠", layout="wide")


def _dataset_label(dataset: Dict[str, Any]) -> str:
    file_name = dataset.get("file_name", "Dataset")
    source_reference = dataset.get("sheet") or dataset.get("table") or "file"
    return f"{file_name} ({source_reference})"


def _serialize_run(run: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if run is None:
        return None
    return {
        "original": {
            "columns": list(run["original"].columns),
            "records": run["original"].to_dict(orient="records"),
        },
        "transformed": {
            "columns": list(run["transformed"].columns),
            "records": run["transformed"].to_dict(orient="records"),
        },
        "audit": run.get("audit", []),
        "validation": run.get("validation", {}),
        "performance": run.get("performance", {}),
    }


def _restore_run(payload: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not payload:
        return None
    original = pd.DataFrame(payload.get("original", {}).get("records", []), columns=payload.get("original", {}).get("columns", []))
    transformed = pd.DataFrame(payload.get("transformed", {}).get("records", []), columns=payload.get("transformed", {}).get("columns", []))
    return {
        "original": original,
        "transformed": transformed,
        "audit": payload.get("audit", []),
        "validation": payload.get("validation", {}),
        "performance": payload.get("performance", {}),
    }


def _default_llm_config() -> Dict[str, Any]:
    return {
        "provider": "ollama",
        "model": "qwen3:4b",
        "model_path": "",
        "cache_dir": "",
        "device": "auto",
        "token": "",
        "max_tokens": 512,
        "use_4bit": False,
        "model_loaded": False,
        "real_inference_success": False,
        "heuristic_fallback_active": False,
        "last_inference_error": None,
    }


def _ensure_llm_session_state() -> None:
    defaults = _default_llm_config()
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    st.session_state["llm_provider"] = st.session_state.get("llm_provider", defaults["provider"])
    st.session_state["llm_model"] = st.session_state.get("llm_model", defaults["model"])
    st.session_state["llm_model_path"] = st.session_state.get("llm_model_path", defaults["model_path"])
    st.session_state["llm_cache_dir"] = st.session_state.get("llm_cache_dir", defaults["cache_dir"])
    st.session_state["llm_device"] = st.session_state.get("llm_device", defaults["device"])
    st.session_state["llm_token"] = st.session_state.get("llm_token", defaults["token"])
    st.session_state["llm_max_tokens"] = st.session_state.get("llm_max_tokens", defaults["max_tokens"])
    st.session_state["llm_use_4bit"] = st.session_state.get("llm_use_4bit", defaults["use_4bit"])


def _apply_llm_session_configuration() -> Dict[str, Any]:
    _ensure_llm_session_state()
    config = {
        "provider": st.session_state.get("llm_provider", "ollama"),
        "model": st.session_state.get("llm_model", "qwen3:4b"),
        "model_path": st.session_state.get("llm_model_path", ""),
        "cache_dir": st.session_state.get("llm_cache_dir", ""),
        "device": st.session_state.get("llm_device", "auto"),
        "token": st.session_state.get("llm_token", ""),
        "max_tokens": int(st.session_state.get("llm_max_tokens", 512)),
        "use_4bit": bool(st.session_state.get("llm_use_4bit", False)),
    }

    if config["provider"] == "ollama" and not config["model"]:
        config["model"] = "qwen3:4b"
    if config["provider"] == "huggingface" and not config["model"]:
        config["model"] = "Qwen/Qwen2.5-3B-Instruct"

    os.environ["ETL_LLM_PROVIDER"] = config["provider"]
    os.environ["ETL_LLM_MODEL"] = config["model"]
    os.environ["ETL_LLM_MODEL_PATH"] = config["model_path"] or ""
    if config["cache_dir"]:
        os.environ["ETL_LLM_CACHE_DIR"] = config["cache_dir"]
    os.environ["ETL_LLM_DEVICE"] = config["device"]
    os.environ["ETL_LLM_MAX_TOKENS"] = str(config["max_tokens"])
    os.environ["ETL_LLM_USE_4BIT"] = str(bool(config["use_4bit"]))
    if config["token"]:
        os.environ["HF_TOKEN"] = config["token"]
        os.environ["HUGGINGFACE_TOKEN"] = config["token"]
        os.environ["ETL_LLM_TOKEN"] = config["token"]
    else:
        for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "ETL_LLM_TOKEN"):
            os.environ.pop(key, None)

    return config


def _render_ai_model_configuration() -> None:
    _ensure_llm_session_state()
    config = _apply_llm_session_configuration()
    st.header("AI Model Configuration")
    st.caption("Select a provider, set model details, and test real local inference without logging sensitive tokens.")

    with st.form("llm_configuration_form"):
        provider = st.selectbox(
            "LLM provider",
            ["ollama", "huggingface"],
            index=0 if config["provider"] == "ollama" else 1,
        )
        model = st.text_input(
            "Model name",
            value=config["model"],
            help="Examples: qwen3:4b for Ollama, Qwen/Qwen2.5-3B-Instruct for Hugging Face.",
        )
        model_path = st.text_input("Model path (optional)", value=config["model_path"], help="Optional local directory or model snapshot for Hugging Face.")
        cache_dir = st.text_input("Model cache directory (optional)", value=config["cache_dir"], help="Directory used to store downloaded models.")
        device = st.selectbox("Inference device", ["auto", "cpu", "cuda"], index=["auto", "cpu", "cuda"].index(config["device"] if config["device"] in {"auto", "cpu", "cuda"} else "auto"))
        token = st.text_input("Hugging Face token", value=config["token"], type="password", help="Only needed for gated models. The value is never displayed after entry.")
        max_tokens = st.number_input("Max tokens", min_value=64, max_value=2048, value=int(config["max_tokens"]))
        use_4bit = st.checkbox("Use 4-bit model loading where supported", value=bool(config["use_4bit"]))
        submitted = st.form_submit_button("Apply model configuration")

    if submitted:
        st.session_state.llm_provider = provider
        st.session_state.llm_model = model
        st.session_state.llm_model_path = model_path
        st.session_state.llm_cache_dir = cache_dir
        st.session_state.llm_device = device
        st.session_state.llm_token = token
        st.session_state.llm_max_tokens = max_tokens
        st.session_state.llm_use_4bit = use_4bit
        _apply_llm_session_configuration()
        st.success("Model configuration updated.")

    status = get_llm_status(
        provider=st.session_state.get("llm_provider", "ollama"),
        model=st.session_state.get("llm_model", "qwen3:4b"),
        config=resolve_model_config(
            provider=st.session_state.get("llm_provider", "ollama"),
            model=st.session_state.get("llm_model", "qwen3:4b"),
            model_path=st.session_state.get("llm_model_path", ""),
            cache_dir=st.session_state.get("llm_cache_dir", ""),
            token=st.session_state.get("llm_token", ""),
            device=st.session_state.get("llm_device", "auto"),
            max_tokens=int(st.session_state.get("llm_max_tokens", 512)),
            use_4bit=bool(st.session_state.get("llm_use_4bit", False)),
        ),
    )

    status_df = pd.DataFrame(
        [
            ("Selected provider", status.get("provider")),
            ("Selected model", status.get("model")),
            ("Provider configured", status.get("provider_configured")),
            ("Model configured", status.get("model_configured")),
            ("Model downloaded", status.get("model_downloaded")),
            ("Model loaded", status.get("model_loaded")),
            ("Real inference successful", status.get("real_inference_success")),
            ("Heuristic fallback active", status.get("heuristic_fallback_active")),
            ("Provider available", status.get("provider_available")),
            ("Transformers available", status.get("huggingface_available")),
            ("PyTorch available", status.get("huggingface_pytorch_available")),
            ("CUDA available", status.get("cuda_available")),
            ("Device", status.get("device")),
            ("Fallback status", status.get("fallback_status", "not_needed")),
            ("Inference error", status.get("last_inference_error") or "None"),
        ],
        columns=["Field", "Value"],
    )
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    with st.form("llm_inference_test_form"):
        test_model_clicked = st.form_submit_button("Test Model")

    if test_model_clicked:
        prompt = "Analyze the following sample schema and suggest appropriate ETL data-quality checks: customer_id INTEGER, email TEXT, order_date TEXT, amount REAL."
        results = run_model_inference(
            prompt,
            provider=st.session_state.get("llm_provider", "ollama"),
            model=st.session_state.get("llm_model", "qwen3:4b"),
            config=resolve_model_config(
                provider=st.session_state.get("llm_provider", "ollama"),
                model=st.session_state.get("llm_model", "qwen3:4b"),
                model_path=st.session_state.get("llm_model_path", ""),
                cache_dir=st.session_state.get("llm_cache_dir", ""),
                token=st.session_state.get("llm_token", ""),
                device=st.session_state.get("llm_device", "auto"),
                max_tokens=int(st.session_state.get("llm_max_tokens", 512)),
                use_4bit=bool(st.session_state.get("llm_use_4bit", False)),
            ),
        )
        st.session_state.model_loaded = results.get("model_loaded", False)
        st.session_state.real_inference_success = results.get("real_inference_success", False)
        st.session_state.heuristic_fallback_active = results.get("heuristic_fallback_active", False)
        st.session_state.last_inference_error = results.get("inference_error")

        if results.get("real_inference_success"):
            st.success(f"Real inference succeeded in {results.get('duration_seconds', 0.0):.2f}s.")
            st.code(results.get("response", ""), language="text")
        else:
            st.warning("Real inference did not succeed.")
            if results.get("inference_error"):
                st.error(f"Inference error: {results.get('inference_error')}")
            if results.get("heuristic_fallback_active"):
                st.info("Heuristic fallback was used because real inference was unavailable or failed. This is not a successful LLM inference result.")
                st.code(results.get("response", ""), language="text")


def _render_dataset_preview(dataset: Dict[str, Any]) -> None:
    with st.container():
        file_label = dataset.get("file_name", "Dataset")
        source_label = dataset.get("sheet") or dataset.get("table") or "file"
        st.subheader(f"{file_label} — {source_label}")

        frame = dataset.get("dataframe", pd.DataFrame())
        if frame.empty:
            st.warning(dataset.get("summary", {}).get("error", "No data to display."))
            return

        preview = frame.head(10)
        st.dataframe(preview, use_container_width=True)

        summary = dataset.get("summary", summarize_dataframe(frame))
        cols = st.columns(5)
        metrics = [
            ("Rows", summary.get("row_count", len(frame))),
            ("Columns", summary.get("column_count", len(frame.columns))),
            ("Missing", summary.get("null_count", int(frame.isna().sum().sum()))),
            ("Duplicates", summary.get("duplicate_rows", int(frame.duplicated().sum()))),
            ("Memory", f"{summary.get('memory_mb', 0)} MB"),
        ]
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.metric(label, value)


def _render_plan(plan: Dict[str, Any], dataframe: pd.DataFrame, mode: str) -> None:
    st.subheader("Generated ETL execution plan")
    steps = plan.get("steps", [])
    if not steps:
        st.info("No steps were produced.")
        return

    for step in steps:
        with st.expander(f"{step.get('id', 'step')} — {step.get('name', 'ETL step')}", expanded=True):
            st.write(step.get("description", ""))
            st.write("Risk:", step.get("risk", "low"))
            st.write("Columns:", ", ".join(step.get("columns", [])) if step.get("columns") else "All columns")
            if step.get("operation"):
                st.write("Operation:", step.get("operation"))
            if mode == "sql":
                st.write("Operation:", step.get("operation", "unknown"))
            else:
                st.code(step.get("python", "df = df.copy()"), language="python")
    if mode == "sql":
        try:
            _, operations = generate_sql_operations(dataframe, plan)
        except (TypeError, ValueError) as exc:
            st.error(f"SQL plan validation failed: {exc}")
        else:
            st.markdown("#### Generated SQLite statements")
            for operation in operations:
                st.code(operation.sql, language="sql")


def _render_ai_pipeline(pipeline: Dict[str, Any]) -> None:
    st.header("AI ETL Pipeline Generator")
    st.caption(
        f"Provider: {pipeline.get('provider', 'unknown')} | "
        f"Model: {pipeline.get('model', 'unknown')} | "
        f"Plan source: {pipeline.get('source', 'unknown')}"
    )
    if pipeline.get("heuristic_fallback_active"):
        st.warning(
            "The configured model did not produce a usable structured plan. "
            "This plan uses deterministic heuristic fallback and is not an LLM-generated plan."
        )
        if pipeline.get("inference_error"):
            st.caption(f"Fallback reason: {pipeline['inference_error']}")
    elif pipeline.get("real_inference_success"):
        st.success("A real model response was received and passed pipeline validation.")
    if pipeline.get("clarification_required"):
        st.warning("Clarification is required before execution:")
        for item in pipeline.get("ambiguities", []):
            st.write(f"- {item}")

    st.markdown("#### Dataset profiles provided to planning")
    for context in pipeline.get("dataset_context", []):
        with st.expander(
            f"{context['dataset_id']} — {context['label']}",
            expanded=False,
        ):
            st.write(
                f"Format: {context['format']} | Rows: {context['row_count']} | "
                f"Columns: {context['column_count']} | Duplicates: {context['duplicate_rows']} | "
                f"Missing values: {context['null_count']}"
            )
            st.dataframe(pd.DataFrame(context.get("columns", [])), use_container_width=True, hide_index=True)
            if context.get("possible_primary_keys"):
                st.caption(f"Possible unique keys (inferred): {', '.join(context['possible_primary_keys'])}")
            if context.get("possible_relationships"):
                st.write("Candidate relationships (not verified):")
                st.dataframe(pd.DataFrame(context["possible_relationships"]), use_container_width=True, hide_index=True)

    operations = pipeline.get("operations", [])
    st.markdown("#### Operation sequence and dependencies")
    st.write(" → ".join(pipeline.get("execution_order", [])) or "No executable operations.")
    edges = pipeline.get("edges", [])
    if operations:
        graph_lines = ["digraph pipeline {", "rankdir=LR;"]
        for operation in operations:
            graph_lines.append(f"{json.dumps(operation['id'])} [label={json.dumps(operation['name'])}];")
        for edge in edges:
            graph_lines.append(f"{json.dumps(edge['from'])} -> {json.dumps(edge['to'])};")
        graph_lines.append("}")
        st.graphviz_chart("\n".join(graph_lines), use_container_width=True)
    elif operations:
        st.caption("Operations have no explicit dependencies and are independent within their dataset.")

    for operation in operations:
        with st.expander(
            f"{operation['id']} — {operation['name']} ({operation['risk']} risk)",
            expanded=True,
        ):
            st.write(operation["description"])
            st.write(f"Dataset: {operation['input_dataset']}")
            st.write(f"Operation: {operation['operation']} | Engine: {operation['engine']}")
            st.write(f"Input columns: {', '.join(operation['input_columns']) or 'all columns'}")
            st.write(f"Output columns: {', '.join(operation['output_columns'])}")
            st.write(f"Dependencies: {', '.join(operation['dependencies']) or 'none'}")
            st.write(f"Expected output: {operation['expected_output']}")
            st.write("Validation rules:")
            for rule in operation["validation_rules"]:
                st.write(f"- {rule}")
            st.code(
                operation["sql"] if operation["engine"] == "sql" else operation["python"],
                language="sql" if operation["engine"] == "sql" else "python",
            )


def _render_pipeline_comparison_studio(
    original: pd.DataFrame,
    transformed: pd.DataFrame,
    last_run: Dict[str, Any],
    plan: Optional[Dict[str, Any]] = None,
    instruction: str = "",
) -> None:
    st.markdown("---")
    st.header("Pipeline Comparison & Validation")

    comparison = compare_datasets(original, transformed, dataset_name=st.session_state.get("sql_dataset_label") or "active_dataset")
    audit = last_run.get("audit", [])
    safe_plan = plan or st.session_state.get("plan") or {}
    impacts = analyze_transformation_impact(audit, safe_plan)
    record_changes = detect_record_level_changes(original, transformed)
    requirements = validate_user_requirements(
        instruction,
        safe_plan,
        transformed,
        last_run.get("validation", {}),
        audit,
    )

    # Optional optimization candidate equivalence
    equivalence: Optional[Dict[str, Any]] = None
    if st.session_state.get("sql_candidate") and st.session_state.get("sql_benchmark"):
        # Check candidate output vs baseline output if benchmark was run
        pass

    # AI Explanation Section
    with st.expander("AI Data Quality & Comparison Advisor", expanded=True):
        col_exp1, col_exp2 = st.columns([3, 1])
        with col_exp2:
            request_ai = st.button("Generate AI Explanation", key="btn_ai_comparison_explain")
        with col_exp1:
            if request_ai or "comparison_explanation" in st.session_state:
                if request_ai:
                    with st.spinner("Analyzing comparison metrics..."):
                        expl = explain_comparison_results(
                            comparison,
                            provider=st.session_state.get("llm_provider"),
                            model=st.session_state.get("llm_model"),
                        )
                        st.session_state.comparison_explanation = expl
                explanation = st.session_state.get("comparison_explanation")
                if explanation:
                    st.caption(f"Source: {explanation.get('source')} | Provider: {explanation.get('provider')}")
                    st.write(f"**Summary:** {explanation.get('summary')}")
                    st.write(f"**Quality Assessment:** {explanation.get('quality_assessment')}")
                    if explanation.get("recommendations"):
                        st.markdown("**Recommendations:**")
                        for rec in explanation["recommendations"]:
                            st.write(f"- {rec}")
            else:
                st.info("Click 'Generate AI Explanation' to request an architectural review of this transformation from the configured LLM.")

    # Multi-View Tabs
    tab_overview, tab_schema, tab_dq, tab_records, tab_impact, tab_equiv, tab_final, tab_export = st.tabs([
        "Dataset Overview",
        "Schema Comparison",
        "Data Quality Comparison",
        "Record-Level Changes",
        "Transformation Impact",
        "Pipeline Equivalence",
        "Final Validation Report",
        "Export Reports",
    ])

    with tab_overview:
        st.subheader("Dataset Overview & Metrics")
        rm = comparison["row_metrics"]
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Original Rows", rm["before_rows"])
        m_col2.metric("Transformed Rows", rm["after_rows"], delta=rm["row_delta"])
        m_col3.metric("Duplicates Removed", rm["duplicates_removed"])
        m_col4.metric("Row Change %", f"{rm['row_loss_pct']}%")

        if rm["unexpected_data_loss"]:
            st.error("ALERT: Unexpected significant data loss detected!")

        col_orig, col_trans = st.columns(2)
        with col_orig:
            st.markdown("##### Original Dataset Preview")
            st.dataframe(original.head(10), use_container_width=True)
        with col_trans:
            st.markdown("##### Transformed Dataset Preview")
            st.dataframe(transformed.head(10), use_container_width=True)

    with tab_schema:
        st.subheader("Schema & Column Differences")
        sc = comparison["schema_comparison"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Added Columns ({len(sc['added_columns'])})**")
            st.write(", ".join(sc["added_columns"]) or "None")
        with c2:
            st.markdown(f"**Removed Columns ({len(sc['removed_columns'])})**")
            st.write(", ".join(sc["removed_columns"]) or "None")
        with c3:
            st.markdown(f"**Renamed Columns ({len(sc['renamed_columns'])})**")
            st.json(sc["renamed_columns"] if sc["renamed_columns"] else "None")

        if sc["dtype_changes"]:
            st.markdown("##### Data Type Changes")
            st.json(sc["dtype_changes"])
        else:
            st.info("No column data type alterations detected.")

    with tab_dq:
        st.subheader("Data Quality Comparison")
        dq = comparison["data_quality"]
        b_dq = dq["before"]
        a_dq = dq["after"]

        q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)
        q_col1.metric("Completeness", f"{a_dq['completeness_score']}%", delta=round(a_dq['completeness_score'] - b_dq['completeness_score'], 2))
        q_col2.metric("Uniqueness", f"{a_dq['uniqueness_score']}%", delta=round(a_dq['uniqueness_score'] - b_dq['uniqueness_score'], 2))
        q_col3.metric("Validity", f"{a_dq['validity_score']}%", delta=round(a_dq['validity_score'] - b_dq['validity_score'], 2))
        q_col4.metric("Consistency", f"{a_dq['consistency_score']}%", delta=round(a_dq['consistency_score'] - b_dq['consistency_score'], 2))
        q_col5.metric("Overall Score", f"{a_dq['overall_score']}%", delta=round(a_dq['overall_score'] - b_dq['overall_score'], 2))

        if dq["improvements"]:
            st.success("Quality Improvements:\n" + "\n".join(f"- {i}" for i in dq["improvements"]))
        if dq["regressions"]:
            st.warning("Quality Regressions:\n" + "\n".join(f"- {r}" for r in dq["regressions"]))
        if dq["remaining_issues"]:
            st.info("Remaining Quality Issues:\n" + "\n".join(f"- {issue}" for issue in dq["remaining_issues"][:5]))

    with tab_records:
        st.subheader("Record-Level Change Detection")
        rc_counts = record_changes["counts"]
        st.caption(f"Comparison Strategy: `{record_changes['strategy']}`" + (f" (Keys: {record_changes['key_columns']})" if record_changes.get('key_columns') else ""))
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        r_col1.metric("Unchanged", rc_counts["unchanged"])
        r_col2.metric("Modified", rc_counts["modified"])
        r_col3.metric("Added", rc_counts["added"])
        r_col4.metric("Removed", rc_counts["removed"])

        if record_changes.get("samples") and "modified" in record_changes["samples"] and record_changes["samples"]["modified"]:
            st.markdown("##### Modified Record Samples")
            st.json(record_changes["samples"]["modified"][:5])

    with tab_impact:
        st.subheader("Transformation Operation Impact")
        if impacts:
            impact_df = pd.DataFrame(impacts)[["operation_name", "operation_type", "input_rows", "output_rows", "rows_affected", "execution_duration_seconds", "explanation"]]
            st.dataframe(impact_df, use_container_width=True, hide_index=True)
        else:
            st.info("No operation audit log available.")

    with tab_equiv:
        st.subheader("Pipeline Optimization Equivalence Validation")
        st.write("Ensure optimized candidates strictly preserve original approved outputs.")
        eq_bench = st.session_state.get("sql_benchmark")
        if eq_bench and "equivalence" in eq_bench:
            eq_info = eq_bench["equivalence"]
            is_eq = eq_info.get("equivalent", False)
            if is_eq:
                st.success("Candidate Pipeline is STRICTLY EQUIVALENT to the baseline pipeline.")
            else:
                st.error("Candidate Pipeline is NOT EQUIVALENT to the baseline pipeline.")
            st.json(eq_info)
        else:
            st.info("Generate and benchmark a SQL candidate in the SQL Optimization Studio to view strict equivalence validation.")

    with tab_final:
        st.subheader("Final User Requirement Validation")
        st.write(f"**Instruction:** \"{requirements['instruction']}\"")
        if requirements["overall_fulfilled"]:
            st.success("All user requirements were fulfilled according to execution and validation evidence.")
        else:
            st.warning("Some requirements were partially fulfilled, unresolved, or pending.")

        for req in requirements.get("requirements", []):
            st.write(f"- **{req['requirement']}:** `{req['status']}` — {req['details']}")

    with tab_export:
        st.subheader("Export Validation Reports")
        report_md = generate_markdown_comparison_report(
            comparison,
            record_changes=record_changes,
            impacts=impacts,
            requirements=requirements,
        )
        st.download_button(
            "Download Comprehensive Markdown Report",
            report_md,
            file_name=f"{comparison.get('dataset_name', 'etl')}_comparison_report.md",
            mime="text/markdown",
        )
        report_json = json.dumps(
            {
                "comparison": comparison,
                "record_changes": record_changes,
                "impacts": impacts,
                "requirements": requirements,
            },
            indent=2,
            default=str,
        )
        st.download_button(
            "Download Validation Report JSON",
            report_json,
            file_name=f"{comparison.get('dataset_name', 'etl')}_validation_report.json",
            mime="application/json",
        )


def _render_sql_optimization_studio(dataset: Dict[str, Any], plan: Dict[str, Any]) -> None:
    st.markdown("---")
    with st.expander("AI-Powered ETL Optimization Studio", expanded=True):
        st.caption(
            "Multi-mode optimization engine: Analyze, benchmark, and optimize AI-generated pipelines, "
            "user-provided Python/SQL code, or saved history pipelines with strict equivalence verification."
        )

        opt_tab1, opt_tab2, opt_tab3 = st.tabs([
            "Mode A: Optimize Active Pipeline",
            "Mode B: User-Provided Pipeline Code",
            "Mode C: Existing Project Pipelines",
        ])

        # -------------------------------------------------------------
        # TAB 1: MODE A — Active Pipeline Optimization
        # -------------------------------------------------------------
        with opt_tab1:
            st.markdown("##### Optimize Active Pipeline (Python / SQL)")
            mode = st.session_state.get("execution_mode", "python")
            df = dataset.get("dataframe")

            if st.button("Run Comprehensive Pipeline Analysis", key="btn_run_active_analysis"):
                with st.spinner("Analyzing pipeline structure and identifying bottlenecks..."):
                    p_analysis = analyze_pipeline_structure(plan, dataframe=df)
                    st.session_state.pipeline_struct_analysis = p_analysis
                    st.session_state.active_opt_candidates = generate_optimization_candidates(plan, df, execution_mode=mode)
                    st.session_state.selected_active_candidate = None
                    st.session_state.active_candidate_benchmark = None

            if "pipeline_struct_analysis" in st.session_state:
                p_analysis = st.session_state.pipeline_struct_analysis
                st.write(f"Total Operations: **{p_analysis['operation_count']}** | Operation Types: {p_analysis['operation_types']}")
                if p_analysis["identified_bottlenecks"]:
                    st.markdown("**Identified Bottlenecks & Opportunities:**")
                    for bn in p_analysis["identified_bottlenecks"]:
                        st.write(f"- `{bn['category']}`: {bn['evidence']}")
                else:
                    st.info("Pipeline structure is lean with no redundant intermediate validation steps.")

            candidates = st.session_state.get("active_opt_candidates", [])
            if candidates:
                st.markdown("##### Optimization Candidates Generated")
                cand_labels = [f"[{c['candidate_id']}] {c['description']} ({c['category']})" for c in candidates]
                chosen_idx = st.selectbox("Select Candidate to Benchmark", range(len(candidates)), format_func=lambda i: cand_labels[i])
                cand = candidates[chosen_idx]
                st.write(f"**Expected Benefit:** {cand['expected_benefit']}")
                st.write(f"**Risks:** {cand['risks']}")
                st.json(cand["plan"])

                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button("Benchmark Candidate (3 Runs)", key=f"btn_bench_active_{cand['candidate_id']}"):
                        with st.spinner("Running alternating benchmark runs with output equivalence validation..."):
                            bench_res = benchmark_pipeline_candidate(df, plan, cand["plan"], execution_mode=mode, repeats=3)
                            st.session_state.active_candidate_benchmark = bench_res

                if st.session_state.get("active_candidate_benchmark"):
                    b_res = st.session_state.active_candidate_benchmark
                    st.markdown("##### Benchmark & Validation Report")
                    st.json(b_res)
                    if b_res.get("equivalence", {}).get("is_equivalent"):
                        st.success(f"Candidate output is STRICTLY EQUIVALENT. Speed change: {b_res.get('speed_improvement_pct')}%")
                        if st.button("Apply Approved Optimized Candidate", key="btn_apply_active_candidate"):
                            st.session_state.plan = cand["plan"]
                            st.session_state.approved_plan = True
                            st.session_state.active_candidate_benchmark = None
                            st.success("Optimized pipeline applied as the active approved plan.")
                            st.rerun()
                    else:
                        st.error(f"Candidate rejected: {b_res.get('status')}")

        # -------------------------------------------------------------
        # TAB 2: MODE B — User-Provided Pipeline Code
        # -------------------------------------------------------------
        with opt_tab2:
            st.markdown("##### Optimize User-Provided Python / SQL Code")
            st.caption(
                "Security Boundary: Code is AST-parsed and executed in an isolated child process via Parquet IPC. "
                "No process memory or application secrets are shared. Operating system resource limits apply."
            )
            code_type = st.radio("Language", ["Python", "SQL"], horizontal=True, key="user_code_type")
            user_code = st.text_area("Paste Python / SQL pipeline transformation code:", height=150, key="user_code_input")

            if st.button("Validate & Execute in Isolated Process", key="btn_validate_user_code"):
                if code_type == "Python":
                    val_res = validate_user_python_code(user_code)
                    if not val_res["valid"]:
                        st.error(f"Security / AST validation failed: {val_res['error']}")
                    else:
                        st.success("Security & syntax checks passed. Launching isolated child process...")
                        with st.spinner("Executing transformation in isolated child process..."):
                            exec_res = execute_python_isolated_process(user_code, dataset["dataframe"], timeout_seconds=10.0)
                        if exec_res["outcome"] == "VALIDATED":
                            st.success(f"Isolated execution succeeded in {exec_res.get('duration_seconds')}s!")
                            st.dataframe(exec_res["dataframe"].head(10), use_container_width=True)
                        elif exec_res["outcome"] == "EXECUTION_TIMEOUT":
                            st.error(f"Timeout: {exec_res['error']}")
                        else:
                            st.error(f"Execution failed: {exec_res['error']}")
                else:
                    val_res = validate_user_sql_code(user_code)
                    if not val_res["valid"]:
                        st.error(f"SQL security validation failed: {val_res['error']}")
                    else:
                        st.success("SQL parsed with sqlparse AST. Single allowed query statement verified.")
                        st.code(user_code, language="sql")
                        st.info("Query analysis complete: Read-only verification passed. No DDL/DML mutation detected.")

        # -------------------------------------------------------------
        # TAB 3: MODE C — Existing / Historical Pipelines
        # -------------------------------------------------------------
        with opt_tab3:
            st.markdown("##### Optimize Historical Workflow Pipelines")
            hist_snapshots = list_snapshots(5)
            if hist_snapshots:
                hist_labels = [f"Snapshot {s['id']}: {s['dataset_label']} ({s['created_at']})" for s in hist_snapshots]
                chosen_hist = st.selectbox("Select historical pipeline to optimize:", hist_labels, key="sel_hist_opt")
                sel_snap = hist_snapshots[hist_labels.index(chosen_hist)]
                st.write(f"Instruction: *\"{sel_snap.get('instruction', '')}\"*")
                st.json(sel_snap.get("plan") or {})
                if st.button("Load this historical pipeline into Studio for optimization", key="btn_load_hist_opt"):
                    st.session_state.plan = sel_snap.get("plan")
                    st.session_state.execution_mode = sel_snap.get("execution_mode", "python")
                    st.success("Loaded historical pipeline. Switch to Mode A tab to inspect and optimize.")
                    st.rerun()
            else:
                st.info("No saved snapshots in history yet.")



def _render_dashboard(original: pd.DataFrame, transformed: pd.DataFrame) -> None:
    st.subheader("Before vs after comparison dashboard")

    original_summary = summarize_dataframe(original)
    transformed_summary = summarize_dataframe(transformed)

    left, middle, right = st.columns(3)
    with left:
        st.metric("Original rows", original_summary["row_count"])
        st.metric("Original nulls", original_summary["null_count"])
    with middle:
        st.metric("Final rows", transformed_summary["row_count"])
        st.metric("Final nulls", transformed_summary["null_count"])
    with right:
        st.metric("Original dupes", original_summary["duplicate_rows"])
        st.metric("Final dupes", transformed_summary["duplicate_rows"])

    comparison = pd.DataFrame(
        {
            "Metric": ["Rows", "Columns", "Missing values", "Duplicate rows"],
            "Original": [
                original_summary["row_count"],
                original_summary["column_count"],
                original_summary["null_count"],
                original_summary["duplicate_rows"],
            ],
            "Final": [
                transformed_summary["row_count"],
                transformed_summary["column_count"],
                transformed_summary["null_count"],
                transformed_summary["duplicate_rows"],
            ],
        }
    )
    st.dataframe(comparison, use_container_width=True)

    chart = px.bar(
        comparison,
        x="Metric",
        y=["Original", "Final"],
        barmode="group",
        title="Transformation summary",
    )
    st.plotly_chart(chart, use_container_width=True)


def _render_performance(dataset: Dict[str, Any], run: Dict[str, Any]) -> None:
    st.subheader("ETL Performance Intelligence")
    performance = run.get("performance", {})
    operations = performance.get("operations", [])

    total_seconds = performance.get("total_duration_seconds", 0.0)
    peak_rss = max(
        (item.get("sampled_peak_rss_bytes") or 0 for item in operations),
        default=0,
    )
    process_cpu = sum(item.get("process_cpu_seconds", 0.0) or 0.0 for item in operations)
    rows = len(run.get("original", []))
    throughput = rows / total_seconds if total_seconds else 0.0
    bottlenecks = identify_bottlenecks(operations)

    metric_columns = st.columns(5)
    metric_columns[0].metric("Execution time", f"{total_seconds:.4f} s")
    metric_columns[1].metric("Sampled peak RSS", f"{peak_rss / (1024 * 1024):.2f} MB")
    metric_columns[2].metric("Process CPU", f"{process_cpu:.4f} s")
    metric_columns[3].metric("Throughput", f"{throughput:,.1f} rows/s")
    metric_columns[4].metric("Bottlenecks", len(bottlenecks))

    st.caption(
        "RSS peak is sampled at 20 ms intervals and includes process-wide memory. "
        "CPU time is process CPU time. Disk/network counters are system-wide snapshots, "
        "not attributed to this ETL run; GPU utilization is unavailable."
    )

    extraction_profile = dataset.get("performance", {})
    if extraction_profile:
        st.markdown("#### Extraction and schema profiling")
        input_metrics = [
            extraction_profile.get("extraction", {}),
            extraction_profile.get("schema_analysis", {}),
        ]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Stage": item.get("stage"),
                        "Name": item.get("name"),
                        "Duration (s)": item.get("duration_seconds"),
                        "Rows out": item.get("records_out"),
                        "DataFrame bytes": item.get("dataframe_output_bytes"),
                        "RSS before (bytes)": item.get("rss_bytes_before"),
                        "Sampled peak RSS (bytes)": item.get("sampled_peak_rss_bytes"),
                    }
                    for item in input_metrics
                    if item
                ]
            ),
            use_container_width=True,
        )

    st.markdown("#### Operation-level profile")
    if operations:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Stage": item.get("stage"),
                        "Operation": item.get("name"),
                        "Status": item.get("status"),
                        "Duration (s)": item.get("duration_seconds"),
                        "CPU (s)": item.get("process_cpu_seconds"),
                        "CPU (% of one core)": item.get("process_cpu_percent_of_one_core"),
                        "Rows in": item.get("records_in"),
                        "Rows out": item.get("records_out"),
                        "Rows/s": item.get("records_per_second"),
                        "Input bytes": item.get("dataframe_input_bytes"),
                        "Output bytes": item.get("dataframe_output_bytes"),
                        "Sampled peak RSS (bytes)": item.get("sampled_peak_rss_bytes"),
                        "Python peak allocations (bytes)": item.get("python_peak_allocation_bytes"),
                    }
                    for item in operations
                ]
            ),
            use_container_width=True,
        )
    else:
        st.info("No operation-level metrics were recorded for this run.")

    st.markdown("#### Evidence-based bottlenecks")
    if bottlenecks:
        st.dataframe(pd.DataFrame(bottlenecks), use_container_width=True)
    else:
        st.info("No operation exceeded 25% of measured operation time.")

    resources_before = performance.get("resources_before")
    resources_after = performance.get("resources_after")
    if resources_before and resources_after:
        with st.expander("System resource snapshots"):
            st.write("Before execution")
            st.json(resources_before)
            st.write("After execution")
            st.json(resources_after)


def main() -> None:
    st.title("AI-Powered Autonomous ETL Pipeline Generator")
    st.caption("Local-first ETL workspace with human approval gates, persistent workflow history, SQL/Python execution modes, and exportable outputs.")

    if "datasets" not in st.session_state:
        st.session_state.datasets = []
    if "plan" not in st.session_state:
        st.session_state.plan = None
    if "approved_plan" not in st.session_state:
        st.session_state.approved_plan = False
    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    if "execution_mode" not in st.session_state:
        st.session_state.execution_mode = "python"
    if "active_dataset_index" not in st.session_state:
        st.session_state.active_dataset_index = 0
    for key, default in (
        ("sql_analysis", None),
        ("sql_candidate", None),
        ("sql_candidate_approved", False),
        ("sql_benchmark", None),
        ("sql_llm_advice", None),
        ("sql_dataset_label", None),
        ("pipeline_selected_indices", None),
        ("pipeline_results", None),
        ("approved_operation_ids", []),
        ("llm_provider", "ollama"),
        ("llm_model", "qwen3:4b"),
        ("llm_model_path", ""),
        ("llm_cache_dir", ""),
        ("llm_device", "auto"),
        ("llm_token", ""),
        ("llm_max_tokens", 512),
        ("llm_use_4bit", False),
        ("model_loaded", False),
        ("real_inference_success", False),
        ("heuristic_fallback_active", False),
        ("last_inference_error", None),
    ):
        if key not in st.session_state:
            st.session_state[key] = default
    _apply_llm_session_configuration()

    with st.sidebar:
        st.header("1. Upload data")
        uploaded_files = st.file_uploader("Upload CSV, Excel, JSON, SQLite or other supported files", accept_multiple_files=True)
        if uploaded_files:
            try:
                loaded = load_uploaded_files(uploaded_files)
                st.session_state.datasets = loaded
                st.session_state.active_dataset_index = 0
                if loaded:
                    record_event(
                        "dataset_loaded",
                        dataset_name=getattr(uploaded_files[0], "name", "uploaded_dataset"),
                        details={
                            "count": len(loaded),
                            "performance": [
                                dataset.get("performance", {})
                                for dataset in loaded
                                if dataset.get("source") != "error"
                            ],
                        },
                    )
                st.success(f"Loaded {len(loaded)} dataset(s).")
            except ValueError as exc:
                st.error(f"Import error: {exc}")

        if st.session_state.datasets:
            labels = [_dataset_label(dataset) for dataset in st.session_state.datasets]
            selected_label = st.selectbox("Active dataset", labels, index=min(st.session_state.active_dataset_index, len(labels) - 1))
            st.session_state.active_dataset_index = labels.index(selected_label)
            pipeline_options = [
                f"{index + 1}: {_dataset_label(dataset)}"
                for index, dataset in enumerate(st.session_state.datasets)
                if dataset.get("source") != "error"
            ]
            selected_options = st.session_state.get("pipeline_dataset_options")
            if selected_options != pipeline_options:
                st.session_state.pipeline_dataset_options = pipeline_options
                active_option = next(
                    (
                        option for option in pipeline_options
                        if option.startswith(f"{st.session_state.active_dataset_index + 1}:")
                    ),
                    pipeline_options[0] if pipeline_options else None,
                )
                st.session_state.pipeline_dataset_labels = [active_option] if active_option else []
            selected_pipeline_labels = st.multiselect(
                "Datasets for AI ETL Pipeline Generator",
                pipeline_options,
                key="pipeline_dataset_labels",
                help="One pipeline can apply supported deterministic operations independently to each selected dataset.",
            )
            st.session_state.pipeline_selected_indices = [
                int(label.split(":", 1)[0]) - 1
                for label in selected_pipeline_labels
            ]

        st.markdown("---")
        _render_ai_model_configuration()
        st.markdown("---")
        st.header("2. ETL instruction")
        instruction = st.text_area(
            "Describe the transformation you need",
            value=(st.session_state.get("instruction") if "instruction" in st.session_state else ""),
            height=180,
            placeholder="Example: Remove duplicate customer records, fill missing numeric values, standardize column names, and normalize dates.",
        )
        st.session_state.instruction = instruction

        st.header("3. Execution mode")
        st.session_state.execution_mode = st.selectbox("Choose execution strategy", ["python", "sql"], index=0 if st.session_state.execution_mode == "python" else 1)

        if st.button("Generate AI ETL pipeline"):
            if not st.session_state.datasets:
                st.warning("Please upload a dataset before generating a plan.")
            elif not instruction.strip():
                st.warning("Please describe the ETL transformation.")
            elif not st.session_state.get("pipeline_selected_indices"):
                st.warning("Select at least one valid dataset for pipeline generation.")
            else:
                try:
                    plan = generate_pipeline_specification(
                        st.session_state.datasets,
                        instruction,
                        selected_indices=st.session_state.pipeline_selected_indices,
                        execution_mode=st.session_state.execution_mode,
                        config=resolve_model_config(
                            provider=st.session_state.get("llm_provider", "ollama"),
                            model=st.session_state.get("llm_model", "qwen3:4b"),
                            model_path=st.session_state.get("llm_model_path") or None,
                            cache_dir=st.session_state.get("llm_cache_dir") or None,
                            token=st.session_state.get("llm_token") or None,
                            device=st.session_state.get("llm_device", "auto"),
                            max_tokens=int(st.session_state.get("llm_max_tokens", 512)),
                            use_4bit=bool(st.session_state.get("llm_use_4bit", False)),
                        ),
                    )
                except (ValueError, RuntimeError) as exc:
                    st.error(f"Could not generate an ETL pipeline: {exc}")
                else:
                    st.session_state.plan = plan
                    st.session_state.approved_plan = False
                    st.session_state.approved_operation_ids = []
                    st.session_state.last_run = None
                    st.session_state.pipeline_results = None
                    st.session_state.sql_analysis = None
                    st.session_state.sql_candidate = None
                    st.session_state.sql_candidate_approved = False
                    st.session_state.sql_benchmark = None
                    st.session_state.sql_llm_advice = None
                    selected_names = [
                        _dataset_label(st.session_state.datasets[index])
                        for index in st.session_state.pipeline_selected_indices
                    ]
                    record_event(
                        "pipeline_generated",
                        dataset_name=", ".join(selected_names),
                        instruction=instruction,
                        plan=plan,
                        details={
                            "execution_mode": st.session_state.execution_mode,
                            "provider": plan["provider"],
                            "model": plan["model"],
                            "source": plan["source"],
                            "real_inference_success": plan["real_inference_success"],
                            "fallback_active": plan["heuristic_fallback_active"],
                            "clarification_required": plan["clarification_required"],
                        },
                    )
                    st.success("AI ETL pipeline generated and validated.")

        if st.button("Save workflow snapshot"):
            if st.session_state.datasets:
                selected_dataset = st.session_state.datasets[st.session_state.active_dataset_index]
                save_snapshot(
                    dataset_name=selected_dataset.get("file_name", "dataset"),
                    dataset_label=_dataset_label(selected_dataset),
                    instruction=st.session_state.get("instruction", ""),
                    execution_mode=st.session_state.execution_mode,
                    plan=st.session_state.plan,
                    last_run=_serialize_run(st.session_state.last_run),
                    state={
                        "active_dataset_index": st.session_state.active_dataset_index,
                        "approved_plan": st.session_state.approved_plan,
                        "approved_operation_ids": st.session_state.approved_operation_ids,
                        "pipeline_selected_indices": st.session_state.pipeline_selected_indices,
                        "dataset_performance": selected_dataset.get("performance", {}),
                        "sql_analysis": st.session_state.sql_analysis,
                        "sql_candidate": st.session_state.sql_candidate,
                        "sql_candidate_approved": st.session_state.sql_candidate_approved,
                        "sql_benchmark": st.session_state.sql_benchmark,
                        "sql_llm_advice": st.session_state.sql_llm_advice,
                    },
                )
                st.success("Workflow snapshot saved.")

        if st.button("Reset workflow"):
            st.session_state.datasets = []
            st.session_state.plan = None
            st.session_state.approved_plan = False
            st.session_state.last_run = None
            st.session_state.pipeline_results = None
            st.session_state.approved_operation_ids = []
            st.session_state.pipeline_selected_indices = None
            st.session_state.pipeline_dataset_options = None
            st.session_state.instruction = ""
            st.session_state.active_dataset_index = 0
            st.session_state.sql_analysis = None
            st.session_state.sql_candidate = None
            st.session_state.sql_candidate_approved = False
            st.session_state.sql_benchmark = None
            st.session_state.sql_llm_advice = None
            st.session_state.sql_dataset_label = None
            st.success("Workflow reset. Upload a dataset to continue.")

        st.markdown("---")
        st.header("Workflow history")
        snapshots = list_snapshots(10)
        if snapshots:
            snapshot_labels = [f"{snapshot['created_at']} | {snapshot['dataset_label']} | {snapshot['execution_mode']}" for snapshot in snapshots]
            selected_snapshot_label = st.selectbox("Resume saved workflow", snapshot_labels, index=0)
            if st.button("Resume saved workflow"):
                snapshot = snapshots[snapshot_labels.index(selected_snapshot_label)]
                st.session_state.instruction = snapshot.get("instruction", "")
                st.session_state.plan = snapshot.get("plan") or None
                st.session_state.execution_mode = snapshot.get("execution_mode", "python")
                st.session_state.last_run = _restore_run(snapshot.get("last_run"))
                state = snapshot.get("state", {})
                st.session_state.approved_plan = False
                st.session_state.approved_operation_ids = []
                st.session_state.pipeline_results = None
                st.session_state.active_dataset_index = state.get("active_dataset_index", 0)
                st.session_state.pipeline_selected_indices = state.get("pipeline_selected_indices")
                st.session_state.sql_analysis = state.get("sql_analysis")
                st.session_state.sql_candidate = state.get("sql_candidate")
                st.session_state.sql_candidate_approved = bool(state.get("sql_candidate_approved"))
                st.session_state.sql_benchmark = state.get("sql_benchmark")
                st.session_state.sql_llm_advice = state.get("sql_llm_advice")
                st.success("Saved workflow restored.")
        else:
            st.info("No saved snapshots yet.")

    if not st.session_state.datasets:
        st.info("Upload a dataset to begin building your ETL pipeline.")
        return

    dataset = st.session_state.datasets[st.session_state.active_dataset_index]
    dataset_label = _dataset_label(dataset)
    if st.session_state.sql_dataset_label != dataset_label:
        st.session_state.sql_analysis = None
        st.session_state.sql_candidate = None
        st.session_state.sql_candidate_approved = False
        st.session_state.sql_benchmark = None
        st.session_state.sql_llm_advice = None
        st.session_state.sql_dataset_label = dataset_label
    if dataset.get("source") == "error":
        st.error(dataset.get("summary", {}).get("error", "Dataset import failed."))
        return

    _render_dataset_preview(dataset)

    if st.session_state.plan and st.session_state.plan.get("pipeline_schema_version") == 1:
        st.markdown("---")
        pipeline = st.session_state.plan
        _render_ai_pipeline(pipeline)

        st.markdown("#### Review or modify operation columns")
        for operation in pipeline.get("operations", []):
            dataset_index = int(operation["input_dataset"].removeprefix("dataset_")) - 1
            operation_frame = st.session_state.datasets[dataset_index]["dataframe"]
            if operation["operation"] in {"normalize_columns", "drop_duplicates", "validate_dataset"}:
                continue
            editor_key = f"pipeline_columns_{hash(pipeline.get('instruction', ''))}_{operation['id']}"
            with st.form(f"edit_{editor_key}"):
                selected_columns = st.multiselect(
                    f"Columns for {operation['name']}",
                    [str(column) for column in operation_frame.columns],
                    default=[column for column in operation["columns"] if column in operation_frame.columns],
                    key=editor_key,
                )
                update_columns = st.form_submit_button("Update operation columns")
            if update_columns:
                try:
                    updated_pipeline = update_pipeline_operation_columns(
                        pipeline,
                        operation["id"],
                        selected_columns,
                        st.session_state.datasets,
                        execution_mode=st.session_state.execution_mode,
                    )
                except ValueError as exc:
                    st.error(f"Could not update operation: {exc}")
                else:
                    st.session_state.plan = updated_pipeline
                    st.session_state.approved_plan = False
                    st.session_state.approved_operation_ids = []
                    record_event(
                        "operation_modified",
                        dataset_name=dataset.get("file_name", "dataset"),
                        instruction=st.session_state.get("instruction", ""),
                        plan=updated_pipeline,
                        details={
                            "operation_id": operation["id"],
                            "columns": selected_columns,
                            "approval_reset": True,
                        },
                    )
                    st.rerun()

        st.markdown("#### Operation-level approvals")
        for operation in pipeline.get("operations", []):
            operation_id = operation["id"]
            approval_form_key = f"approval_{hash(pipeline.get('instruction', ''))}_{operation_id}"
            with st.form(approval_form_key):
                is_approved = st.checkbox(
                    f"Approve {operation_id}: {operation['name']}",
                    value=operation_id in st.session_state.approved_operation_ids,
                    key=f"{approval_form_key}_checkbox",
                )
                save_approval = st.form_submit_button("Save operation approval")
            if save_approval:
                approved_ids = set(st.session_state.approved_operation_ids)
                if is_approved:
                    approved_ids.add(operation_id)
                    event_type = "pipeline_operation_approved"
                else:
                    approved_ids.discard(operation_id)
                    event_type = "pipeline_operation_rejected"
                st.session_state.approved_operation_ids = sorted(approved_ids)
                st.session_state.approved_plan = False
                record_event(
                    event_type,
                    dataset_name=dataset.get("file_name", "dataset"),
                    instruction=st.session_state.get("instruction", ""),
                    plan=pipeline,
                    details={"operation_id": operation_id},
                )
                st.rerun()

        all_operation_ids = {operation["id"] for operation in pipeline.get("operations", [])}
        approved_operation_ids = set(st.session_state.approved_operation_ids)
        approval_ready = bool(all_operation_ids) and (
            st.session_state.approved_plan
            or approved_operation_ids == all_operation_ids
        )

        approve_col, reject_col, execute_col = st.columns(3)
        with approve_col:
            if st.button(
                "Approve complete pipeline",
                disabled=not pipeline.get("operations") or pipeline.get("clarification_required"),
                key="approve_ai_pipeline",
            ):
                st.session_state.approved_plan = True
                st.session_state.approved_operation_ids = sorted(all_operation_ids)
                record_event(
                    "pipeline_approved",
                    dataset_name=", ".join(
                        _dataset_label(st.session_state.datasets[index])
                        for index in st.session_state.pipeline_selected_indices
                    ),
                    instruction=st.session_state.get("instruction", ""),
                    plan=pipeline,
                    details={"operation_ids": pipeline.get("execution_order", [])},
                )
                st.success("Pipeline approved. Execution is now enabled.")
        with reject_col:
            if st.button("Reject pipeline", key="reject_ai_pipeline"):
                record_event(
                    "pipeline_rejected",
                    dataset_name=", ".join(
                        _dataset_label(st.session_state.datasets[index])
                        for index in st.session_state.pipeline_selected_indices
                    ),
                    instruction=st.session_state.get("instruction", ""),
                    plan=pipeline,
                    details={"operation_ids": pipeline.get("execution_order", [])},
                )
                st.session_state.plan = None
                st.session_state.approved_plan = False
                st.session_state.approved_operation_ids = []
                st.session_state.pipeline_results = None
                st.warning("Pipeline rejected.")
        with execute_col:
            if st.button(
                "Execute approved pipeline",
                disabled=(
                    not approval_ready
                    or pipeline.get("clarification_required")
                    or not pipeline.get("operations")
                ),
                key="execute_ai_pipeline",
            ):
                resources_before = system_resource_snapshot()
                execution_started = time.perf_counter()
                try:
                    pipeline_results = execute_pipeline(
                        st.session_state.datasets,
                        pipeline,
                        approved=approval_ready,
                        approved_operation_ids=approved_operation_ids if not st.session_state.approved_plan else all_operation_ids,
                        execution_mode=st.session_state.execution_mode,
                    )
                except (SQLExecutionError, ValueError, PermissionError) as exc:
                    st.session_state.pipeline_results = None
                    record_event(
                        "pipeline_execution_failed",
                        dataset_name=", ".join(
                            _dataset_label(st.session_state.datasets[index])
                            for index in st.session_state.pipeline_selected_indices
                        ),
                        instruction=st.session_state.get("instruction", ""),
                        plan=pipeline,
                        details={
                            "error_category": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                    st.error(f"Pipeline execution failed: {exc}")
                else:
                    total_duration = time.perf_counter() - execution_started
                    resources_after = system_resource_snapshot()
                    st.session_state.pipeline_results = pipeline_results
                    active_result_id = dataset_id_for_index(st.session_state.active_dataset_index)
                    if active_result_id not in pipeline_results:
                        active_result_id = next(iter(pipeline_results))
                    active_result = pipeline_results[active_result_id]
                    operation_metrics = [
                        entry["performance"]
                        for result in pipeline_results.values()
                        for entry in result["audit"]
                        if entry.get("performance")
                    ]
                    st.session_state.last_run = {
                        **active_result,
                        "performance": {
                            "total_duration_seconds": total_duration,
                            "operations": operation_metrics,
                            "resources_before": resources_before,
                            "resources_after": resources_after,
                            "bottlenecks": identify_bottlenecks(operation_metrics),
                        },
                    }
                    result_summary = {
                        dataset_id: {
                            "dataset_label": result["dataset_label"],
                            "input_rows": len(result["original"]),
                            "output_rows": len(result["transformed"]),
                            "validation": result["validation"],
                            "operation_count": result["operation_count"],
                        }
                        for dataset_id, result in pipeline_results.items()
                    }
                    record_event(
                        "pipeline_execution_complete",
                        dataset_name=", ".join(
                            result["dataset_label"] for result in pipeline_results.values()
                        ),
                        instruction=st.session_state.get("instruction", ""),
                        plan=pipeline,
                        details={
                            "execution_mode": st.session_state.execution_mode,
                            "duration_seconds": total_duration,
                            "results": result_summary,
                        },
                    )
                    st.success("Approved pipeline execution finished; inspect per-dataset validation below.")

        if st.session_state.get("pipeline_results"):
            st.markdown("#### Execution results by dataset")
            for result_id, result in st.session_state.pipeline_results.items():
                with st.expander(result["dataset_label"], expanded=True):
                    st.write(
                        f"Rows: {len(result['original'])} → {len(result['transformed'])} | "
                        f"Operations: {result['operation_count']} | "
                        f"Validation: {result['validation']['status']}"
                    )
                    if result["validation"]["issues"]:
                        st.warning("; ".join(result["validation"]["issues"]))
                    else:
                        st.success("Final validation checks passed.")
                    st.dataframe(result["transformed"].head(20), use_container_width=True)
                    st.json(result["validation"])
                    st.dataframe(pd.DataFrame(result["audit"]), use_container_width=True, hide_index=True)

    if st.session_state.plan and st.session_state.plan.get("pipeline_schema_version") != 1:
        st.markdown("---")
        _render_plan(
            st.session_state.plan,
            dataset.get("dataframe", pd.DataFrame()),
            st.session_state.execution_mode,
        )
        _render_sql_optimization_studio(dataset, st.session_state.plan)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Approve plan"):
                st.session_state.approved_plan = True
                record_event("plan_approved", dataset_name=dataset.get("file_name", "dataset"), instruction=st.session_state.get("instruction", ""), plan=st.session_state.plan, details={"execution_mode": st.session_state.execution_mode})
                st.success("Plan approved. You can execute it next.")
        with col2:
            if st.button("Reject plan"):
                st.session_state.approved_plan = False
                st.session_state.plan = None
                record_event("plan_rejected", dataset_name=dataset.get("file_name", "dataset"), instruction=st.session_state.get("instruction", ""), details={"execution_mode": st.session_state.execution_mode})
                st.warning("Plan rejected. Update your instruction and regenerate.")
        with col3:
            if st.button("Execute approved plan"):
                if not st.session_state.approved_plan:
                    st.warning("Please approve the plan before execution.")
                else:
                    st.session_state.last_run = None
                    original_frame = dataset.get("dataframe", pd.DataFrame()).copy()
                    resources_before = system_resource_snapshot()
                    execution_started = time.perf_counter()
                    try:
                        transformed_frame, audit = execute_plan(
                            original_frame,
                            st.session_state.plan,
                            mode=st.session_state.execution_mode,
                        )
                    except SQLExecutionError as exc:
                        record_event(
                            "execution_failed",
                            dataset_name=dataset.get("file_name", "dataset"),
                            instruction=st.session_state.get("instruction", ""),
                            plan=st.session_state.plan,
                            details={
                                **exc.to_dict(),
                                "execution_mode": st.session_state.execution_mode,
                                "completed_steps": [
                                    entry
                                    for entry in exc.audit
                                    if entry.get("execution_status") == "succeeded"
                                ],
                                "failed_step": next(
                                    (
                                        entry
                                        for entry in exc.audit
                                        if entry.get("execution_status") == "failed"
                                    ),
                                    None,
                                ),
                            },
                        )
                        st.error(f"ETL execution failed: {exc}")
                        st.json(exc.to_dict())
                    else:
                        total_duration_seconds = time.perf_counter() - execution_started
                        resources_after = system_resource_snapshot()
                        validation = validate_dataframe(original_frame, transformed_frame)
                        operation_metrics = [
                            entry["performance"]
                            for entry in audit
                            if entry.get("performance")
                        ]
                        performance = {
                            "total_duration_seconds": total_duration_seconds,
                            "operations": operation_metrics,
                            "resources_before": resources_before,
                            "resources_after": resources_after,
                            "bottlenecks": identify_bottlenecks(operation_metrics),
                        }
                        st.session_state.last_run = {
                            "original": original_frame,
                            "transformed": transformed_frame,
                            "audit": audit,
                            "validation": validation,
                            "performance": performance,
                        }
                        record_event(
                            "execution_complete",
                            dataset_name=dataset.get("file_name", "dataset"),
                            instruction=st.session_state.get("instruction", ""),
                            plan=st.session_state.plan,
                            details={
                                "execution_mode": st.session_state.execution_mode,
                                "validation": validation,
                                "steps": len(audit),
                                "performance": performance,
                            },
                        )
                        save_snapshot(
                            dataset_name=dataset.get("file_name", "dataset"),
                            dataset_label=_dataset_label(dataset),
                            instruction=st.session_state.get("instruction", ""),
                            execution_mode=st.session_state.execution_mode,
                            plan=st.session_state.plan,
                            last_run=_serialize_run(st.session_state.last_run),
                            state={
                                "active_dataset_index": st.session_state.active_dataset_index,
                                "approved_plan": st.session_state.approved_plan,
                                "dataset_performance": dataset.get("performance", {}),
                                "sql_analysis": st.session_state.sql_analysis,
                                "sql_candidate": st.session_state.sql_candidate,
                                "sql_candidate_approved": st.session_state.sql_candidate_approved,
                                "sql_benchmark": st.session_state.sql_benchmark,
                                "sql_llm_advice": st.session_state.sql_llm_advice,
                            },
                        )
                        st.success("ETL execution completed.")

    if st.session_state.last_run:
        st.markdown("---")
        original = st.session_state.last_run["original"]
        transformed = st.session_state.last_run["transformed"]

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            csv_payload = transformed.to_csv(index=False)
            st.download_button("Download final CSV", csv_payload, file_name=f"{dataset.get('file_name', 'etl_result')}_final.csv", mime="text/csv")
        with export_col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                transformed.to_excel(writer, index=False, sheet_name="Final Data")
            buffer.seek(0)
            st.download_button("Download Excel", buffer.getvalue(), file_name=f"{dataset.get('file_name', 'etl_result')}_final.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        exec_tab, performance_tab, audit_tab, history_tab = st.tabs(
            ["Final dataset", "ETL Performance Intelligence", "Audit log", "Workflow history"]
        )

        with exec_tab:
            st.subheader("Transformed dataset preview")
            st.dataframe(transformed.head(20), use_container_width=True)
            _render_dashboard(original, transformed)

        with performance_tab:
            _render_performance(dataset, st.session_state.last_run)

        with audit_tab:
            for entry in st.session_state.last_run["audit"]:
                status = entry["status"]
                color = "green" if status == "passed" else "orange"
                st.markdown(f"<span style='color:{color};font-weight:bold'>{entry['step']}</span>", unsafe_allow_html=True)
                st.write(entry["description"])
                if "sql" in entry:
                    st.code(entry["sql"], language="sql")
                st.json(entry.get("details", {}))
                if entry.get("performance"):
                    st.json(entry["performance"])

        with history_tab:
            for event in list_history(10):
                with st.expander(f"{event['event_type']} — {event['created_at']}", expanded=False):
                    st.write(event.get("instruction") or "No instruction recorded")
                    st.json(event.get("details") or {})

        _render_pipeline_comparison_studio(
            original,
            transformed,
            st.session_state.last_run,
            plan=st.session_state.plan,
            instruction=st.session_state.get("instruction", ""),
        )



if __name__ == "__main__":
    main()
