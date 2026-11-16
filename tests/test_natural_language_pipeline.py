from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from src.etl.llm_service import ModelConfig
from src.etl.pipeline_generator import (
    build_dataset_context,
    execute_pipeline,
    generate_pipeline_specification,
    validate_pipeline,
)


def _dataset() -> dict:
    frame = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 3],
            "email": [" A@EXAMPLE.COM ", " A@EXAMPLE.COM ", None, "b@example.com"],
            "order_date": ["2025/01/02", "2025/01/02", "02-03-2025", "2025-03-04"],
            "amount": [10.0, 10.0, None, 30.0],
        }
    )
    return {
        "file_name": "orders.csv",
        "source": "file",
        "dataframe": frame,
    }


def _mock_model_response() -> str:
    return json.dumps(
        {
            "operations": [
                {
                    "id": "dedupe",
                    "name": "Remove exact duplicate rows",
                    "operation": "drop_duplicates",
                    "description": "Remove exact duplicate records.",
                    "input_dataset": "dataset_1",
                    "columns": [],
                    "parameters": {},
                    "dependencies": [],
                    "risk": "low",
                },
                {
                    "id": "fill-amount",
                    "name": "Fill missing amount",
                    "operation": "fill_missing",
                    "description": "Fill missing amounts using the deterministic numeric median.",
                    "input_dataset": "dataset_1",
                    "columns": ["amount"],
                    "parameters": {},
                    "dependencies": ["dedupe"],
                    "risk": "medium",
                },
                {
                    "id": "parse-date",
                    "name": "Parse order date",
                    "operation": "convert_dates",
                    "description": "Parse the order_date values as datetimes.",
                    "input_dataset": "dataset_1",
                    "columns": ["order_date"],
                    "parameters": {},
                    "dependencies": ["fill-amount"],
                    "risk": "medium",
                },
            ],
            "ambiguities": [],
        }
    )


def test_dataset_context_uses_bounded_real_profile_values() -> None:
    context = build_dataset_context([_dataset()])
    assert context[0]["row_count"] == 4
    assert context[0]["duplicate_rows"] == 1
    assert context[0]["dataset_id"] == "dataset_1"
    assert [column["name"] for column in context[0]["columns"]] == [
        "customer_id",
        "email",
        "order_date",
        "amount",
    ]
    assert len(context[0]["columns"][1]["sample_values"]) <= 3
    assert context[0]["columns"][3]["numeric_statistics"]["median"] == 10.0


def test_mocked_model_plan_executes_only_approved_allowlisted_operations() -> None:
    dataset = _dataset()
    plan = generate_pipeline_specification(
        [dataset],
        "Remove duplicate records, fill missing amounts, and standardize order dates.",
        config=ModelConfig(provider="ollama", model="mock-model"),
        inference_fn=lambda prompt, config: _mock_model_response(),
    )

    assert plan["source"] == "llm"
    assert plan["real_inference_success"] is True
    assert plan["execution_order"] == ["dedupe", "fill-amount", "parse-date"]
    assert plan["edges"] == [
        {"from": "dedupe", "to": "fill-amount"},
        {"from": "fill-amount", "to": "parse-date"},
    ]
    with pytest.raises(PermissionError):
        execute_pipeline([dataset], plan, approved=False)

    results = execute_pipeline([dataset], plan, approved=True)
    result = results["dataset_1"]
    transformed = result["transformed"]
    assert len(transformed) == 3
    assert transformed["amount"].isna().sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(transformed["order_date"])
    assert result["validation"]["status"] == "passed"
    assert dataset["dataframe"]["amount"].isna().sum() == 1


def test_schema_validation_rejects_missing_columns_and_unsupported_operations() -> None:
    context = build_dataset_context([_dataset()])
    with pytest.raises(ValueError, match="unknown column"):
        validate_pipeline(
            {
                "operations": [
                    {
                        "id": "bad-column",
                        "operation": "clean_text",
                        "input_dataset": "dataset_1",
                        "columns": ["invented"],
                        "parameters": {},
                        "dependencies": [],
                    }
                ]
            },
            context,
        )
    with pytest.raises(ValueError, match="Unsupported operation"):
        validate_pipeline(
            {
                "operations": [
                    {
                        "id": "raw-code",
                        "operation": "execute_python",
                        "input_dataset": "dataset_1",
                        "columns": [],
                        "parameters": {},
                        "dependencies": [],
                    }
                ]
            },
            context,
        )


def test_dependency_cycle_is_rejected() -> None:
    context = build_dataset_context([_dataset()])
    operations = [
        {
            "id": operation_id,
            "operation": "clean_text",
            "input_dataset": "dataset_1",
            "columns": ["email"],
            "parameters": {},
            "dependencies": [dependency],
        }
        for operation_id, dependency in (("first", "second"), ("second", "first"))
    ]
    with pytest.raises(ValueError, match="cycle"):
        validate_pipeline({"operations": operations}, context)


def test_invalid_or_unavailable_model_uses_explicit_heuristic_fallback() -> None:
    with patch(
        "src.etl.pipeline_generator.request_llm_completion",
        side_effect=RuntimeError("model unavailable"),
    ):
        plan = generate_pipeline_specification(
            [_dataset()],
            "Remove duplicate records and handle missing values.",
            config=ModelConfig(provider="ollama", model="mock-model"),
        )
    assert plan["source"] == "heuristic_fallback"
    assert plan["real_inference_success"] is False
    assert plan["heuristic_fallback_active"] is True
    assert "model unavailable" in plan["inference_error"]
    assert {operation["operation"] for operation in plan["operations"]} == {
        "drop_duplicates",
        "fill_missing",
    }


def test_ambiguous_categorical_correction_blocks_execution() -> None:
    plan = generate_pipeline_specification(
        [_dataset()],
        "Correct inconsistent categorical values.",
        config=ModelConfig(provider="ollama", model="mock-model"),
        inference_fn=lambda prompt, config: json.dumps({"operations": [], "ambiguities": []}),
    )
    assert plan["clarification_required"] is True
    with pytest.raises(ValueError, match="clarification"):
        execute_pipeline([_dataset()], plan, approved=True)


def test_sql_pipeline_is_generated_and_executes_with_the_existing_sqlite_engine() -> None:
    dataset = _dataset()
    response = {
        "operations": [
            {
                "id": "dedupe",
                "name": "Remove duplicates",
                "operation": "drop_duplicates",
                "description": "Remove duplicate rows.",
                "input_dataset": "dataset_1",
                "columns": [],
                "parameters": {},
                "dependencies": [],
                "risk": "low",
            }
        ],
        "ambiguities": [],
    }
    plan = generate_pipeline_specification(
        [dataset],
        "Remove duplicate records.",
        execution_mode="sql",
        config=ModelConfig(provider="ollama", model="mock-model"),
        inference_fn=lambda prompt, config: json.dumps(response),
    )
    assert "CREATE TABLE" in plan["operations"][0]["sql"]
    results = execute_pipeline([dataset], plan, approved=True, execution_mode="sql")
    assert len(results["dataset_1"]["transformed"]) == 3
    assert results["dataset_1"]["validation"]["status"] == "passed"
