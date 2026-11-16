from __future__ import annotations

import os
from unittest.mock import patch

from app import _apply_llm_session_configuration
from src.etl.llm_service import (
    get_llm_status,
    resolve_model_config,
    run_model_inference,
)


def test_provider_selection_uses_huggingface_config() -> None:
    config = resolve_model_config(provider="huggingface", model="Qwen/Qwen2.5-3B-Instruct")
    assert config.provider == "huggingface"
    assert config.model == "Qwen/Qwen2.5-3B-Instruct"


def test_provider_switching_respects_ollama_default() -> None:
    config = resolve_model_config(provider="ollama", model="qwen3:4b")
    assert config.provider == "ollama"
    assert config.model == "qwen3:4b"


def test_status_reporting_includes_provider_and_runtime_details() -> None:
    with patch("src.etl.llm_service.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        status = get_llm_status(provider="ollama", model="qwen3:4b")
    assert status["provider"] == "ollama"
    assert status["model"] == "qwen3:4b"
    assert status["provider_configured"] is True
    assert "provider_available" in status
    assert "cuda_available" in status
    assert "last_inference_error" in status


def test_missing_transformers_and_pytorch_are_flagged() -> None:
    with patch("src.etl.llm_service.torch", None), patch("src.etl.llm_service.AutoTokenizer", None), patch("src.etl.llm_service.AutoModelForCausalLM", None):
        status = get_llm_status(provider="huggingface", model="Qwen/Qwen2.5-3B-Instruct")
    assert status["huggingface_available"] is False
    assert status["huggingface_pytorch_available"] is False


def test_missing_ollama_is_reported_as_unavailable() -> None:
    with patch("src.etl.llm_service.requests.get", side_effect=RuntimeError("offline")):
        status = get_llm_status(provider="ollama", model="qwen3:4b")
    assert status["ollama_available"] is False
    assert status["provider_available"] is False


def test_invalid_model_configuration_uses_safe_default() -> None:
    config = resolve_model_config(provider="huggingface", model="")
    assert config.provider == "huggingface"
    assert config.model in {"Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"}


def test_inference_success_returns_real_output() -> None:
    with patch("src.etl.llm_service._ollama_inference", return_value="Completed validation checks."):
        result = run_model_inference("sample prompt", provider="ollama", model="qwen3:4b")
    assert result["real_inference_success"] is True
    assert result["response"] == "Completed validation checks."
    assert result["heuristic_fallback_active"] is False


def test_inference_failure_uses_fallback_and_reports_reason() -> None:
    with patch("src.etl.llm_service._ollama_inference", side_effect=RuntimeError("model not found")):
        result = run_model_inference("sample prompt", provider="ollama", model="qwen3:4b")
    assert result["real_inference_success"] is False
    assert result["heuristic_fallback_active"] is True
    assert "model not found" in result["inference_error"]


def test_streamlit_configuration_integration_updates_environment() -> None:
    os.environ.pop("ETL_LLM_PROVIDER", None)
    os.environ.pop("ETL_LLM_MODEL", None)
    os.environ.pop("HF_TOKEN", None)
    from streamlit import session_state

    session_state.clear()
    session_state.llm_provider = "huggingface"
    session_state.llm_model = "Qwen/Qwen2.5-3B-Instruct"
    session_state.llm_model_path = ""
    session_state.llm_cache_dir = "/tmp/cache"
    session_state.llm_device = "cpu"
    session_state.llm_token = "supersecret"
    session_state.llm_max_tokens = 256
    session_state.llm_use_4bit = True
    config = _apply_llm_session_configuration()
    assert config["provider"] == "huggingface"
    assert os.environ["ETL_LLM_PROVIDER"] == "huggingface"
    assert os.environ["ETL_LLM_MODEL"] == "Qwen/Qwen2.5-3B-Instruct"
    assert os.environ["HF_TOKEN"] == "supersecret"
    assert os.environ["ETL_LLM_DEVICE"] == "cpu"
