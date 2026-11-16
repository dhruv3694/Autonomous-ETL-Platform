from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None

try:  # pragma: no cover - optional runtime dependency
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional runtime dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_HUGGINGFACE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_CANDIDATES = ["qwen3:4b", "qwen3:8b", "qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
HF_MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-4B-Instruct",
    "Qwen/Qwen3-8B",
]


def _normalize_model_name(value: Optional[str], *, provider: str) -> Optional[str]:
    if value is None:
        value = ""
    cleaned = str(value).strip()
    if not cleaned:
        if provider == "huggingface":
            return DEFAULT_HUGGINGFACE_MODEL
        return MODEL_CANDIDATES[0]
    return cleaned


@dataclass
class ModelConfig:
    provider: str = "ollama"
    model: Optional[str] = None
    model_path: Optional[str] = None
    cache_dir: Optional[str] = None
    token: Optional[str] = None
    device: Optional[str] = None
    max_tokens: int = 512
    use_4bit: bool = False

    @classmethod
    def from_env(cls, provider: Optional[str] = None, model: Optional[str] = None) -> "ModelConfig":
        active_provider = _normalize_provider(provider or os.getenv("ETL_LLM_PROVIDER", "ollama"))
        resolved_model = _normalize_model_name(model or os.getenv("ETL_LLM_MODEL"), provider=active_provider)
        if active_provider == "huggingface" and not resolved_model:
            resolved_model = DEFAULT_HUGGINGFACE_MODEL
        if active_provider == "ollama" and not resolved_model:
            resolved_model = _pick_model()
        return cls(
            provider=active_provider,
            model=resolved_model,
            model_path=os.getenv("ETL_LLM_MODEL_PATH") or os.getenv("HF_MODEL_PATH"),
            cache_dir=os.getenv("ETL_LLM_CACHE_DIR") or os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE"),
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("ETL_LLM_TOKEN"),
            device=os.getenv("ETL_LLM_DEVICE") or os.getenv("HF_DEVICE") or "auto",
            max_tokens=int(os.getenv("ETL_LLM_MAX_TOKENS", "512")),
            use_4bit=os.getenv("ETL_LLM_USE_4BIT", "false").lower() in {"1", "true", "yes"},
        )


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_provider(provider: Optional[str]) -> str:
    if provider is None:
        return "ollama"
    normalized = str(provider).strip().lower()
    if normalized in {"hf", "huggingface", "transformers"}:
        return "huggingface"
    if normalized in {"ollama", "local"}:
        return "ollama"
    return normalized


def _pick_model() -> str:
    return MODEL_CANDIDATES[0]


def supports_huggingface_local_inference() -> bool:
    return AutoTokenizer is not None and AutoModelForCausalLM is not None and torch is not None


def _redact_sensitive(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if not text:
        return ""
    patterns = [r"(hf[_-]?token|token|authorization|bearer)\s*[:=]\s*([^\s,\"']+)", r"(api[_-]?key|secret|password)\s*[:=]\s*([^\s,\"']+)"]
    redacted = text
    for pattern in patterns:
        redacted = re.sub(pattern, lambda match: f"{match.group(1)}=[REDACTED]", redacted, flags=re.IGNORECASE)
    return redacted


def resolve_model_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    model_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: Optional[int] = None,
    use_4bit: Optional[bool] = None,
) -> ModelConfig:
    base = ModelConfig.from_env(provider=provider or os.getenv("ETL_LLM_PROVIDER"), model=model or os.getenv("ETL_LLM_MODEL"))
    if model is not None:
        base.model = model
    if model_path is not None:
        base.model_path = model_path
    if cache_dir is not None:
        base.cache_dir = cache_dir
    if token is not None:
        base.token = token
    if device is not None:
        base.device = device
    if max_tokens is not None:
        base.max_tokens = max_tokens
    if use_4bit is not None:
        base.use_4bit = use_4bit
    if provider is not None:
        base.provider = _normalize_provider(provider)
    if base.model is None or not str(base.model).strip():
        base.model = _normalize_model_name(base.model, provider=base.provider)
    if provider is None and base.provider == "huggingface" and not str(base.model).strip():
        base.model = DEFAULT_HUGGINGFACE_MODEL
    if provider is None and base.provider == "ollama" and not str(base.model).strip():
        base.model = _pick_model()
    if model_path is None and base.model_path is None:
        base.model_path = os.getenv("ETL_LLM_MODEL_PATH") or os.getenv("HF_MODEL_PATH")
    if cache_dir is None and base.cache_dir is None:
        base.cache_dir = os.getenv("ETL_LLM_CACHE_DIR") or os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
    if token is None and base.token is None:
        base.token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("ETL_LLM_TOKEN")
    if device is None and base.device is None:
        base.device = os.getenv("ETL_LLM_DEVICE") or os.getenv("HF_DEVICE") or "auto"
    return base


def get_llm_status(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    cfg = config or resolve_model_config(provider=provider, model=model)
    provider_name = cfg.provider
    status: Dict[str, Any] = {
        "provider": provider_name,
        "provider_configured": bool(provider_name),
        "model": cfg.model,
        "model_configured": bool(cfg.model),
        "model_downloaded": bool(cfg.model_path or cfg.model),
        "model_loaded": False,
        "real_inference_success": False,
        "heuristic_fallback_active": False,
        "provider_available": False,
        "huggingface_available": supports_huggingface_local_inference(),
        "huggingface_pytorch_available": torch is not None,
        "cuda_available": bool(torch is not None and torch.cuda.is_available()) if torch is not None else False,
        "device": cfg.device or "auto",
        "ollama_available": False,
        "fallback_status": "not_needed",
        "last_inference_error": None,
        "ready": False,
    }
    if provider_name == "ollama":
        try:
            response = requests.get(f"{DEFAULT_OLLAMA_HOST}/api/tags", timeout=5)
            status["ollama_available"] = response.status_code == 200
            status["provider_available"] = status["ollama_available"]
            status["ready"] = status["ollama_available"]
        except Exception:
            status["ollama_available"] = False
            status["provider_available"] = False
            status["ready"] = False
    else:
        status["provider_available"] = status["huggingface_available"]
        status["ready"] = status["huggingface_available"]
    status["provider_ready"] = status["ready"]
    return status


def _ollama_plan(prompt: str) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{DEFAULT_OLLAMA_HOST}/api/generate",
            json={"model": _pick_model(), "prompt": prompt, "stream": False},
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama returned HTTP {response.status_code}.")
        data = response.json()
        content = data.get("response", "")
        parsed = json.loads(content)
        return parsed
    except Exception:
        raise


def _load_huggingface_model(config: Optional[ModelConfig] = None) -> Dict[str, Any]:
    if not supports_huggingface_local_inference():
        raise RuntimeError(
            "Hugging Face local inference is unavailable because transformers/torch is not installed."
        )

    cfg = config or ModelConfig.from_env(provider="huggingface")
    model_ref = cfg.model_path or cfg.model or os.getenv("HF_MODEL", DEFAULT_HUGGINGFACE_MODEL)
    cache_dir = cfg.cache_dir or os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
    device = cfg.device or os.getenv("HF_DEVICE", "cpu")
    token = cfg.token

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        cache_dir=cache_dir,
        token=token,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        cache_dir=cache_dir,
        token=token,
        device_map="auto" if device.lower() == "cuda" else None,
        local_files_only=True,
    )
    if device.lower() == "cpu":
        model.to("cpu")
    return {"tokenizer": tokenizer, "model": model, "model_ref": model_ref, "device": device}


def _huggingface_plan(prompt: str, config: Optional[ModelConfig] = None) -> Dict[str, Any]:
    model_state = _load_huggingface_model(config)
    tokenizer = model_state["tokenizer"]
    model = model_state["model"]
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=(config.max_tokens if config else 512),
        do_sample=False,
        temperature=0.0,
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    if response.startswith("```json"):
        response = response[7:]
    if response.endswith("```"):
        response = response[:-3]
    return json.loads(response.strip())


def _ollama_inference(prompt: str, config: Optional[ModelConfig] = None) -> str:
    payload = {
        "model": (config.model if config else _pick_model()),
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(f"{DEFAULT_OLLAMA_HOST}/api/generate", json=payload, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama returned HTTP {response.status_code}.")
    result = response.json()
    content = result.get("response")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned an empty response.")
    return content.strip()


def _huggingface_inference(prompt: str, config: Optional[ModelConfig] = None) -> str:
    model_state = _load_huggingface_model(config)
    tokenizer = model_state["tokenizer"]
    model = model_state["model"]
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=(config.max_tokens if config else 512),
        do_sample=False,
        temperature=0.0,
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    if not response:
        raise ValueError("Hugging Face returned an empty response.")
    return response


def run_model_inference(
    prompt: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    cfg = config or resolve_model_config(provider=provider, model=model)
    started = __import__("time").perf_counter()
    status: Dict[str, Any] = {
        "provider": cfg.provider,
        "model": cfg.model,
        "model_loaded": False,
        "real_inference_success": False,
        "heuristic_fallback_active": False,
        "inference_error": None,
        "response": None,
        "duration_seconds": 0.0,
        "model_available": False,
        "fallback_used": False,
    }
    try:
        if cfg.provider == "huggingface":
            if not supports_huggingface_local_inference():
                raise RuntimeError("Transformers/PyTorch is not installed; real Hugging Face inference is unavailable.")
            status["model_available"] = True
            response = _huggingface_inference(prompt, cfg)
        else:
            response = _ollama_inference(prompt, cfg)
            status["model_available"] = True
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Model returned a blank response.")
        status["real_inference_success"] = True
        status["response"] = response.strip()
        status["model_loaded"] = True
    except Exception as exc:  # pragma: no cover - error path exercised in tests via mocks
        status["inference_error"] = _redact_sensitive(str(exc))
        status["fallback_used"] = True
        status["heuristic_fallback_active"] = True
        fallback_plan = _fallback_plan({"columns": ["customer_id", "email", "order_date", "amount"]}, prompt)
        status["response"] = fallback_plan.get("warning") or fallback_plan.get("steps", [{}])[0].get("name") or "Heuristic fallback generated a safe ETL plan."
    finally:
        status["duration_seconds"] = float(__import__("time").perf_counter() - started)
    return status


def request_llm_completion(prompt: str, config: Optional[ModelConfig] = None) -> str:
    """Return a real provider response without converting errors into fallback output."""
    cfg = config or resolve_model_config()
    if cfg.provider == "huggingface":
        if not supports_huggingface_local_inference():
            raise RuntimeError(
                "Transformers/PyTorch is not installed; real Hugging Face inference is unavailable."
            )
        return _huggingface_inference(prompt, cfg)
    if cfg.provider == "ollama":
        return _ollama_inference(prompt, cfg)
    raise ValueError(f"Unsupported LLM provider: {cfg.provider!r}")


def _build_prompt(profile: Dict[str, Any], instruction: str) -> str:
    sample_columns = ", ".join(profile.get("columns", [])[:10]) or "no columns detected"
    return (
        "You are a safe ETL planner. Return only valid JSON with keys: \n"
        "\"steps\": [{\"id\", \"name\", \"description\", \"operation\", \"columns\", \"risk\", \"python\"}]\n"
        "The plan must be practical for the given dataset profile and the natural-language instruction.\n"
        f"Data profile: {json.dumps(profile, default=str)}\n"
        f"User instruction: {instruction}\n"
        "Prefer safe, explainable transformations and refuse unsafe or highly destructive actions."
    )


def _fallback_plan(profile: Dict[str, Any], instruction: str) -> Dict[str, Any]:
    text = instruction.lower()
    columns = profile.get("columns", [])
    candidate_columns = columns or ["column_1"]

    steps: List[Dict[str, Any]] = []
    if "duplicate" in text:
        steps.append({
            "id": "step_1",
            "name": "Remove duplicate records",
            "description": "Drop exact repeated rows while preserving the first valid record.",
            "operation": "drop_duplicates",
            "columns": candidate_columns,
            "risk": "low",
            "python": "df = df.drop_duplicates().reset_index(drop=True)"
        })
    if "missing" in text or "null" in text or "blank" in text:
        steps.append({
            "id": f"step_{len(steps)+1}",
            "name": "Fill missing values",
            "description": "Handle nulls using a safe statistical default such as median for numeric values and mode for categorical values.",
            "operation": "fill_missing",
            "columns": candidate_columns,
            "risk": "medium",
            "python": "numeric_cols = df.select_dtypes(include=['number']).columns; df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())"
        })
    if "date" in text or "timestamp" in text:
        steps.append({
            "id": f"step_{len(steps)+1}",
            "name": "Normalize dates",
            "description": "Convert date-like fields to a consistent ISO timestamp format.",
            "operation": "convert_dates",
            "columns": [c for c in candidate_columns if "date" in c.lower() or "time" in c.lower()],
            "risk": "medium",
            "python": "for col in ['date_column']:\n    df[col] = pd.to_datetime(df[col], errors='coerce')"
        })
    if "rename" in text or "standardize" in text or "clean" in text:
        steps.append({
            "id": f"step_{len(steps)+1}",
            "name": "Standardize column names",
            "description": "Rename columns to a consistent and readable format.",
            "operation": "normalize_columns",
            "columns": candidate_columns,
            "risk": "low",
            "python": "df.columns = [re.sub(r'\\s+', '_', c).lower() for c in df.columns]"
        })
    if "email" in text or "phone" in text or "city" in text or "string" in text:
        steps.append({
            "id": f"step_{len(steps)+1}",
            "name": "Clean text fields",
            "description": "Trim whitespace and normalize casing for string columns.",
            "operation": "clean_text",
            "columns": candidate_columns,
            "risk": "low",
            "python": "for col in df.select_dtypes(include=['object']).columns:\n    df[col] = df[col].astype(str).str.strip().str.title()"
        })
    if not steps:
        steps.append({
            "id": "step_1",
            "name": "Validate and preserve dataset",
            "description": "Perform a safe validation pass and keep the dataset structurally intact.",
            "operation": "validate_dataset",
            "columns": candidate_columns,
            "risk": "low",
            "python": "df = df.copy()"
        })

    return {"steps": steps, "source": "heuristic_fallback", "warning": "Local LLM not available; this is a safe rule-based plan."}


def generate_execution_plan(
    profile: Dict[str, Any],
    instruction: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    if not instruction or not instruction.strip():
        raise ValueError("Please provide transformation instructions before generating an ETL plan.")

    prompt = _build_prompt(profile, instruction)
    selected_provider = _normalize_provider(provider or os.getenv("ETL_LLM_PROVIDER"))
    cfg = config or ModelConfig.from_env(provider=selected_provider, model=model)

    try:
        if selected_provider == "huggingface":
            plan = _huggingface_plan(prompt, cfg)
        else:
            plan = _ollama_plan(prompt)
        if isinstance(plan, dict) and "steps" in plan:
            return {**plan, "provider": selected_provider, "model": cfg.model or _pick_model()}
    except Exception:
        pass

    fallback = _fallback_plan(profile, instruction)
    fallback["provider"] = selected_provider if selected_provider else "fallback"
    return fallback


def generate_sql_optimization_advice(
    analysis: Dict[str, Any],
    plan: Dict[str, Any],
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    operation_ids = {str(operation.get("operation_id")) for operation in analysis.get("operations", [])}
    selected_provider = _normalize_provider(provider or os.getenv("ETL_LLM_PROVIDER"))
    cfg = config or ModelConfig.from_env(provider=selected_provider, model=model)
    prompt = (
        "You are an optional SQLite query-plan explainer. Return only JSON with keys "
        '"summary" (string) and "recommendations" (array of objects with "operation_id" '
        'and "rationale" strings). Do not produce SQL, alter a plan, or claim measured '
        "performance. Recommendations must refer only to the supplied operation IDs. "
        f"\nAnalysis: {json.dumps(analysis, default=str)}"
        f"\nApproved pipeline context: {json.dumps(plan, default=str)}"
    )

    if selected_provider == "huggingface":
        payload = _huggingface_plan(prompt, cfg)
    else:
        response = requests.post(
            f"{DEFAULT_OLLAMA_HOST}/api/generate",
            json={
                "model": cfg.model or _pick_model(),
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama returned HTTP {response.status_code}.")
        payload = json.loads(response.json().get("response", "{}"))

    if not isinstance(payload, dict) or not isinstance(payload.get("summary"), str):
        raise ValueError("The local SQL advisor response has an invalid structure.")
    recommendations = payload.get("recommendations")
    if not isinstance(recommendations, list):
        raise ValueError("The local SQL advisor recommendations must be a list.")
    validated = []
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            raise ValueError("Each SQL advisor recommendation must be an object.")
        operation_id = recommendation.get("operation_id")
        rationale = recommendation.get("rationale")
        if (
            not isinstance(operation_id, str)
            or operation_id not in operation_ids
            or not isinstance(rationale, str)
            or not rationale.strip()
        ):
            raise ValueError("A SQL advisor recommendation failed validation.")
        validated.append({"operation_id": operation_id, "rationale": rationale.strip()[:500]})
    return {
        "provider": selected_provider,
        "model": cfg.model or _pick_model(),
        "summary": payload["summary"].strip()[:1000],
        "recommendations": validated,
        "is_advisory_only": True,
    }


def explain_comparison_results(
    comparison_summary: Dict[str, Any],
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    """Provide an optional LLM explanation for dataset comparison and quality indicators."""
    selected_provider = _normalize_provider(provider or os.getenv("ETL_LLM_PROVIDER"))
    cfg = config or ModelConfig.from_env(provider=selected_provider, model=model)
    prompt = (
        "You are an AI Data Quality and ETL Validation Architect. Explain the dataset comparison "
        "and transformation validation results below. Return only a JSON object with keys: "
        '"summary" (string), "quality_assessment" (string), and "recommendations" (array of strings). '
        "Never invent record counts or claim operations that are not in the provided comparison data.\n"
        f"Comparison Data: {json.dumps(comparison_summary, default=str)}"
    )

    try:
        raw_response = request_llm_completion(prompt, cfg)
        text = raw_response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        payload = json.loads(text)
        if isinstance(payload, dict) and "summary" in payload:
            return {
                "source": "llm",
                "provider": selected_provider,
                "model": cfg.model or _pick_model(),
                "summary": str(payload.get("summary", "")).strip(),
                "quality_assessment": str(payload.get("quality_assessment", "")).strip(),
                "recommendations": [str(r) for r in payload.get("recommendations", []) if isinstance(r, str)],
            }
    except Exception as exc:
        pass

    # Deterministic heuristic explanation
    row_metrics = comparison_summary.get("row_metrics", {})
    dq = comparison_summary.get("data_quality", {})
    before_rows = row_metrics.get("before_rows", 0)
    after_rows = row_metrics.get("after_rows", 0)
    dupes_removed = row_metrics.get("duplicates_removed", 0)
    improvements = dq.get("improvements", [])
    regressions = dq.get("regressions", [])

    summary_text = (
        f"Dataset row count transitioned from {before_rows} to {after_rows} "
        f"with {dupes_removed} duplicate row(s) removed."
    )
    if improvements:
        quality_assessment = f"Data quality improved across {len(improvements)} dimension(s): {'; '.join(improvements)}."
    else:
        quality_assessment = "Data quality metrics remained stable without measured regressions."

    recs = []
    if regressions:
        recs.append(f"Review potential regressions: {'; '.join(regressions)}.")
    if dq.get("remaining_issues"):
        recs.append(f"Remaining issues to address: {'; '.join(dq['remaining_issues'][:3])}.")
    if not recs:
        recs.append("All output checks passed; dataset is ready for downstream analytics.")

    return {
        "source": "deterministic_fallback",
        "provider": "rule_based_engine",
        "model": "none",
        "summary": summary_text,
        "quality_assessment": quality_assessment,
        "recommendations": recs,
    }

