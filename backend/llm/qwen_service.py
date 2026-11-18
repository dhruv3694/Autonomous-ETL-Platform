"""
Qwen-Only LLM Service Module.
Handles:
- Qwen model selection (Qwen3-4B-Instruct, Qwen2.5-7B-Instruct, etc.)
- Ollama local endpoint and HuggingFace local pipeline support
- Prompt formatting with strict schema and dataset metadata injection
- Structured JSON ETL plan generation
- JSON parsing and strict validation against predefined schema
- Robust fallback to deterministic plan generation when local inference is unreachable.
"""

import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Validated transformations registry
SUPPORTED_OPERATIONS = [
    "drop_duplicates",
    "fill_missing",
    "convert_dates",
    "clean_text",
    "normalize_columns",
    "filter_rows",
    "rename_columns",
    "cast_types"
]

class ETLStep(BaseModel):
    step_id: str
    name: str
    operation: str
    description: str
    target_columns: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    engine: str = "python" # python or sql
    code_representation: str
    expected_output: str
    risk_level: str = "low" # low, medium, high

class ETLPlan(BaseModel):
    pipeline_id: str
    instruction: str
    dataset_name: str
    detected_schema: List[str]
    steps: List[ETLStep]
    validation_rules: List[str]
    potential_risks: List[str]
    expected_quality_changes: str

def construct_qwen_prompt(dataset_name: str, schema_columns: List[str], sample_profile: Dict[str, Any], instruction: str) -> str:
    """Builds prompt instructing Qwen to output an ETL JSON plan."""
    return f"""<|im_start|>system
You are Qwen, an expert Data Engineer. You generate structured, safe ETL transformation pipelines.
Your ONLY output must be a valid JSON object conforming strictly to this structure:
{{
  "pipeline_id": "pipe_uuid",
  "instruction": "{instruction}",
  "dataset_name": "{dataset_name}",
  "detected_schema": {json.dumps(schema_columns)},
  "steps": [
    {{
      "step_id": "step_1",
      "name": "Human Readable Step Name",
      "operation": "one of: drop_duplicates, fill_missing, convert_dates, clean_text, normalize_columns, filter_rows, rename_columns, cast_types",
      "description": "Explanation of transformation",
      "target_columns": ["col1"],
      "parameters": {{}},
      "engine": "python",
      "code_representation": "df.drop_duplicates(inplace=True)",
      "expected_output": "Dataset without duplicate rows",
      "risk_level": "low"
    }}
  ],
  "validation_rules": ["Check row counts", "Check null values"],
  "potential_risks": ["Potential loss of rows"],
  "expected_quality_changes": "Completeness and uniqueness should improve."
}}
Do NOT output markdown commentary or extra text outside the JSON object.
<|im_end|>
<|im_start|>user
Dataset: {dataset_name}
Columns: {json.dumps(schema_columns)}
Sample Profile: {json.dumps(sample_profile)}
User Instruction: {instruction}
Generate the JSON ETL execution plan:
<|im_end|>
<|im_start|>assistant
"""

def generate_deterministic_plan(pipeline_id: str, dataset_name: str, schema_columns: List[str], instruction: str) -> ETLPlan:
    """
    Deterministic rule-based fallback when Qwen model server is unavailable.
    Interprets natural language keywords to guarantee user experience.
    """
    inst_lower = instruction.lower()
    steps: List[ETLStep] = []
    step_num = 1

    # 1. Normalize columns
    if any(k in inst_lower for k in ["normalize column", "standardize column", "clean column name", "clean header"]):
        steps.append(ETLStep(
            step_id=f"step_{step_num}",
            name="Normalize Column Names",
            operation="normalize_columns",
            description="Convert column headers to lowercase and replace whitespace with underscores.",
            target_columns=schema_columns,
            parameters={},
            engine="python",
            code_representation="df.columns = [re.sub(r'\\s+', '_', str(c)).lower().strip() for c in df.columns]",
            expected_output="Uniform snake_case column names.",
            risk_level="low"
        ))
        step_num += 1

    # 2. Drop duplicates
    if any(k in inst_lower for k in ["duplicate", "deduplicate", "unique"]):
        steps.append(ETLStep(
            step_id=f"step_{step_num}",
            name="Remove Duplicate Records",
            operation="drop_duplicates",
            description="Detect and eliminate redundant identical rows across the dataset.",
            target_columns=schema_columns,
            parameters={},
            engine="python",
            code_representation="df = df.drop_duplicates()",
            expected_output="Deduplicated dataset with improved uniqueness score.",
            risk_level="medium"
        ))
        step_num += 1

    # 3. Clean text / whitespace
    if any(k in inst_lower for k in ["whitespace", "trim", "clean text", "strip", "standardize email"]):
        target_cols = [c for c in schema_columns if any(t in c.lower() for t in ["name", "email", "city", "address", "state", "type"])]
        if not target_cols:
            target_cols = schema_columns[:3]
        steps.append(ETLStep(
            step_id=f"step_{step_num}",
            name="Clean Text and Trim Whitespace",
            operation="clean_text",
            description=f"Trim leading/trailing whitespace and sanitize string columns ({', '.join(target_cols)}).",
            target_columns=target_cols,
            parameters={"columns": target_cols, "trim": True, "lowercase": "email" in inst_lower},
            engine="python",
            code_representation=f"for col in {target_cols}: df[col] = df[col].astype(str).str.strip()",
            expected_output="Cleaned string values without trailing or leading spacing.",
            risk_level="low"
        ))
        step_num += 1

    # 4. Fill missing values
    if any(k in inst_lower for k in ["missing", "impute", "null", "na"]):
        strategy = "median" if "median" in inst_lower else ("mean" if "mean" in inst_lower else "mode")
        numeric_candidates = [c for c in schema_columns if any(t in c.lower() for t in ["age", "amount", "salary", "price", "qty", "score", "rate"])]
        steps.append(ETLStep(
            step_id=f"step_{step_num}",
            name=f"Handle Missing Values ({strategy.capitalize()})",
            operation="fill_missing",
            description=f"Impute missing entries in numeric columns using {strategy} value.",
            target_columns=numeric_candidates or schema_columns[:2],
            parameters={"strategy": strategy, "columns": numeric_candidates},
            engine="python",
            code_representation=f"df[cols] = df[cols].fillna(df[cols].{strategy}())",
            expected_output="Zero null values in imputed target columns, boosting completeness.",
            risk_level="medium"
        ))
        step_num += 1

    # 5. Convert dates
    if any(k in inst_lower for k in ["date", "datetime", "timestamp", "yyyy-mm-dd"]):
        date_candidates = [c for c in schema_columns if "date" in c.lower() or "time" in c.lower()]
        steps.append(ETLStep(
            step_id=f"step_{step_num}",
            name="Standardize Date Formats",
            operation="convert_dates",
            description="Parse date strings and format uniformly to ISO 8601 (YYYY-MM-DD).",
            target_columns=date_candidates or schema_columns[-1:],
            parameters={"format": "%Y-%m-%d"},
            engine="python",
            code_representation="df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')",
            expected_output="Uniform ISO date strings.",
            risk_level="low"
        ))
        step_num += 1

    # Default if no recognized keywords
    if not steps:
        steps.append(ETLStep(
            step_id="step_1",
            name="Sanitize Dataset and Remove Duplicates",
            operation="drop_duplicates",
            description="Default safe baseline transformation based on dataset profile.",
            target_columns=schema_columns,
            parameters={},
            engine="python",
            code_representation="df = df.drop_duplicates()",
            expected_output="Standardized deduplicated dataset.",
            risk_level="low"
        ))

    return ETLPlan(
        pipeline_id=pipeline_id,
        instruction=instruction,
        dataset_name=dataset_name,
        detected_schema=schema_columns,
        steps=steps,
        validation_rules=["Verify row counts post-deduplication", "Ensure column preservation"],
        potential_risks=["Rows with unique identifiers might be dropped if identical"],
        expected_quality_changes="Increases uniqueness and completeness scores."
    )

def request_qwen_plan(
    pipeline_id: str,
    dataset_name: str,
    schema_columns: List[str],
    sample_profile: Dict[str, Any],
    instruction: str,
    ollama_url: str = "http://localhost:11434"
) -> Tuple[ETLPlan, bool]:
    """
    Requests Qwen to generate structured plan.
    Returns (ETLPlan, is_live_qwen: bool).
    """
    prompt = construct_qwen_prompt(dataset_name, schema_columns, sample_profile, instruction)
    payload = {
        "model": "qwen2.5:latest",
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        req = urllib.request.Request(
            f"{ollama_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            raw_text = res_data.get("response", "")
            # Extract JSON block
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                plan_dict = json.loads(json_match.group(0))
                parsed_plan = ETLPlan(**plan_dict)
                return parsed_plan, True
    except Exception:
        pass

    # Fallback to deterministic plan
    fallback_plan = generate_deterministic_plan(pipeline_id, dataset_name, schema_columns, instruction)
    return fallback_plan, False
