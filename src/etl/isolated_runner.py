from __future__ import annotations

import os
import sys
import tempfile
import time
from typing import Any, Dict, Optional

import pandas as pd


_CHILD_RUNNER_SCRIPT = """
import sys
import time
import pandas as pd
import numpy as np
import re

def main():
    if len(sys.argv) < 4:
        sys.exit(2)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    code_path = sys.argv[3]

    try:
        df = pd.read_parquet(in_path)
    except Exception as exc:
        sys.stderr.write(f"INPUT_DESERIALIZATION_ERROR: {exc}\\n")
        sys.exit(3)

    try:
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as exc:
        sys.stderr.write(f"CODE_READ_ERROR: {exc}\\n")
        sys.exit(4)

    safe_globals = {
        "__builtins__": {
            "range": range,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "None": None,
            "True": True,
            "False": False,
        },
        "pd": pd,
        "np": np,
        "re": re,
    }
    safe_locals = {"df": df}

    try:
        compiled = compile(code, "<isolated_transform>", "exec")
        exec(compiled, safe_globals, safe_locals)
    except Exception as exc:
        sys.stderr.write(f"RUNTIME_ERROR: {type(exc).__name__}: {exc}\\n")
        sys.exit(5)

    res_df = safe_locals.get("df")
    if not isinstance(res_df, pd.DataFrame):
        sys.stderr.write("INVALID_OUTPUT_TYPE: Result 'df' is not a pandas DataFrame.\\n")
        sys.exit(6)

    try:
        res_df.to_parquet(out_path, index=False)
    except Exception as exc:
        sys.stderr.write(f"OUTPUT_SERIALIZATION_ERROR: {exc}\\n")
        sys.exit(7)

    sys.exit(0)

if __name__ == "__main__":
    main()
"""


def execute_python_isolated_process(
    code: str,
    dataframe: pd.DataFrame,
    *,
    timeout_seconds: float = 10.0,
    max_memory_mb: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Python code in a dedicated child process with explicit Parquet serialization."""
    import subprocess

    with tempfile.TemporaryDirectory(prefix="etl_iso_") as temp_dir:
        in_path = os.path.join(temp_dir, "input.parquet")
        out_path = os.path.join(temp_dir, "output.parquet")
        code_path = os.path.join(temp_dir, "transform.py")
        runner_path = os.path.join(temp_dir, "runner.py")

        try:
            dataframe.to_parquet(in_path, index=False)
        except Exception as exc:
            return {
                "outcome": "EXECUTION_FAILED",
                "error": f"Failed to serialize input dataset to Parquet: {exc}",
                "dataframe": None,
                "isolated": True,
            }

        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(_CHILD_RUNNER_SCRIPT)

        # Build clean environment with NO application secrets inherited
        # Pass site-packages / stdlib via PYTHONPATH so standard modules (pandas, numpy, pyarrow) load
        clean_pythonpath = os.pathsep.join(p for p in sys.path if p and p != ".")
        clean_env = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
            "PATH": os.environ.get("PATH", ""),
            "PATHEXT": os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD"),
            "TEMP": temp_dir,
            "TMP": temp_dir,
            "PYTHONPATH": clean_pythonpath,
            "PYTHONUNBUFFERED": "1",
        }

        cmd = [sys.executable, runner_path, in_path, out_path, code_path]
        start_time = time.perf_counter()

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=temp_dir,
                env=clean_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            elapsed = time.perf_counter() - start_time

            if proc.returncode != 0:
                err_clean = (stderr or stdout).strip().split("\n")[-1]
                return {
                    "outcome": "EXECUTION_FAILED",
                    "error": err_clean or f"Child process exited with code {proc.returncode}.",
                    "dataframe": None,
                    "isolated": True,
                    "duration_seconds": round(elapsed, 4),
                }

            if not os.path.exists(out_path):
                return {
                    "outcome": "EXECUTION_FAILED",
                    "error": "Child process completed but output Parquet file was not produced.",
                    "dataframe": None,
                    "isolated": True,
                    "duration_seconds": round(elapsed, 4),
                }

            out_df = pd.read_parquet(out_path)
            return {
                "outcome": "VALIDATED",
                "error": None,
                "dataframe": out_df,
                "isolated": True,
                "duration_seconds": round(elapsed, 4),
            }

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return {
                "outcome": "EXECUTION_TIMEOUT",
                "error": f"Transformation exceeded process timeout limit of {timeout_seconds}s and was terminated.",
                "dataframe": None,
                "isolated": True,
            }
        except Exception as exc:
            return {
                "outcome": "EXECUTION_FAILED",
                "error": f"Failed to execute isolated child process: {exc}",
                "dataframe": None,
                "isolated": True,
            }
