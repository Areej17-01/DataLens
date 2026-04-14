import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pydantic import PrivateAttr
import google.generativeai as genai

from agentpro.tools.base_tool import Tool

from ..config import TEMP_ROOT


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.ndarray, pd.Series, list, tuple)):
        return [
            _serialize_value(v) for v in (value.tolist() if hasattr(value, "tolist") else value)
        ]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


class CsvAnalystTool(Tool):
    name: str = "CSV Data Analyst"
    action_type: str = "csv_analyst"
    input_format: str = "A dict containing {'session_id': '<uuid>', 'query': '<question>'}"
    description: str = (
        "Loads uploaded CSV data and metadata, generates pandas code, executes it, "
        "and returns the code, reasoning, and output dictionary."
    )

    _llm: Any = PrivateAttr()
    _temp_root: Path = PrivateAttr()

    def __init__(self, llm: Any = None, **data):
        super().__init__(**data)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self._llm = genai.GenerativeModel("gemini-2.0-flash")
        self._temp_root = Path(TEMP_ROOT)

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self._llm.generate_content(full_prompt)
        return response.text or ""

    def run(self, input_data: Any) -> Dict[str, Any]:
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                return self._error("Expected input format {'session_id': '<uuid>', 'query': '<question>'}.")

        if not isinstance(input_data, dict):
            return self._error("Expected input format {'session_id': '<uuid>', 'query': '<question>'}")

        session_id = input_data.get("session_id")
        query = input_data.get("query")

        if not session_id or not query:
            return self._error("Missing session_id or query.")

        csv_folder = self._temp_root / session_id / "csv"
        csv_files = sorted(csv_folder.glob("*.csv")) if csv_folder.exists() else []
        if not csv_files:
            return self._error("CSV file not found for this session.")

        csv_path = csv_files[0]
        metadata_path = self._temp_root / session_id / "json" / f"{csv_path.stem}.json"

        if not metadata_path.exists():
            return self._error("CSV metadata file not found for this session.")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        df = pd.read_csv(csv_path)

        prompt = self._build_code_prompt(query, metadata)
        generated_code = self._generate_python_code(prompt)
        sanitized_code = self._sanitize_code(generated_code)

        execution = self._execute_code(sanitized_code, df, metadata)

        insight = execution.get("insight", "No insight could be generated.")
        reasoning = execution.get("reasoning", "No reasoning available.")
        python_output = execution.get("output", "")
        context_used = self._describe_context(metadata)

        return {
            "insight": insight,
            "context_used": context_used,
            "code": sanitized_code,
            "reasoning": reasoning,
            "python_output": python_output,
            "llm_response": insight,
        }

    def _error(self, msg: str) -> Dict[str, Any]:
        return {
            "insight": f"Error: {msg}",
            "context_used": "",
            "code": "",
            "reasoning": "",
            "python_output": "",
            "llm_response": "",
        }

    def _build_code_prompt(self, query: str, metadata: Dict[str, Any]) -> str:
        sample_columns = [col["name"] for col in metadata.get("columns", [])]
        col_details = "\n".join(
            f"  - {col['name']} (dtype: {col.get('dtype', 'unknown')})"
            for col in metadata.get("columns", [])
        )
        sample_preview = json.dumps(metadata.get("sample_rows", []), indent=2)

        prompt = (
            "You are an expert Python data analyst. Your job is to write clean, thorough pandas code "
            "that fully answers the user's question with well-labeled, human-readable output.\n\n"

            "STRICT RULES:\n"
            "1. Do NOT include any import statements — pd, np, df, and metadata are already available.\n"
            "2. Do NOT generate plots, charts, or any visualization code.\n"
            "3. Do NOT use a DataFrame or Series as a boolean (e.g. `if df` or `if df[col]`). "
            "   Use `.empty`, `.any()`, or `.all()` instead.\n"
            "4. Do NOT return raw pandas objects. Convert everything to plain Python: "
            "   use `.to_dict()`, `.tolist()`, `str()`, or format into a readable string.\n"
            "5. `answer` MUST be a human-readable string — NOT a list of raw numbers or unlabeled values. "
            "   Format it clearly so anyone reading it understands what the numbers mean.\n"
            "6. For grouped/aggregated results (e.g. sum by category, count by folder, average by group): "
            "   format each row as 'Label: value' lines, or build a readable summary string. "
            "   Never return just a bare list like [100, 200, 300] — always pair values with their labels.\n"
            "7. `reasoning` must be a short plain-English explanation of the steps taken.\n"
            "8. Handle edge cases: missing values, empty results, type mismatches. "
            "   If the result is empty, set answer to a helpful message like 'No data found for ...'.\n\n"

            "OUTPUT FORMAT EXAMPLES:\n"
            "  Bad:  answer = [1200, 340, 780]\n"
            "  Good: answer = 'Sales by Region:\\n  North: 1200\\n  South: 340\\n  West: 780'\n\n"
            "  Bad:  answer = df.groupby('folder')['size'].sum().tolist()\n"
            "  Good: result = df.groupby('folder')['size'].sum().reset_index()\n"
            "        answer = '\\n'.join(f\"{row['folder']}: {row['size']:,.0f}\" for _, row in result.iterrows())\n\n"

            "AVAILABLE VARIABLES:\n"
            "  - `df`       : pandas DataFrame with the full CSV data\n"
            "  - `pd`       : pandas module\n"
            "  - `np`       : numpy module\n"
            "  - `metadata` : dict with dataset info\n\n"

            "DATASET INFO:\n"
            f"  file_name:    {metadata.get('file_name')}\n"
            f"  row_count:    {metadata.get('row_count')}\n"
            f"  column_count: {metadata.get('column_count')}\n"
            f"  columns:\n{col_details}\n\n"
            f"  sample_rows:\n{sample_preview}\n\n"

            f"USER QUESTION: {query}\n\n"

            "Write complete Python code below. End with `answer` (a readable string) and `reasoning` (a short explanation). "
            "No markdown, no comments, no imports."
        )
        return prompt

    def _generate_python_code(self, prompt: str) -> str:
        response = self._call_gemini(
            system_prompt="You are an expert Python data analyst writing thorough, well-labeled pandas code.",
            user_prompt=prompt,
        )
        return self._extract_code(response)

    def _extract_code(self, text: str) -> str:
        if not text:
            return ""
        fence_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()
        return text.strip()

    def _sanitize_code(self, code: str) -> str:
        forbidden = [
            "import matplotlib", "import seaborn", "plot(", "imshow(",
            "hist(", "bar(", "scatter(", "plotly", "altair"
        ]
        lowercase = code.lower()
        for term in forbidden:
            if term in lowercase:
                raise ValueError(
                    f"Generated code contains a forbidden term '{term}' (plotting/visualization is not allowed)."
                )

        if re.search(r"^\s*(if|while)\s+df(\s|\[|\.|$)", code, re.IGNORECASE | re.MULTILINE):
            raise ValueError(
                "Generated code uses a DataFrame directly in a boolean condition. "
                "Use `.empty`, `.any()`, or `.all()` instead."
            )

        # Strip import lines — everything is pre-injected into exec environment
        clean_lines = [
            line for line in code.splitlines()
            if not re.match(r"^\s*(import |from \S+ import )", line)
        ]
        return "\n".join(clean_lines).strip()

    def _execute_code(self, code: str, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not code:
            return {
                "insight": "No code was generated.",
                "reasoning": "Code generation returned no executable Python.",
                "output": "",
            }

        safe_builtins = {
            "len": len, "min": min, "max": max, "sum": sum,
            "sorted": sorted, "round": round, "str": str,
            "int": int, "float": float, "bool": bool,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "enumerate": enumerate, "range": range, "abs": abs,
            "any": any, "all": all, "print": print,
            "isinstance": isinstance, "type": type, "zip": zip,
            "map": map, "filter": filter, "format": format,
        }

        exec_globals = {
            "pd": pd,
            "np": np,
            "df": df,
            "metadata": metadata,
            "__builtins__": safe_builtins,
        }
        exec_locals: Dict[str, Any] = {"answer": None, "reasoning": None}

        try:
            exec(code, exec_globals, exec_locals)
        except Exception as exc:
            message = str(exc)
            if "truth value of a DataFrame is ambiguous" in message or "truth value of a Series is ambiguous" in message:
                error_message = (
                    "Code execution failed: a pandas DataFrame or Series was used in a boolean context. "
                    "Use `.empty`, `.any()`, or `.all()` instead of `if df` or `if df[...]`."
                )
            else:
                error_message = f"Code execution failed: {exc}"
            return {
                "insight": error_message,
                "reasoning": "The generated code could not execute successfully.",
                "output": "",
            }

        answer = exec_locals.get("answer")
        reasoning = exec_locals.get("reasoning")

        if answer is None:
            answer = exec_locals.get("result")
        if answer is None:
            answer = "The generated code did not produce an answer."
        if reasoning is None:
            reasoning = "Code executed, but no reasoning variable was defined."

        # If answer is still a pandas object, do a best-effort readable conversion
        if isinstance(answer, pd.DataFrame):
            answer = answer.to_string(index=False)
        elif isinstance(answer, pd.Series):
            answer = "\n".join(f"{idx}: {val}" for idx, val in answer.items())
        else:
            answer = str(_serialize_value(answer))

        return {
            "insight": answer,
            "reasoning": str(reasoning),
            "output": answer,
        }

    def _describe_context(self, metadata: Dict[str, Any]) -> str:
        columns = [col.get("name") for col in metadata.get("columns", [])]
        return (
            f"CSV file '{metadata.get('file_name')}' with {metadata.get('row_count')} rows, "
            f"{metadata.get('column_count')} columns ({', '.join(columns)})."
        )