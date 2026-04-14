import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import PrivateAttr
import google.generativeai as genai

from agentpro.tools.base_tool import Tool
from ..config import TEMP_ROOT


class PlotGeneratorTool(Tool):
    name: str = "Plot Generator"
    action_type: str = "plot_generator"
    input_format: str = "A dict containing {'session_id': '<uuid>', 'query': '<what to plot>', 'data': '<any text, numbers, tables, JSON — from any source>'}"
    description: str = (
        "Generates matplotlib plots from ANY data source — RAG, CSV, yfinance, web search, or raw text/numbers. "
        "Pass all relevant data as the 'data' field. Saves PNGs to temp/{uuid}/outputs/plot_generated/."
    )

    _llm: Any = PrivateAttr()
    _temp_root: Path = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self._llm = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config={"max_output_tokens": 2000},
        )
        self._temp_root = Path(TEMP_ROOT)

    def _call_gemini(self, prompt: str) -> str:
        return self._llm.generate_content(prompt).text or ""

    def run(self, input_data: Any) -> Dict[str, Any]:
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                return self._error("Invalid input format.")

        if not isinstance(input_data, dict):
            return self._error("Expected a dict.")

        session_id = input_data.get("session_id")
        query      = input_data.get("query", "")
        data       = str(input_data.get("data", ""))[:2000]  # cap at 2000 chars

        if not session_id or not query:
            return self._error("Missing session_id or query.")

        output_dir = self._temp_root / session_id / "outputs" / "plot_generated"
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt = f"""You are a Python matplotlib expert. Generate plotting code based on the data and request below.

RULES:
- No import statements — available: `plt`, `pd`, `np`, `os`, `output_dir` (str), `plot_paths` (list), `reasoning` (set this)
- Save each figure: plt.savefig(os.path.join(output_dir, "plot_N.png"), dpi=150, bbox_inches="tight") then plt.close()
- Append each saved path string to `plot_paths`
- Do NOT call plt.show()
- Use only the real data provided — no hallucination
- Use plt.style.use('seaborn-v0_8-whitegrid'), clear titles and labels
- Set `reasoning` to explain what was plotted
- Return ONLY executable Python code, no markdown, no imports

DATA:
{data}

USER REQUEST: {query}

Write the code now:"""

        code = self._extract_code(self._call_gemini(prompt))
        code = self._strip_imports(code)

        result = self._execute(code, output_dir)
        return {
            "plot_paths":   result.get("plot_paths", []),
            "reasoning":    result.get("reasoning", ""),
            "code":         code,
            "llm_response": result.get("reasoning", ""),
        }

    def _extract_code(self, text: str) -> str:
        fence = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        return fence.group(1).strip() if fence else text.strip()

    def _strip_imports(self, code: str) -> str:
        lines = [l for l in code.splitlines()
                 if not re.match(r"^\s*(import |from \S+ import )", l)
                 and "plt.show()" not in l]
        return "\n".join(lines).strip()

    def _execute(self, code: str, output_dir: Path) -> Dict[str, Any]:
        if not code:
            return {"plot_paths": [], "reasoning": "No code generated."}

        exec_globals = {
            "plt": plt, "pd": pd, "np": np, "os": os,
            "output_dir": str(output_dir),
            "__builtins__": {
                "len": len, "min": min, "max": max, "sum": sum,
                "range": range, "round": round, "str": str, "int": int,
                "float": float, "list": list, "dict": dict, "zip": zip,
                "enumerate": enumerate, "sorted": sorted, "print": print,
                "isinstance": isinstance, "abs": abs, "any": any, "all": all,
            },
        }
        exec_locals = {"plot_paths": [], "reasoning": None}

        try:
            exec(code, exec_globals, exec_locals)
        except Exception as e:
            plt.close("all")
            return {"plot_paths": [], "reasoning": f"Execution failed: {e}"}

        raw   = exec_locals.get("plot_paths", [])
        troot = str(self._temp_root)
        paths = []
        for p in raw:
            s = str(p)
            rel = s[len(troot):].replace("\\", "/").lstrip("/") if s.startswith(troot) else s
            paths.append(f"/temp/{rel}")

        return {"plot_paths": paths, "reasoning": str(exec_locals.get("reasoning") or "Done.")}

    def _error(self, msg: str) -> Dict[str, Any]:
        return {"plot_paths": [], "reasoning": msg, "code": "", "llm_response": msg}