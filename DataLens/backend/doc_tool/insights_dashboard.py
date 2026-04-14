import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from pydantic import PrivateAttr
import google.generativeai as genai

from agentpro.tools.base_tool import Tool
from ..config import TEMP_ROOT


class DashboardGeneratorTool(Tool):
    name: str = "Dashboard Generator"
    action_type: str = "dashboard_generator"
    input_format: str = "A dict containing {'session_id': '<uuid>', 'query': '<dashboard description>', 'data': '<any text, numbers, tables, JSON — from any source>'}"
    description: str = (
        "Generates a self-contained interactive HTML dashboard from ANY data source — "
        "RAG, CSV, yfinance, web search, or raw text/numbers. "
        "Pass all relevant data as the 'data' field. Saves to temp/{uuid}/outputs/dashboards/dashboard.html."
    )

    _llm: Any = PrivateAttr()
    _temp_root: Path = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self._llm = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config={"max_output_tokens": 8000},  # enough for a rich dashboard
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
        data       = str(input_data.get("data", ""))[:6000]  # raised cap for richer data

        if not session_id or not query:
            return self._error("Missing session_id or query.")

        output_dir  = self._temp_root / session_id / "outputs" / "dashboards"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dashboard.html"

        prompt = self._build_prompt(query, data)
        html = self._extract_html(self._call_gemini(prompt))

        if not html or len(html) < 200:
            return self._error("LLM did not return valid HTML.")

        output_path.write_text(html, encoding="utf-8")

        troot = str(self._temp_root)
        pstr  = str(output_path)
        rel   = pstr[len(troot):].replace("\\", "/").lstrip("/")
        url   = f"/temp/{rel}"

        return {
            "dashboard_url":  url,
            "dashboard_path": pstr,
            "dashboard":      url,
            "reasoning":      f"Dashboard saved to {url}",
            "llm_response":   f"Dashboard ready at: {url}",
        }

    def _build_prompt(self, query: str, data: str) -> str:
        return f"""You are a senior data visualization engineer building Power BI-style HTML dashboards.

Your job: produce ONE complete self-contained HTML file that feels like a professional BI report.

═══════════════════════════════════════════
STRICT OUTPUT RULES
═══════════════════════════════════════════
- Return ONLY raw HTML starting with <!DOCTYPE html>. Zero markdown, zero explanation.
- All CSS and JS must be inline — no external files except CDN libraries.
- CDN allowed: Chart.js (https://cdn.jsdelivr.net/npm/chart.js), no others needed.
- Embed ALL data as hardcoded inline JS variables. Use only real values from the data provided.
- Do NOT hallucinate numbers. If a value isn't in the data, derive it (sum, average, count, max, min) or skip it.

═══════════════════════════════════════════
DESIGN — POWER BI DARK THEME
═══════════════════════════════════════════
Background:     #0f1117  (near-black)
Card surface:   #1a1d27
Accent blue:    #4f8ef7
Accent green:   #2ecc71
Accent orange:  #f39c12
Accent red:     #e74c3c
Text primary:   #e8eaf0
Text muted:     #8b8fa8
Border/divider: #2a2d3e
Font: 'Segoe UI', system-ui, sans-serif

Layout rules:
- Full-width header with dashboard title and a subtitle (date or data source note)
- Row of KPI metric cards (3–5 cards). Each card shows: icon + label + big bold value + a small trend or sub-note
- At least 2–3 charts below the cards. Charts should be small-to-medium (max 320px tall), NOT massive full-page plots.
- After the charts, add an "Insights" section: 3–5 bullet points of plain-English observations derived from the data
  (e.g. "Category A accounts for 42% of total", "Peak value was X in period Y", "Average is Z")
- Optional: a small data summary table if the data has multiple named rows/columns

Animations:
- Cards fade+slide up on load (CSS keyframe, 0.4s staggered)
- Charts animate in with Chart.js built-in animation (duration: 1000ms, easing: easeOutQuart)
- Subtle hover lift on cards (transform: translateY(-3px), box-shadow)

═══════════════════════════════════════════
CHARTS — WHAT TO GENERATE
═══════════════════════════════════════════
Look at the data and automatically choose the most meaningful chart types:
- If there are categories + numeric values → Bar chart (horizontal or vertical)
- If there is time series / sequential data → Line chart
- If there are proportions / shares → Doughnut chart
- If there are multiple metrics across groups → Grouped bar or radar
- Always label axes, show a legend, and use the accent color palette above.
- Chart.js config must include: responsive: true, maintainAspectRatio: true

═══════════════════════════════════════════
KPI CARDS — WHAT TO CALCULATE
═══════════════════════════════════════════
From the data, always derive and show these if possible:
- Total / Grand Sum
- Count of unique items / categories
- Maximum value and which item it belongs to
- Minimum value and which item it belongs to
- Average value
If the data doesn't support one of these, replace it with a relevant derived metric.

═══════════════════════════════════════════
DATA TO VISUALIZE
═══════════════════════════════════════════
{data}

═══════════════════════════════════════════
USER REQUEST
═══════════════════════════════════════════
{query}

Now output the complete HTML file. Start with <!DOCTYPE html> immediately:"""

    def _extract_html(self, text: str) -> str:
        fence = re.search(r"```(?:html)?\s*(<!DOCTYPE[\s\S]*?)```", text, re.IGNORECASE)
        if fence:
            return fence.group(1).strip()
        start = text.find("<!DOCTYPE")
        if start == -1:
            start = text.find("<html")
        return text[start:].strip() if start != -1 else text.strip()

    def _error(self, msg: str) -> Dict[str, Any]:
        return {"dashboard_url": "", "dashboard_path": "", "reasoning": msg, "llm_response": msg}