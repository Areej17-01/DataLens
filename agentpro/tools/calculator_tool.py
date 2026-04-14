from .base_tool import Tool
from typing import Any, Optional, Dict
from abc import ABC, abstractmethod
from pydantic import BaseModel, PrivateAttr
import math
import requests
import json
import os

# Calculator tool
class CalculateTool(Tool):
    name: str = "Calculator"
    description: str = "Evaluates a basic math expression and returns the result."
    action_type: str = "calculate"
    input_format: str = "A math expression as a string. Example: '2 + 3 * (5 - 1)'"

    def run(self, input_text: Any) -> str:
        expr = input_text
        try:
            # Allow only safe characters
            safe_expr = ''.join(c for c in expr if c in '0123456789+-*/(). ')
            result = eval(safe_expr)
            return str(result)
        except Exception:
            return "Error: Invalid calculation."