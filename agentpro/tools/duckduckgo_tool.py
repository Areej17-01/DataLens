from .base_tool import Tool
from typing import Any, Optional, Dict
from abc import ABC, abstractmethod
from pydantic import BaseModel, PrivateAttr
import math
import requests
import json
import os

# Try importing DuckDuckGo Search
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("Warning: duckduckgo-search not installed. Using mock search instead.")


# DuckDuckGo search tool
class QuickInternetTool(Tool):
    name: str = "Quick Internet Search"
    description: str = "Searches internet quickly using DuckDuckGo for a given query and returns top 5 results."
    action_type: str = "search"
    input_format: str = "A search query as a string. Example: 'Latest advancements in AI'"

    ddg: Optional[Any] = None  # Important: Declare ddg properly for Pydantic

    def __init__(self, **data):
        super().__init__(**data)
        # Set ddg safely even with BaseModel
        object.__setattr__(self, 'ddg', DDGS() if DDGS_AVAILABLE else None)

    def run(self, input_text: Any) -> str:
        query = input_text
        if not self.ddg:
            return f"Mock search results for: {query}\n1. Sample Result\n2. Another Result"

        try:
            results = list(self.ddg.text(query, max_results=5))
            if not results:
                return "No results found."
            return "\n".join(
                f"{i+1}. {r['title']}\n   {r['body']}" for i, r in enumerate(results)
            )
        except Exception as e:
            return f"Error performing search: {str(e)}"