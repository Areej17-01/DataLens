from .base_tool import Tool
from typing import Any, Optional, Dict
from abc import ABC, abstractmethod
from pydantic import BaseModel, PrivateAttr
import math
import requests
import json
import os

class UserInputTool(Tool):
    name: str = "User Input Requester"
    description: str = "Requests more information from the user to continue solving the task."
    action_type: str = "request_user_input"
    input_format: str = "A string that defines the prompt/question to ask the user. Example: 'Please provide more details about your project goals.'"

    def run(self, input_text: Any) -> str:  # <<< Change 'input' to 'input_text'
        if not isinstance(input_text, str):
            return "Error: Expected a prompt string to request user input."
        
        # Ask user for input properly
        user_response = input(f"\n🧠 AI assistant: {input_text}\n👤 User response: ")
        return user_response