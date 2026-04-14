from .base_tool import Tool
import json
from typing import Any, Optional, Dict
from pydantic import PrivateAttr
import requests
import os

class TraversaalProRAGTool(Tool):
    name: str = "Traversaal Pro RAG"
    action_type: str = "traversaalpro_rag"
    input_format: str = "A query string for document search. Example: 'chemical safety protocol'"
    description: str = "Searches documents using the Traversaal Pro RAG API and returns a context-aware answer and document excerpts."

    _config: Dict[str, Any] = PrivateAttr()

    def __init__(self, api_key: Optional[str] = None, document_names: Optional[str] = None, timeout: int = 30, **data):
        if document_names:
            data["description"] = (
                f"Searches {document_names} documents using the Traversaal Pro RAG API and returns a context-aware answer and document excerpts."
            )

        super().__init__(**data)
        
        # Store the API key and timeout as private attributes
        self._config = {
            "api_key": api_key or os.getenv("TRAVERSAAL_PRO_API_KEY"),
            "timeout": timeout
        }
        

    def run(self, input_text: Any) -> str:
            
        if not isinstance(input_text, str):
            return "❌ Error: Expected a query string. Example: 'chemical safety protocol'"

        # Validate API key
        api_key = self._config.get("api_key")
        timeout = self._config.get("timeout", 30)  # Default to 30 seconds if not specified

        if not api_key:
            return "❌ Error: API key is required. Provide it during initialization or set TRAVERSAAL_PRO_API_KEY environment variable."

        url = "https://pro-documents.traversaal-api.com/documents/search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "query": input_text.strip("'\""),
            "rag": False
        }

        try:
            # Added timeout parameter to the request
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()  # This will raise an exception for HTTP error codes

            result = response.json()

            return result
            
            # # Use the correct keys from the response
            # answer = result.get("response", "").strip()
            # references = result.get("references", [])

            # if not answer:
            #     return "No answer found for this query. Please try a different question."

            # output = f"**Answer:**\n{answer}\n\n"

            # if references:
            #     output += "**Source Document Snippets:**\n"
            #     for idx, ref in enumerate(references, 1):
            #         file_id = ref.get("file_id", "Unknown")
            #         s3_key = ref.get("s3_bucket_key", "")
            #         file_name = s3_key.split("/")[-1] if s3_key else "Unknown Document"
            #         snippet = ref.get("chunk_text", "").strip()
            #         score = ref.get("score", 0)
                    
            #         output += f"{idx}. *{file_name}* (Relevance: {score:.2f})\n{snippet}...\n\n"

            # return output.strip()

        except requests.exceptions.Timeout:
            return "❌ Error: The request timed out. Please try again later or with a simpler query."
        except requests.exceptions.HTTPError as e:
            return f"❌ API Error: {e.response.status_code} - {e.response.text}"
        except requests.exceptions.RequestException as e:
            return f"❌ HTTP Request Error: {e}"
        except Exception as e:
            return f"❌ Unexpected Error: {e}"
