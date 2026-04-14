# import os
# import json
# from typing import Any, Dict, List

# from pydantic import PrivateAttr

# from agentpro import create_model
# from agentpro.tools.base_tool import Tool

# from ..services.embeddings import EmbeddingService
# from ..services.qdrant_store import QdrantVectorStore


# class DataLensRAGTool(Tool):
#     name: str = "DataLens Document RAG"
#     action_type: str = "datalens_rag"
#     input_format: str = "A dict containing {'session_id': '<uuid>', 'query': '<question>'}"
#     description: str = (
#         "Retrieves relevant text and image context from uploaded documents and returns an LLM answer."
#     )
#     _qdrant_store: QdrantVectorStore = PrivateAttr()
#     _embeddings: EmbeddingService = PrivateAttr()
#     _llm: Any = PrivateAttr()

#     def __init__(
#         self,
#         qdrant_store: QdrantVectorStore,
#         embeddings: EmbeddingService,
#         llm=None,
#         **data,
#     ):
#         super().__init__(**data)
#         self._qdrant_store = qdrant_store
#         self._embeddings = embeddings
#         self._llm = llm or create_model(
#             provider="openrouter",
#             model_name=os.getenv("OPEN_ROUTER_MODEL", "z-ai/glm-4.5-air:free"),
#             api_key=os.getenv("OPEN_ROUTER_KEY"),
#         )

#     def run(self, input_data: Any) -> Dict[str, Any]:
#         if isinstance(input_data, str):
#             try:
#                 input_data = json.loads(input_data)
#             except json.JSONDecodeError:
#                 return {
#                     "llm_response": "Error: Expected input format {'session_id': '<uuid>', 'query': '<question>'}.",
#                     "retrieved_context": "",
#                     "image_paths": [],
#                 }
#         if not isinstance(input_data, dict):
#             return {
#                 "llm_response": "❌ Error: Expected input format {'session_id': '<uuid>', 'query': '<question>'}",
#                 "retrieved_context": "",
#                 "image_paths": [],
#             }

#         session_id = input_data.get("session_id")
#         query = input_data.get("query")

#         if not session_id or not query:
#             return {
#                 "llm_response": "❌ Error: Missing session_id or query.",
#                 "retrieved_context": "",
#                 "image_paths": [],
#             }

#         query_vector = self._embeddings.get_text_embeddings(query)
#         text_hits = self._qdrant_store.query_text(session_id, query_vector, limit=3)
#         image_hits = self._qdrant_store.query_images(session_id, query_vector, limit=3)

#         retrieved_texts = [hit.payload["content"] for hit in text_hits if hit.payload.get("content")]
#         retrieved_context = "\n\n---\n\n".join(retrieved_texts)

#         image_paths = []
#         for hit in image_hits:
#             image_path = hit.payload.get("image_path")
#             if image_path:
#                 image_paths.append(f"/temp/{session_id}/{image_path}")

#         prompt = self._build_prompt(query, retrieved_context, image_paths)
#         llm_response = self._call_llm(prompt)

#         return {
#             "llm_response": llm_response,
#             "retrieved_context": retrieved_context,
#             "image_paths": image_paths,
#         }

#     def _build_prompt(self, query: str, retrieved_context: str, image_paths: List[str]) -> str:
#         image_section = (
#             "Retrieved images are available and can be reviewed via the paths below:\n" + "\n".join(image_paths)
#             if image_paths
#             else "No images were retrieved for this query."
#         )

#         return f"""
# Based on the given document context and retrieved images, answer the user query using only the provided context.

# User query: {query}

# Retrieved text context:
# {retrieved_context}

# {image_section}

# Provide a concise, accurate answer based on the retrieved document content. Do not hallucinate or invent information.
# """

#     def _call_llm(self, prompt: str) -> str:
#         return self._llm.chat_completion(system_prompt="You are a helpful document assistant.", user_prompt=prompt)


import os
import json
import base64
from typing import Any, Dict, List, Optional

import requests
from pydantic import PrivateAttr

from agentpro import create_model
from agentpro.tools.base_tool import Tool

from ..services.embeddings import EmbeddingService
from ..services.qdrant_store import QdrantVectorStore

# Base directory where temp files are stored
TEMP_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "temp")

# Free vision-capable models on OpenRouter, tried in order
VISION_MODELS_FALLBACK = [
    "google/gemma-3-4b-it:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    
]


class DataLensRAGTool(Tool):
    name: str = "DataLens Document RAG"
    action_type: str = "datalens_rag"
    input_format: str = "A dict containing {'session_id': '<uuid>', 'query': '<question>'}"
    description: str = (
        "Retrieves relevant text and image context from uploaded documents and returns an LLM answer."
    )
    _qdrant_store: QdrantVectorStore = PrivateAttr()
    _embeddings: EmbeddingService = PrivateAttr()
    _llm: Any = PrivateAttr()
    _openrouter_api_key: str = PrivateAttr()

    def __init__(
        self,
        qdrant_store: QdrantVectorStore,
        embeddings: EmbeddingService,
        llm=None,
        **data,
    ):
        super().__init__(**data)
        self._qdrant_store = qdrant_store
        self._embeddings = embeddings
        self._llm = llm or create_model(
            provider="openrouter",
            model_name=os.getenv("OPEN_ROUTER_MODEL", "google/gemma-3-27b-it:free"),
            api_key=os.getenv("OPEN_ROUTER_KEY"),
        )
        self._openrouter_api_key = os.getenv("OPEN_ROUTER_KEY", "")

    def run(self, input_data: Any) -> Dict[str, Any]:
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                return {
                    "llm_response": "Error: Expected input format {'session_id': '<uuid>', 'query': '<question>'}.",
                    "retrieved_context": "",
                    "image_paths": [],
                }
        if not isinstance(input_data, dict):
            return {
                "llm_response": "❌ Error: Expected input format {'session_id': '<uuid>', 'query': '<question>'}",
                "retrieved_context": "",
                "image_paths": [],
            }

        session_id = input_data.get("session_id")
        query = input_data.get("query")

        if not session_id or not query:
            return {
                "llm_response": "❌ Error: Missing session_id or query.",
                "retrieved_context": "",
                "image_paths": [],
            }

        query_vector = self._embeddings.get_text_embeddings(query)
        text_hits = self._qdrant_store.query_text(session_id, query_vector, limit=3)
        image_hits = self._qdrant_store.query_images(session_id, query_vector, limit=3)

        retrieved_texts = [hit.payload["content"] for hit in text_hits if hit.payload.get("content")]
        retrieved_context = "\n\n---\n\n".join(retrieved_texts)

        # Build frontend-facing image paths and local disk paths
        image_paths = []
        local_image_paths = []
        for hit in image_hits:
            image_path = hit.payload.get("image_path")
            if image_path:
                image_paths.append(f"/temp/{session_id}/{image_path}")
                local_path = os.path.normpath(
                    os.path.join(TEMP_BASE_DIR, session_id, image_path)
                )
                local_image_paths.append(local_path)

        # Try vision call with top 2 images, fall back to text-only if all models fail
        if local_image_paths:
            llm_response = self._call_vision_llm_with_fallback(query, retrieved_context, local_image_paths[:2])
        else:
            llm_response = self._call_text_llm_safe(query, retrieved_context)

        return {
            "llm_response": llm_response,
            "retrieved_context": retrieved_context,
            "image_paths": image_paths,
        }

    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"⚠️ Image not found: {image_path}")
            return None

    def _call_vision_llm_with_fallback(self, query: str, retrieved_context: str, local_image_paths: List[str]) -> str:
        """Try each vision model in order. If all are rate-limited, fall back to text-only."""
        content = [
            {
                "type": "text",
                "text": (
                    f"You are a helpful document assistant.\n\n"
                    f"Answer the user query using the retrieved text context and the images provided below.\n\n"
                    f"User query: {query}\n\n"
                    f"Retrieved text context:\n{retrieved_context}\n\n"
                    f"Now also analyze the following retrieved document images to support your answer. "
                    f"Do not hallucinate or invent information."
                ),
            }
        ]

        for path in local_image_paths:
            b64 = self._encode_image_to_base64(path)
            if b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })

        for model in VISION_MODELS_FALLBACK:
            try:
                print(f"🔍 Trying vision model: {model}")
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": content}],
                    },
                    timeout=60,
                )
                response.raise_for_status()
                print(f"✅ Vision model succeeded: {model}")
                return response.json()["choices"][0]["message"]["content"]

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    print(f"⚠️ Rate limited on {model}, trying next...")
                    continue
                print(f"⚠️ HTTP error on {model}: {e}")
                continue
            except Exception as e:
                print(f"⚠️ Unexpected error on {model}: {e}")
                continue

        # All vision models failed — fall back to text-only
        print("⚠️ All vision models failed. Falling back to text-only.")
        return self._call_text_llm_safe(query, retrieved_context)

    def _call_text_llm_safe(self, query: str, retrieved_context: str) -> str:
        """Text-only LLM call with safe error handling."""
        prompt = self._build_text_prompt(query, retrieved_context)
        try:
            return self._llm.chat_completion(
                system_prompt="You are a helpful document assistant.",
                user_prompt=prompt,
            )
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                return (
                    "⚠️ The AI model is currently rate-limited. "
                    "Please wait a moment and try again."
                )
            return f"⚠️ An error occurred while generating a response: {error_str}"

    def _build_text_prompt(self, query: str, retrieved_context: str) -> str:
        return f"""
Based on the given document context, answer the user query using only the provided context.

User query: {query}

Retrieved text context:
{retrieved_context}

Provide a concise, accurate answer based on the retrieved document content. Do not hallucinate or invent information.
"""