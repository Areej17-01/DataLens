import os
import uuid
from typing import List

import numpy as np
from qdrant_client import QdrantClient, models


class QdrantVectorStore:
    def __init__(self, url: str, api_key: str, text_dim: int, image_dim: int):
        self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.collection_prefix = "datalens"

    def _ensure_collection(self, collection_name: str, vector_size: int) -> None:
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def _sanitize_session_id(self, session_id: str) -> str:
        return "".join(char if char.isalnum() else "_" for char in session_id.lower())

    def _text_collection_name(self, session_id: str) -> str:
        return f"{self.collection_prefix}_text_{self._sanitize_session_id(session_id)}"

    def _image_collection_name(self, session_id: str) -> str:
        return f"{self.collection_prefix}_images_{self._sanitize_session_id(session_id)}"

    def ensure_session_collections(self, session_id: str) -> None:
        self._ensure_collection(self._text_collection_name(session_id), self.text_dim)
        self._ensure_collection(self._image_collection_name(session_id), self.image_dim)

    def upload_text_points(self, session_id: str, text_documents: List[dict]) -> None:
        self.ensure_session_collections(session_id)
        points = [
            models.PointStruct(
                id=doc["id"],
                vector=np.array(doc["embedding"]),
                payload={
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                },
            )
            for doc in text_documents
        ]
        if points:
            self.client.upload_points(collection_name=self._text_collection_name(session_id), points=points)

    def upload_image_points(self, session_id: str, image_paths: List[str], embeddings: List[np.ndarray]) -> None:
        self.ensure_session_collections(session_id)
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}_{os.path.basename(image_path)}_{idx}")),
                vector=np.array(embedding),
                payload={
                    "image_path": image_path,
                },
            )
            for idx, (image_path, embedding) in enumerate(zip(image_paths, embeddings))
        ]
        if points:
            self.client.upload_points(collection_name=self._image_collection_name(session_id), points=points)

    def query_text(self, session_id: str, query_vector: np.ndarray, limit: int = 3):
        collection_name = self._text_collection_name(session_id)
        if not self.client.collection_exists(collection_name):
            return []
        return self._search_points(collection_name, query_vector, limit)

    def query_images(self, session_id: str, query_vector: np.ndarray, limit: int = 3):
        collection_name = self._image_collection_name(session_id)
        if not self.client.collection_exists(collection_name):
            return []
        return self._search_points(collection_name, query_vector, limit)

    def _search_points(self, collection_name: str, query_vector: np.ndarray, limit: int):
        result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
        )
        return result.points