import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image
import torch

TEXT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
IMAGE_MODEL_NAME = "nomic-ai/nomic-embed-vision-v1.5"


class EmbeddingService:
    def __init__(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True)
        self.text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True)
        self.image_processor = AutoProcessor.from_pretrained(IMAGE_MODEL_NAME, trust_remote_code=True)
        self.image_model = AutoModel.from_pretrained(IMAGE_MODEL_NAME, trust_remote_code=True)

        self.text_model.to("cpu")
        self.image_model.to("cpu")

    @property
    def text_dim(self) -> int:
        return self.text_model.config.hidden_size

    @property
    def image_dim(self) -> int:
        return self.image_model.config.hidden_size

    def get_text_embeddings(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            text = str(text)

        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        self.text_model.eval()

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings[0].cpu().numpy()

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt")
        self.image_model.eval()

        with torch.no_grad():
            outputs = self.image_model(**inputs)
            embeddings = outputs.last_hidden_state

        image_embedding = embeddings.mean(dim=1).squeeze().cpu().numpy()
        return image_embedding

    def embed_text_list(self, texts):
        return [self.get_text_embeddings(text) for text in texts]

    def embed_image_list(self, image_paths):
        return [self.get_image_embedding(path) for path in image_paths]
