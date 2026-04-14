# model.py
from typing import Dict, Any, Optional, List, Union
import openai
import litellm
from litellm import completion
import os

class ModelClient:
    """Base class for different model clients"""
    def __init__(self, model_name: str = None, temperature: float = 0.7, max_tokens: Optional[int] = None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens or 2048  # Default max_tokens if not provided
        
    def chat_completion(self, system_prompt: str, user_prompt: str, 
                       temperature: Optional[float] = None, 
                       max_tokens: Optional[int] = None) -> str:
        """
        Chat completion method. Uses instance defaults if parameters not provided.
        """
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIClient(ModelClient):
    """Client for OpenAI models"""
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o", 
                 temperature: float = 0.7, max_tokens: Optional[int] = None):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    
    def chat_completion(self, system_prompt: str, user_prompt: str, 
                       temperature: Optional[float] = None, 
                       max_tokens: Optional[int] = None) -> str:
        # Use provided parameters or fall back to instance defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temp,
            max_tokens=tokens
        )
        return response.choices[0].message.content
        

class OpenRouterClient(ModelClient):
    """Client for OpenRouter-hosted models via the OpenAI-compatible API."""
    def __init__(self, api_key: str = None, model_name: str = "z-ai/glm-4.5-air:free", 
                 temperature: float = 0.7, max_tokens: Optional[int] = None):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        router_key = api_key or os.environ.get("OPEN_ROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.client = openai.OpenAI(api_key=router_key, base_url="https://openrouter.ai/api/v1")
    
    def chat_completion(self, system_prompt: str, user_prompt: str, 
                       temperature: Optional[float] = None, 
                       max_tokens: Optional[int] = None) -> str:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temp,
            max_tokens=tokens
        )
        return response.choices[0].message.content

class LiteLLMClient(ModelClient):
    """Client for LiteLLM which supports multiple providers"""
    def __init__(self, api_key: str = None, model_name: str = "gpt-4", 
                 litellm_provider: str = None, temperature: float = 0.7, 
                 max_tokens: Optional[int] = None):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.litellm_provider = litellm_provider
        
        # Set up API key for the specified provider
        if self.litellm_provider == "openai":
            os.environ["OPENAI_API_KEY"] = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        elif self.litellm_provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        elif self.litellm_provider == "gemini":
            os.environ["GEMINI_API_KEY"] = self.api_key or os.environ.get("GEMINI_API_KEY", "")
        elif self.litellm_provider == "openrouter":
            openrouter_key = self.api_key or os.environ.get("OPEN_ROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
            os.environ["OPENROUTER_API_KEY"] = openrouter_key
        # Add other providers as needed
    
    def chat_completion(self, system_prompt: str, user_prompt: str, 
                       temperature: Optional[float] = None, 
                       max_tokens: Optional[int] = None) -> str:
        # Use provided parameters or fall back to instance defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # If a specific provider is defined, use it
        model_param = f"{self.litellm_provider}/{self.model_name}"
        
        response = litellm.completion(
            model=model_param,
            messages=messages,
            temperature=temp,
            max_tokens=tokens
        )
        
        return response.choices[0].message.content

class ModelConfig:
    """Configuration class for a LLM model"""
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = None,
        api_key: str = None,
        litellm_provider: str = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.litellm_provider = litellm_provider
        self.temperature = temperature
        self.max_tokens = max_tokens or 2048  # Default max_tokens
        
        # Set defaults based on provider
        if not self.model_name:
            if self.provider == "openai":
                self.model_name = "gpt-4o"
            elif self.provider == "litellm":
                self.model_name = "gpt-4o"
            elif self.provider == "openrouter":
                self.model_name = "z-ai/glm-4.5-air:free"
        
    def create_client(self) -> ModelClient:
        """Create and return a model client based on this configuration"""
        if self.provider == "openai":
            return OpenAIClient(
                api_key=self.api_key, 
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        elif self.provider == "litellm":
            return LiteLLMClient(
                api_key=self.api_key, 
                model_name=self.model_name, 
                litellm_provider=self.litellm_provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        elif self.provider == "openrouter":
            return OpenRouterClient(
                api_key=self.api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

def create_model(
    provider: str = "openai",
    model_name: str = None,
    api_key: str = None,
    litellm_provider: str = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> ModelClient:
    """
    Create and return a model client with the specified configuration
    
    Args:
        provider: The LLM provider (openai, litellm)
        model_name: The specific model to use
        api_key: The API key for the provider
        litellm_provider: For litellm, the specific provider to use
        temperature: The temperature parameter for the model (default: 0.7)
        max_tokens: The maximum tokens for the model (default: 2048)
        
    Returns:
        ModelClient: A configured model client
    """
    config = ModelConfig(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        litellm_provider=litellm_provider,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return config.create_client()