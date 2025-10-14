"""OpenRouter API client."""

from abc import ABC, abstractmethod
from typing import Any
import httpx
import os
from dotenv import load_dotenv

load_dotenv()


class LLMClient(ABC):
    """Abstract LLM client."""

    @abstractmethod
    def complete(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Generate completion from messages."""
        pass


class OpenRouterClient(LLMClient):
    """OpenRouter API client."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        self.model = model
        self.client = httpx.Client(timeout=60.0)

    def complete(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Generate completion via OpenRouter.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Response dict with content, usage, etc.
        """
        # TODO: Implement API call
        # - POST to /chat/completions
        # - Handle errors and retries
        # - Track token usage and cost
        # raise NotImplementedError
        # TODO: add retry with tenacity
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {"model": self.model, "messages": messages, **kwargs}

        response = self.client.post(
            f"{self.BASE_URL}/chat/completions", json=payload, headers=headers
        )

        response.raise_for_status()

        data = response.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
            "model": data.get("model"),
        }
