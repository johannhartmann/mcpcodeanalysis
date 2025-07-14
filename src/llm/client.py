"""LLM client stub for migration intelligence features."""

from typing import Any


class LLMClient:
    """Stub LLM client for testing migration intelligence."""

    def __init__(self, model: str = "gpt-4", api_key: str | None = None):
        """Initialize LLM client.

        Args:
            model: Model name to use
            api_key: API key for authentication
        """
        self.model = model
        self.api_key = api_key

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        # Stub implementation - returns empty string
        return ""

    async def analyze(self, content: str, instructions: str) -> dict[str, Any]:
        """Analyze content with specific instructions.

        Args:
            content: Content to analyze
            instructions: Analysis instructions

        Returns:
            Analysis results
        """
        # Stub implementation - returns empty dict
        return {}
