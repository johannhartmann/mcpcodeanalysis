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

    async def generate(self, _prompt: str, **_kwargs: Any) -> str:
        """Generate a response from the LLM.

        Args:
            _prompt: Input prompt (unused in stub)
            **_kwargs: Additional generation parameters (unused in stub)

        Returns:
            Generated text response
        """
        # Stub implementation - returns empty string
        return ""

    async def analyze(self, _content: str, _instructions: str) -> dict[str, Any]:
        """Analyze content with specific instructions.

        Args:
            _content: Content to analyze (unused in stub)
            _instructions: Analysis instructions (unused in stub)

        Returns:
            Analysis results
        """
        # Stub implementation - returns empty dict
        return {}
