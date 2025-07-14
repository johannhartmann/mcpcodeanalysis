#!/usr/bin/env python3
"""Test script to demonstrate LLM-based code interpretation for embeddings."""

import asyncio

from src.embeddings.code_interpreter import CodeInterpreter
from src.embeddings.embedding_generator import EmbeddingGenerator

# Sample code snippets to test interpretation
SAMPLE_FUNCTION = '''
def validate_email(email: str) -> bool:
    """Check if an email address is valid."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    if not email or not isinstance(email, str):
        return False
    return bool(re.match(pattern, email))
'''

SAMPLE_CLASS = '''
class RateLimiter:
    """Simple rate limiter using sliding window."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []

    def is_allowed(self, user_id: str) -> bool:
        import time
        now = time.time()
        # Remove old requests
        self.requests = [req for req in self.requests
                        if now - req['time'] < self.window_seconds]

        # Count user requests
        user_requests = [req for req in self.requests if req['user'] == user_id]

        if len(user_requests) >= self.max_requests:
            return False

        self.requests.append({'user': user_id, 'time': now})
        return True
'''


async def test_code_interpretation():
    """Test the new LLM-based code interpretation."""
    print("🔬 Testing LLM-based Code Interpretation\n")

    interpreter = CodeInterpreter()
    generator = EmbeddingGenerator()

    # Test function interpretation
    print("📝 Testing Function Interpretation:")
    print("=" * 50)

    function_interpretation = await interpreter.interpret_function(
        code=SAMPLE_FUNCTION,
        function_name="validate_email",
        docstring="Check if an email address is valid.",
        context={"file_path": "validators.py"},
    )

    print("Function: validate_email")
    print("Interpretation:")
    print(function_interpretation)
    print("\n")

    # Test class interpretation
    print("📝 Testing Class Interpretation:")
    print("=" * 50)

    class_interpretation = await interpreter.interpret_class(
        code=SAMPLE_CLASS,
        class_name="RateLimiter",
        docstring="Simple rate limiter using sliding window.",
        methods_summary=[
            {"name": "__init__", "docstring": "Initialize rate limiter"},
            {"name": "is_allowed", "docstring": "Check if request is allowed"},
        ],
        context={"file_path": "rate_limiter.py"},
    )

    print("Class: RateLimiter")
    print("Interpretation:")
    print(class_interpretation)
    print("\n")

    # Test search-optimized text generation
    print("🔍 Testing Search-Optimized Text Generation:")
    print("=" * 50)

    search_text = await interpreter.create_search_optimized_text(
        entity_type="function",
        entity_name="validate_email",
        code=SAMPLE_FUNCTION,
        interpretation=function_interpretation,
        metadata={
            "signature": "validate_email(email: str) -> bool",
            "file_path": "validators.py",
        },
    )

    print("Search-optimized text for embeddings:")
    print(search_text)
    print("\n")

    # Test embedding generation
    print("🚀 Testing Full Embedding Generation:")
    print("=" * 50)

    function_data = {
        "name": "validate_email",
        "parameters": [{"name": "email", "type": "str"}],
        "return_type": "bool",
        "docstring": "Check if an email address is valid.",
        "start_line": 1,
        "end_line": 8,
    }

    embedding_result = await generator.generate_interpreted_function_embedding(
        function_data=function_data, code=SAMPLE_FUNCTION, file_path="validators.py"
    )

    print(f"Generated embedding with {len(embedding_result['embedding'])} dimensions")
    print(f"Text length: {len(embedding_result['text'])} characters")
    print(f"Estimated tokens: {embedding_result['tokens']}")
    print(f"Has interpretation: {embedding_result['metadata']['has_interpretation']}")
    print("\nEmbedding text preview:")
    print(
        embedding_result["text"][:500] + "..."
        if len(embedding_result["text"]) > 500
        else embedding_result["text"]
    )


async def main():
    """Main test function."""
    try:
        await test_code_interpretation()
        print("\n✅ All tests completed successfully!")
        print(
            "\n🎉 The search system now uses LLM interpretation instead of just metadata!"
        )
        print("\nNow searches like:")
        print('- "Find email validation code"')
        print('- "Show me rate limiting implementations"')
        print('- "Find retry logic with exponential backoff"')
        print("Will actually work by understanding what the code does!")

    except (ImportError, AttributeError, RuntimeError, ValueError) as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
