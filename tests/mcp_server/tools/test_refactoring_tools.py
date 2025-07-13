"""Tests for refactoring suggestion tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Class, File, Function
from src.mcp_server.tools.analysis_tools import AnalysisTools


@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_mcp():
    """Create mock FastMCP instance."""
    mcp = MagicMock(spec=FastMCP)
    mcp.tool = MagicMock(side_effect=lambda **kwargs: lambda func: func)
    return mcp


@pytest.fixture
def analysis_tools(mock_db_session, mock_mcp):
    """Create analysis tools fixture with mocked LLM."""
    with (
        patch("src.mcp_server.tools.analysis_tools.settings") as mock_settings,
        patch("src.mcp_server.tools.analysis_tools.ChatOpenAI") as mock_chat,
    ):
        mock_settings.openai_api_key.get_secret_value.return_value = "test-key"
        mock_settings.llm.model = "gpt-4"
        mock_settings.llm.temperature = 0.2

        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_chat.return_value = mock_llm_instance

        tools = AnalysisTools(mock_db_session, mock_mcp)
        tools.llm = mock_llm_instance
        return tools


class TestRefactoringTools:
    """Tests for refactoring suggestion tools."""

    @pytest.mark.asyncio
    async def test_suggest_refactoring_file_not_found(
        self, analysis_tools, mock_db_session
    ):
        """Test suggesting refactoring when file is not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await analysis_tools.suggest_refactoring(
            "nonexistent.py", focus="performance"
        )

        assert result["error"] == "File not found: nonexistent.py"

    @pytest.mark.asyncio
    async def test_suggest_refactoring_complex_function(
        self, analysis_tools, mock_db_session
    ):
        """Test suggesting refactoring for complex function."""
        # Mock file
        mock_file = MagicMock(spec=File)
        mock_file.id = 1
        mock_file.path = "/src/utils/data_processor.py"
        mock_file.repository_id = 10

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock complex function
        mock_function = MagicMock(spec=Function)
        mock_function.name = "process_large_dataset"
        mock_function.complexity_score = 25  # High complexity
        mock_function.start_line = 50
        mock_function.end_line = 150
        mock_function.parameters = '["data", "config", "options"]'
        mock_function.return_type = "Dict[str, Any]"
        mock_function.docstring = "Process large dataset with multiple transformations"

        # Mock nested functions
        mock_nested1 = MagicMock(spec=Function)
        mock_nested1.name = "_validate_data"
        mock_nested1.complexity_score = 8
        mock_nested1.parent_id = mock_function.id

        mock_nested2 = MagicMock(spec=Function)
        mock_nested2.name = "_transform_data"
        mock_nested2.complexity_score = 12
        mock_nested2.parent_id = mock_function.id

        functions_result = MagicMock()
        functions_result.scalars.return_value.all.return_value = [
            mock_function,
            mock_nested1,
            mock_nested2,
        ]

        # Mock file content
        mock_content = '''def process_large_dataset(data, config, options):
    """Process large dataset with multiple transformations."""
    # Validation logic
    if not data:
        raise ValueError("Data is empty")
    if config.get("validate"):
        for item in data:
            if not validate_item(item):
                logger.warning(f"Invalid item: {item}")
                continue

    # Complex transformation logic with nested loops
    results = {}
    for category in data.get("categories", []):
        category_results = []
        for item in category.get("items", []):
            if item.get("type") == "A":
                result = process_type_a(item)
            elif item.get("type") == "B":
                result = process_type_b(item)
            else:
                result = process_default(item)

            if result and result.get("valid"):
                category_results.append(result)

        if category_results:
            results[category["name"]] = category_results

    # More processing...
    return results
'''

        # Mock LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.content = """## Refactoring Suggestions

### 1. Extract Method Pattern
**Issue**: The function is doing too many things (validation, transformation, categorization).
**Suggestion**: Break down into smaller, focused methods:
```python
def process_large_dataset(data, config, options):
    validate_dataset(data, config)
    return transform_dataset(data, config, options)

def validate_dataset(data, config):
    if not data:
        raise ValueError("Data is empty")
    if config.get("validate"):
        validate_items(data)

def transform_dataset(data, config, options):
    transformer = DatasetTransformer(config, options)
    return transformer.transform(data)
```

### 2. Strategy Pattern for Type Processing
**Issue**: Multiple if-elif branches for type processing.
**Suggestion**: Use a strategy pattern:
```python
TYPE_PROCESSORS = {
    "A": process_type_a,
    "B": process_type_b,
}

def process_item(item):
    processor = TYPE_PROCESSORS.get(item.get("type"), process_default)
    return processor(item)
```

### 3. Reduce Nesting Levels
**Issue**: Deep nesting makes code hard to read.
**Suggestion**: Use early returns and extract nested logic.

### 4. Add Type Hints
**Issue**: Complex data structures without type hints.
**Suggestion**: Define data classes or TypedDict for clarity."""

        analysis_tools.llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        # Mock code extraction
        with patch.object(
            analysis_tools.code_extractor,
            "get_file_content",
            return_value=mock_content,
        ):
            mock_db_session.execute.side_effect = [file_result, functions_result]

            result = await analysis_tools.suggest_refactoring(
                "/src/utils/data_processor.py", focus="complexity"
            )

        assert result["file"] == "/src/utils/data_processor.py"
        assert "refactoring_suggestions" in result
        assert "Extract Method Pattern" in result["refactoring_suggestions"]
        assert "Strategy Pattern" in result["refactoring_suggestions"]
        assert result["code_metrics"]["total_functions"] == 3
        assert result["code_metrics"]["max_complexity"] == 25
        assert result["code_metrics"]["avg_complexity"] == pytest.approx(15.0, 0.1)

    @pytest.mark.asyncio
    async def test_suggest_refactoring_code_smells(
        self, analysis_tools, mock_db_session
    ):
        """Test suggesting refactoring for various code smells."""
        # Mock file
        mock_file = MagicMock(spec=File)
        mock_file.id = 1
        mock_file.path = "/src/models/user.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock class with many methods (potential God Object)
        mock_class = MagicMock(spec=Class)
        mock_class.name = "User"
        mock_class.method_count = 25  # Many methods
        mock_class.complexity_score = 150  # High total complexity

        classes_result = MagicMock()
        classes_result.scalars.return_value.all.return_value = [mock_class]

        # Mock functions with code smells
        functions = []

        # Long method
        long_method = MagicMock(spec=Function)
        long_method.name = "process_user_data"
        long_method.start_line = 100
        long_method.end_line = 300  # 200 lines!
        long_method.complexity_score = 30
        long_method.parameters = (
            '["data", "options", "config", "flags", "extra"]'  # Many params
        )
        functions.append(long_method)

        # Method with duplicate logic
        dup_method1 = MagicMock(spec=Function)
        dup_method1.name = "validate_email"
        dup_method1.complexity_score = 5
        functions.append(dup_method1)

        dup_method2 = MagicMock(spec=Function)
        dup_method2.name = "validate_username"
        dup_method2.complexity_score = 5
        functions.append(dup_method2)

        functions_result = MagicMock()
        functions_result.scalars.return_value.all.return_value = functions

        # Mock file content with code smells
        mock_content = '''class User:
    """User model with many responsibilities."""

    def __init__(self, data):
        self.data = data
        self.email = data.get("email")
        self.username = data.get("username")
        # ... many more attributes

    def validate_email(self):
        """Validate email format."""
        if not self.email:
            return False
        if "@" not in self.email:
            return False
        if len(self.email) > 255:
            return False
        # Similar pattern...
        return True

    def validate_username(self):
        """Validate username format."""
        if not self.username:
            return False
        if len(self.username) < 3:
            return False
        if len(self.username) > 50:
            return False
        # Similar pattern...
        return True

    def process_user_data(self, data, options, config, flags, extra):
        """Process user data with many responsibilities."""
        # 200 lines of code doing many things...
        pass
'''

        # Mock LLM response for code smells
        mock_llm_response = MagicMock()
        mock_llm_response.content = """## Code Smell Analysis & Refactoring

### 1. God Object Anti-pattern
**Issue**: User class has 25 methods - doing too much.
**Suggestion**: Apply Single Responsibility Principle:
- Extract UserValidator class for validation logic
- Extract UserSerializer for data transformation
- Extract UserNotifier for notification logic
- Keep User as a simple data model

### 2. Long Method
**Issue**: process_user_data is 200 lines with 5 parameters.
**Suggestion**:
- Break into smaller methods (max 20-30 lines)
- Use parameter object pattern for multiple parameters
- Consider command pattern for complex operations

### 3. Duplicate Code
**Issue**: validate_email and validate_username have similar structure.
**Suggestion**: Extract common validation pattern:
```python
class FieldValidator:
    def validate_required(self, value, field_name):
        if not value:
            raise ValidationError(f"{field_name} is required")

    def validate_length(self, value, min_len, max_len, field_name):
        if len(value) < min_len or len(value) > max_len:
            raise ValidationError(f"{field_name} length must be between {min_len} and {max_len}")
```

### 4. Feature Envy
**Issue**: Methods accessing data more than using it.
**Suggestion**: Move methods closer to the data they use."""

        analysis_tools.llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        with patch.object(
            analysis_tools.code_extractor,
            "get_file_content",
            return_value=mock_content,
        ):
            mock_db_session.execute.side_effect = [
                file_result,
                classes_result,
                functions_result,
            ]

            result = await analysis_tools.suggest_refactoring(
                "/src/models/user.py", focus="code_smells"
            )

        assert "God Object" in result["refactoring_suggestions"]
        assert "Long Method" in result["refactoring_suggestions"]
        assert "Duplicate Code" in result["refactoring_suggestions"]
        assert result["code_metrics"]["total_classes"] == 1
        assert result["code_metrics"]["total_functions"] == 3

    @pytest.mark.asyncio
    async def test_suggest_refactoring_performance_focus(
        self, analysis_tools, mock_db_session
    ):
        """Test suggesting performance-focused refactoring."""
        # Mock file
        mock_file = MagicMock(spec=File)
        mock_file.id = 1
        mock_file.path = "/src/data/processor.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock functions with performance issues
        mock_function = MagicMock(spec=Function)
        mock_function.name = "find_duplicates"
        mock_function.complexity_score = 15
        mock_function.calls_count = 1000  # Called frequently

        functions_result = MagicMock()
        functions_result.scalars.return_value.all.return_value = [mock_function]

        # Mock content with performance issues
        mock_content = '''def find_duplicates(items):
    """Find duplicate items in list - O(n²) complexity."""
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates

def process_large_file(filename):
    """Read entire file into memory."""
    with open(filename, 'r') as f:
        data = f.read()  # Loads entire file
        lines = data.split('\\n')
        return [process_line(line) for line in lines]
'''

        # Mock LLM response for performance
        mock_llm_response = MagicMock()
        mock_llm_response.content = """## Performance Optimization Suggestions

### 1. Algorithm Optimization
**Issue**: find_duplicates has O(n²) time complexity.
**Suggestion**: Use set for O(n) complexity:
```python
def find_duplicates(items):
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)
```

### 2. Memory Optimization
**Issue**: process_large_file loads entire file into memory.
**Suggestion**: Use generator for streaming:
```python
def process_large_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield process_line(line.strip())
```

### 3. Caching Opportunity
**Issue**: Function called 1000 times, possibly with repeated inputs.
**Suggestion**: Add memoization:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def find_duplicates(items):
    # Convert to tuple for hashability
    return _find_duplicates_impl(tuple(items))
```"""

        analysis_tools.llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        with patch.object(
            analysis_tools.code_extractor,
            "get_file_content",
            return_value=mock_content,
        ):
            mock_db_session.execute.side_effect = [file_result, functions_result]

            result = await analysis_tools.suggest_refactoring(
                "/src/data/processor.py", focus="performance"
            )

        assert "Algorithm Optimization" in result["refactoring_suggestions"]
        assert "Memory Optimization" in result["refactoring_suggestions"]
        assert "O(n²)" in result["refactoring_suggestions"]
        assert "generator" in result["refactoring_suggestions"]

    @pytest.mark.asyncio
    async def test_suggest_refactoring_readability_focus(
        self, analysis_tools, mock_db_session
    ):
        """Test suggesting readability-focused refactoring."""
        # Mock file
        mock_file = MagicMock(spec=File)
        mock_file.id = 1
        mock_file.path = "/src/utils/helpers.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # Mock poorly named functions
        functions = []
        for name, complexity in [
            ("calc", 10),  # Poor name
            ("process_data_2", 8),  # Numbered function
            ("do_stuff", 12),  # Vague name
        ]:
            func = MagicMock(spec=Function)
            func.name = name
            func.complexity_score = complexity
            func.docstring = None  # No documentation
            functions.append(func)

        functions_result = MagicMock()
        functions_result.scalars.return_value.all.return_value = functions

        # Mock content with readability issues
        mock_content = """def calc(x, y, z):
    return x * y + z / 2 if x > 0 else y - z

def process_data_2(d):
    r = []
    for i in d:
        if i[0] > i[1]:
            r.append(i[0])
        else:
            r.append(i[1])
    return r

def do_stuff(lst, flg=True):
    if flg:
        return [x for x in lst if x % 2 == 0]
    else:
        return [x for x in lst if x % 2 != 0]
"""

        # Mock LLM response for readability
        mock_llm_response = MagicMock()
        mock_llm_response.content = """## Readability Improvements

### 1. Descriptive Naming
**Issue**: Functions have vague or abbreviated names.
**Suggestions**:
- `calc` → `calculate_weighted_sum`
- `process_data_2` → `extract_maximum_values`
- `do_stuff` → `filter_by_parity`

### 2. Add Documentation
**Issue**: No docstrings or comments explaining logic.
**Suggestion**: Add clear docstrings:
```python
def calculate_weighted_sum(base: float, multiplier: float, offset: float) -> float:
    \"\"\"Calculate weighted sum with conditional logic.

    Returns base * multiplier + offset/2 if base is positive,
    otherwise returns multiplier - offset.
    \"\"\"
```

### 3. Simplify Complex Expressions
**Issue**: Nested ternary operators and complex comprehensions.
**Suggestion**: Break into clear steps:
```python
def filter_by_parity(numbers: List[int], keep_even: bool = True) -> List[int]:
    if keep_even:
        return [n for n in numbers if n % 2 == 0]
    return [n for n in numbers if n % 2 != 0]
```

### 4. Use Type Hints
**Issue**: No type annotations make it hard to understand data flow.
**Suggestion**: Add comprehensive type hints for clarity."""

        analysis_tools.llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        with patch.object(
            analysis_tools.code_extractor,
            "get_file_content",
            return_value=mock_content,
        ):
            mock_db_session.execute.side_effect = [file_result, functions_result]

            result = await analysis_tools.suggest_refactoring(
                "/src/utils/helpers.py", focus="readability"
            )

        assert "Descriptive Naming" in result["refactoring_suggestions"]
        assert "Add Documentation" in result["refactoring_suggestions"]
        assert "Type Hints" in result["refactoring_suggestions"]
        assert result["code_metrics"]["functions_without_docstrings"] == 3

    @pytest.mark.asyncio
    async def test_suggest_refactoring_no_functions(
        self, analysis_tools, mock_db_session
    ):
        """Test suggesting refactoring for file with no functions."""
        # Mock file
        mock_file = MagicMock(spec=File)
        mock_file.id = 1
        mock_file.path = "/src/constants.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        # No functions or classes
        functions_result = MagicMock()
        functions_result.scalars.return_value.all.return_value = []

        classes_result = MagicMock()
        classes_result.scalars.return_value.all.return_value = []

        # Mock content with only constants
        mock_content = '''"""Application constants."""

API_VERSION = "v1"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30
DATABASE_URL = "postgresql://localhost/myapp"
SECRET_KEY = "hardcoded-secret-key"  # Security issue!
'''

        # Mock LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.content = """## Configuration Refactoring

### 1. Security Issue
**Issue**: Hardcoded secret key in code.
**Suggestion**: Use environment variables:
```python
import os
SECRET_KEY = os.environ.get('SECRET_KEY', 'default-dev-key')
```

### 2. Configuration Management
**Issue**: Constants scattered without organization.
**Suggestion**: Use configuration classes:
```python
from dataclasses import dataclass

@dataclass
class APIConfig:
    version: str = "v1"
    max_retries: int = 3
    timeout_seconds: int = 30
```"""

        analysis_tools.llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        with patch.object(
            analysis_tools.code_extractor,
            "get_file_content",
            return_value=mock_content,
        ):
            mock_db_session.execute.side_effect = [
                file_result,
                classes_result,
                functions_result,
            ]

            result = await analysis_tools.suggest_refactoring("/src/constants.py")

        assert "Security Issue" in result["refactoring_suggestions"]
        assert "environment variables" in result["refactoring_suggestions"]
        assert result["code_metrics"]["total_functions"] == 0
        assert result["code_metrics"]["total_classes"] == 0

    @pytest.mark.asyncio
    async def test_suggest_refactoring_error_handling(
        self, analysis_tools, mock_db_session
    ):
        """Test error handling in refactoring suggestions."""
        # Mock file
        mock_file = MagicMock(spec=File)
        mock_file.id = 1
        mock_file.path = "/src/error.py"

        file_result = MagicMock()
        file_result.scalar_one_or_none.return_value = mock_file

        functions_result = MagicMock()
        functions_result.scalars.return_value.all.return_value = []

        # Mock LLM error
        analysis_tools.llm.ainvoke = AsyncMock(
            side_effect=Exception("LLM service unavailable")
        )

        with patch.object(
            analysis_tools.code_extractor,
            "get_file_content",
            return_value="# Some code",
        ):
            mock_db_session.execute.side_effect = [file_result, functions_result]

            result = await analysis_tools.suggest_refactoring("/src/error.py")

        assert "error" in result
        assert "Failed to generate" in result["error"]
