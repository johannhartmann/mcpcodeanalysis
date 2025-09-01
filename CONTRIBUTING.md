# Contributing to MCP Code Analysis Server

Thank you for your interest in contributing to the MCP Code Analysis Server! We welcome contributions from the community and are grateful for any help you can provide.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in our [issue tracker](https://github.com/johannhartmann/mcp-code-analysis-server/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Detailed description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)

### Suggesting Features

1. Check existing issues and discussions for similar ideas
2. Open a new issue with the "enhancement" label
3. Describe the feature and its use case
4. Explain why it would benefit the project

### Contributing Code

#### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mcp-code-analysis-server.git
   cd mcp-code-analysis-server
   ```

3. Set up development environment:
   ```bash
   nix develop  # If using Nix
   uv sync --all-extras
   uv run pre-commit install
   ```

4. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Development Workflow

1. **Write tests first** - We follow TDD practices
2. **Make your changes** - Follow existing code style
3. **Run quality checks**:
   ```bash
   make qa  # Runs all checks
   ```

4. **Commit your changes**:
   - Use conventional commits format
   - Write clear, descriptive commit messages
   - Examples:
     - `feat: add support for JavaScript parsing`
     - `fix: handle empty docstrings in parser`
     - `docs: update installation instructions`

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

#### Pull Request Guidelines

1. **PR Title**: Use conventional commit format
2. **Description**: Include:
   - What changes were made and why
   - Related issue numbers
   - Screenshots if applicable
   - Breaking changes if any

3. **Requirements**:
   - All tests must pass
   - Code coverage must not decrease
   - All quality checks must pass
   - Documentation updated if needed

### Code Style Guidelines

#### Python Code Style

- Follow PEP 8 with Black formatting
- Use type hints for all functions
- Write docstrings in Google style:

```python
def analyze_code(file_path: Path, options: Dict[str, Any]) -> CodeAnalysis:
    """Analyze Python code and extract structure.

    Args:
        file_path: Path to the Python file to analyze
        options: Analysis options including depth and filters

    Returns:
        CodeAnalysis object containing extracted information

    Raises:
        FileNotFoundError: If file_path doesn't exist
        ParseError: If the code cannot be parsed
    """
```

#### Import Organization

```python
# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
from fastapi import FastAPI

# Local
from src.parser import CodeParser
from src.utils import logger
```

#### Error Handling

```python
# Good
try:
    result = parse_file(path)
except FileNotFoundError:
    logger.error(f"File not found: {path}")
    raise
except ParseError as e:
    logger.warning(f"Parse error in {path}: {e}")
    return None

# Bad
try:
    result = parse_file(path)
except:  # Never use bare except
    pass  # Never silently ignore errors
```

### Testing Guidelines

#### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestCodeParser:
    """Tests for CodeParser class."""

    def test_parse_simple_function(self):
        """Test parsing a simple function definition."""
        # Arrange
        code = "def hello(): pass"
        parser = CodeParser()

        # Act
        result = parser.parse(code)

        # Assert
        assert len(result.functions) == 1
        assert result.functions[0].name == "hello"

    @pytest.mark.parametrize("code,expected", [
        ("class A: pass", "A"),
        ("class B(): pass", "B"),
        ("class C(Base): pass", "C"),
    ])
    def test_parse_class_definitions(self, code, expected):
        """Test parsing various class definitions."""
        parser = CodeParser()
        result = parser.parse(code)
        assert result.classes[0].name == expected
```

#### Test Categories

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

Mark tests appropriately:
```python
@pytest.mark.unit
def test_function():
    pass

@pytest.mark.integration
@pytest.mark.requires_db
def test_database_integration():
    pass
```

### Documentation

#### Code Documentation

- All public APIs must have docstrings
- Include examples in docstrings for complex functions
- Keep documentation up-to-date with code changes

#### Project Documentation

- Update README.md for user-facing changes
- Update ARCHITECTURE.md for design changes
- Add entries to CHANGELOG.md

### Release Process

1. Ensure all tests pass
2. Update version in `pyproject.toml`
3. Update CHANGELOG.md
4. Create PR with title "Release vX.Y.Z"
5. After merge, tag the release

## Getting Help

- 💬 [GitHub Discussions](https://github.com/johannhartmann/mcp-code-analysis-server/discussions)
- 📧 Email: johann-peter.hartmann@mayflower.de
- 🐛 [Issue Tracker](https://github.com/johannhartmann/mcp-code-analysis-server/issues)

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- Annual contributor spotlight

Thank you for contributing to MCP Code Analysis Server! 🚀
