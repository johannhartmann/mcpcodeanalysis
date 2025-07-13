"""Language plugins for multi-language support."""

from src.parser.plugins.java_plugin import JavaLanguagePlugin
from src.parser.plugins.javascript_plugin import JavaScriptLanguagePlugin
from src.parser.plugins.php_plugin import PHPLanguagePlugin
from src.parser.plugins.python_plugin import PythonLanguagePlugin
from src.parser.plugins.typescript_plugin import TypeScriptLanguagePlugin

__all__ = [
    "JavaLanguagePlugin",
    "JavaScriptLanguagePlugin",
    "PHPLanguagePlugin",
    "PythonLanguagePlugin",
    "TypeScriptLanguagePlugin",
]
