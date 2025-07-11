# Multi-Language Support

The MCP Code Analysis Server supports multiple programming languages through a configurable language registry system.

## Supported Languages

Currently, the following languages are defined in the system:

| Language | Extensions | Parser Available | Features |
|----------|------------|------------------|----------|
| Python | .py, .pyw, .pyi | ✅ | Classes, Functions, Imports, Modules, Docstrings, Type Hints |
| JavaScript | .js, .jsx, .mjs | ❌ | Classes, Functions, Imports, Modules |
| TypeScript | .ts, .tsx, .d.ts | ❌ | Classes, Functions, Imports, Modules, Type Hints |
| Java | .java | ❌ | Classes, Methods, Imports, JavaDoc, Type Hints |
| Go | .go | ❌ | Functions, Imports, Modules, Doc Comments, Type Hints |
| Rust | .rs | ❌ | Functions, Imports, Modules, Doc Comments, Type Hints |
| C++ | .cpp, .cc, .hpp, etc. | ❌ | Classes, Functions, Includes, Doxygen, Type Hints |
| C | .c, .h | ❌ | Functions, Includes, Doxygen, Type Hints |
| C# | .cs | ❌ | Classes, Methods, Using Statements, Namespaces, XML Docs, Type Hints |
| Ruby | .rb, .rake | ❌ | Classes, Methods, Requires, Modules, RDoc |
| PHP | .php | ❌ | Classes, Functions, Use/Require, Namespaces, PHPDoc, Type Hints |

## Configuration

Languages are configured in the `settings.toml` file:

```toml
[default.parser]
# Languages to parse
languages = ["python", "javascript", "typescript"]
```

Or via the `config.yaml`:

```yaml
parser:
  languages:
    - python
    - javascript
    - typescript
```

## Adding Support for a New Language

To add support for a new language, you need to:

1. **Register the language** in `src/parser/language_config.py`:
   ```python
   "newlang": LanguageConfig(
       name="newlang",
       display_name="New Language",
       extensions=[".nl", ".nlx"],
       parser_available=False,
       features={
           "classes": True,
           "functions": True,
           "imports": True,
           "modules": False,
           "docstrings": True,
           "type_hints": True,
       },
   )
   ```

2. **Create a parser** implementing the base parser interface:
   ```python
   # src/parser/newlang_parser.py
   from src.parser.base_parser import BaseParser
   
   class NewLangParser(BaseParser):
       def parse_file(self, file_path: Path) -> dict[str, Any]:
           # Implement parsing logic
           pass
   ```

3. **Register the parser** in `src/parser/parser_factory.py`:
   ```python
   _parsers[".nl"] = NewLangParser
   _parsers[".nlx"] = NewLangParser
   _language_parsers["newlang"] = NewLangParser
   ```

4. **Update the language registry** to mark the parser as available:
   ```python
   # In language_loader.py
   if lang_name == "newlang":
       lang_config.parser_available = True
   ```

## Architecture

The multi-language support system consists of:

- **LanguageConfig**: Defines language properties and capabilities
- **LanguageRegistry**: Central registry of all supported languages
- **ParserFactory**: Creates appropriate parsers based on file extension or language
- **LanguageLoader**: Loads configuration and initializes enabled languages

## Future Enhancements

- TreeSitter grammars for all major languages
- Language-specific entity extraction (e.g., interfaces in TypeScript, traits in Rust)
- Cross-language dependency analysis
- Language-specific documentation extraction
- IDE-style language services integration