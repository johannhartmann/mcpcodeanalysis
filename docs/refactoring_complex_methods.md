# Refactoring Complex Methods - PLR0912 Issues

## Overview

We have 4 remaining ruff PLR0912 issues (methods with too many branches). These methods have between 14-18 branches, exceeding the recommended limit of 12.

## Current Issues

1. `src/parser/treesitter_parser.py:625` - PHPTreeSitterParser.extract_classes (16 branches)
2. `src/parser/treesitter_parser.py:985` - JavaTreeSitterParser.extract_functions (18 branches)
3. `src/parser/treesitter_parser.py:1094` - JavaTreeSitterParser.extract_classes (15 branches)
4. `src/scanner/code_processor.py:608` - CodeProcessor._resolve_references (14 branches)

## Refactoring Strategies

### 1. PHPTreeSitterParser.extract_classes (16 branches)

Current structure has many if/elif checks for child node types. Refactor using:

```python
class PHPTreeSitterParser:
    def extract_classes(self, tree, content):
        """Extract class definitions."""
        classes = []
        class_nodes = self.find_nodes_by_type(tree.root_node, "class_declaration")

        for node in class_nodes:
            class_data = self._create_class_data(node)
            self._process_class_node(node, content, class_data)
            classes.append(class_data)

        return classes

    def _create_class_data(self, node):
        """Create initial class data structure."""
        return {
            "name": None,
            "docstring": None,
            "base_classes": [],
            "decorators": [],
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "is_abstract": False,
            "methods": [],
            "traits": [],
        }

    def _process_class_node(self, node, content, class_data):
        """Process class node using dispatch table."""
        handlers = {
            "name": self._handle_class_name,
            "base_clause": self._handle_base_clause,
            "class_interface_clause": self._handle_interface_clause,
            "abstract_modifier": lambda n, c, d: self._set_abstract(d),
            "final_modifier": lambda n, c, d: None,  # No-op for final
            "declaration_list": self._handle_declaration_list,
        }

        for child in node.children:
            handler = handlers.get(child.type)
            if handler:
                handler(child, content, class_data)

        # Add docstring and methods
        class_data["docstring"] = self._get_php_docstring(node, content)
        class_data["methods"] = self.extract_functions(tree, content, node)

    def _handle_class_name(self, node, content, class_data):
        """Handle class name node."""
        class_data["name"] = self.get_node_text(node, content)

    def _handle_base_clause(self, node, content, class_data):
        """Handle base class clause."""
        for child in node.children:
            if child.type == "name":
                class_data["base_classes"].append(
                    self.get_node_text(child, content)
                )

    def _handle_interface_clause(self, node, content, class_data):
        """Handle interface implementation clause."""
        for child in node.children:
            if child.type == "name":
                class_data["base_classes"].append(
                    self.get_node_text(child, content)
                )

    def _set_abstract(self, class_data):
        """Mark class as abstract."""
        class_data["is_abstract"] = True

    def _handle_declaration_list(self, node, content, class_data):
        """Handle class body declarations (extract traits)."""
        use_nodes = self.find_nodes_by_type(node, "use_declaration")
        for use_node in use_nodes:
            for child in use_node.children:
                if child.type == "name":
                    trait_name = self.get_node_text(child, content)
                    class_data["traits"].append(trait_name)
```

### 2. JavaTreeSitterParser.extract_functions (18 branches)

Split into smaller focused methods:

```python
class JavaTreeSitterParser:
    def extract_functions(self, tree, content, parent_class=None):
        """Extract method definitions."""
        functions = []
        root = parent_class if parent_class else tree.root_node

        # Get all function nodes
        function_nodes = self._collect_function_nodes(root)

        for node in function_nodes:
            func_data = self._extract_function_data(node, content)
            functions.append(func_data)

        return functions

    def _collect_function_nodes(self, root):
        """Collect all function/method nodes."""
        function_nodes = self.find_nodes_by_type(
            root, "method_declaration", max_depth=4
        )
        constructor_nodes = self.find_nodes_by_type(
            root, "constructor_declaration", max_depth=4
        )
        return function_nodes + constructor_nodes

    def _extract_function_data(self, node, content):
        """Extract all function data from a node."""
        func_data = self._create_function_data(node)

        # Extract modifiers first
        self._extract_modifiers(node, content, func_data)

        # Extract name and return type
        self._extract_signature(node, content, func_data)

        # Handle special constructor case
        if node.type == "constructor_declaration" and not func_data["name"]:
            func_data["name"] = self._resolve_constructor_name(node, content)

        # Add documentation and complexity
        func_data["docstring"] = self._get_javadoc(node, content)
        func_data["complexity"] = self.complexity_calculator.calculate_complexity(
            node, content
        )

        return func_data

    def _extract_modifiers(self, node, content, func_data):
        """Extract method modifiers and annotations."""
        for child in node.children:
            if child.type == "modifiers":
                self._process_modifiers(child, content, func_data)
                break

    def _process_modifiers(self, modifiers_node, content, func_data):
        """Process modifier nodes."""
        modifiers_text = self.get_node_text(modifiers_node, content)
        if "static" in modifiers_text:
            func_data["is_staticmethod"] = True

        for child in modifiers_node.children:
            if child.type in ("marker_annotation", "annotation"):
                annotation = self._extract_annotation(child, content)
                if annotation:
                    func_data["decorators"].append(annotation)

    def _extract_signature(self, node, content, func_data):
        """Extract method name, parameters, and return type."""
        for i, child in enumerate(node.children):
            if child.type == "identifier":
                func_data["name"] = self.get_node_text(child, content)
                # Check previous sibling for return type
                if i > 0:
                    func_data["return_type"] = self._check_return_type(
                        node.children[i - 1], content
                    )
            elif child.type == "formal_parameters" and func_data["name"]:
                func_data["parameters"] = self._extract_parameters(child, content)
```

### 3. Suppressing the Issues (Alternative)

If refactoring is too risky or time-consuming, you can suppress these specific issues:

```python
# Add to each complex method:
def extract_classes(  # noqa: PLR0912
    self,
    tree: tree_sitter.Tree,
    content: bytes,
) -> list[dict[str, Any]]:
    """Extract class definitions."""
    # ... existing code ...
```

Or configure ruff to allow higher complexity for these specific files:

```toml
# In pyproject.toml
[tool.ruff.per-file-ignores]
"src/parser/treesitter_parser.py" = ["PLR0912"]
"src/scanner/code_processor.py" = ["PLR0912"]
```

## Recommendation

For this codebase, I recommend:

1. **Short term**: Add `noqa: PLR0912` comments to these 4 methods. These are inherently complex parsing methods where the branches represent different AST node types that need to be handled.

2. **Long term**: Refactor using the dispatch table pattern shown above. This would make the code more maintainable and easier to extend with new node types.

The complexity in these methods is inherent to the problem domain (parsing different AST node types). While refactoring would improve maintainability, the current implementation is functional and the complexity is well-contained within these specific parsing methods.

## Implementation Priority

1. Start with `_resolve_references` in `code_processor.py` - it's the least complex (14 branches) and not part of the parser hierarchy
2. Then tackle the parser methods, starting with the simpler PHP parser
3. The Java parser methods are the most complex and would benefit most from a full redesign using the strategy pattern
