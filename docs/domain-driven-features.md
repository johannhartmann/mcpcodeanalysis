# Domain-Driven Design Features

The MCP Code Analysis Server includes advanced domain-driven design (DDD) capabilities that help you understand and improve the business architecture of your codebase.

## Overview

The domain analysis features use Large Language Models (LLMs) to extract semantic meaning from code, going beyond traditional static analysis to understand:

- **Business Entities**: Core business concepts, not just technical classes
- **Bounded Contexts**: Cohesive groups of related domain concepts
- **Aggregate Boundaries**: Consistency boundaries in your domain model
- **Business Capabilities**: What your code actually does in business terms

## How It Works

### 1. LLM-Based Entity Extraction

Instead of just parsing code structure, the system uses LLMs to understand:

```python
# Traditional parsing sees: class Order
# Domain analysis understands: 
# - Aggregate root for purchase transactions
# - Maintains order lifecycle consistency
# - Enforces business rules about minimum quantities
```

### 2. Semantic Graph Building

The system builds a knowledge graph of your domain:

1. **Entities** are extracted with their business rules and responsibilities
2. **Relationships** are identified based on how entities collaborate
3. **Communities** are detected using the Leiden algorithm
4. **Bounded contexts** emerge from highly cohesive entity groups

### 3. Hierarchical Summarization

Summaries are generated at multiple levels:
- **Function level**: What business capability it provides
- **Class level**: Its role in the domain model
- **Module level**: The subdomain it represents
- **Context level**: The business capability it handles

## Using Domain Features

### Extract Domain Model

```json
{
  "tool": "extract_domain_model",
  "parameters": {
    "code_path": "src/orders/order.py",
    "include_relationships": true
  }
}
```

Returns:
- Domain entities (Order, OrderItem, Customer)
- Business rules and invariants
- Relationships between entities

### Find Aggregate Roots

```json
{
  "tool": "find_aggregate_roots",
  "parameters": {
    "context_name": "Sales"
  }
}
```

Identifies the main entities that maintain consistency boundaries.

### Analyze Bounded Contexts

```json
{
  "tool": "analyze_bounded_context",
  "parameters": {
    "context_name": "Sales"
  }
}
```

Returns:
- Entities within the context
- Ubiquitous language used
- Cohesion and coupling metrics
- External dependencies

### Domain-Enhanced Search

```json
{
  "tool": "semantic_search",
  "parameters": {
    "query": "payment processing",
    "use_domain_knowledge": true,
    "bounded_context": "Sales"
  }
}
```

Search understands business concepts and returns results weighted by domain relevance.

### Business Capability Search

```json
{
  "tool": "search_by_business_capability",
  "parameters": {
    "capability": "process customer refunds"
  }
}
```

Finds code that implements specific business capabilities, regardless of technical naming.

### DDD Refactoring Suggestions

```json
{
  "tool": "suggest_ddd_refactoring",
  "parameters": {
    "code_path": "src/services/order_service.py"
  }
}
```

Provides suggestions like:
- Missing aggregate roots
- Anemic domain models
- Entities that should be value objects
- Bloated entities that need domain services
- Context boundary violations

### Generate Context Maps

```json
{
  "tool": "generate_context_map",
  "parameters": {
    "output_format": "mermaid"
  }
}
```

Creates visual representations of bounded contexts and their relationships.

## Domain Indexing Process

### Initial Indexing

When a repository is scanned with domain analysis enabled:

1. **Code Chunking**: Files are split into semantic chunks
2. **Entity Extraction**: LLM analyzes each chunk for domain concepts
3. **Relationship Discovery**: Connections between entities are identified
4. **Graph Construction**: A semantic graph is built
5. **Community Detection**: Leiden algorithm finds bounded contexts
6. **Summary Generation**: Hierarchical summaries are created

### Incremental Updates

The system tracks changes and only re-analyzes modified code, maintaining the domain model efficiently.

## Benefits

### For Understanding Legacy Code

- Quickly grasp business concepts in unfamiliar codebases
- Identify implicit domain boundaries
- Find where business rules are implemented

### For Refactoring

- Discover hidden coupling between contexts
- Identify candidates for extracting microservices
- Find duplicate implementations of business concepts

### For Documentation

- Generate accurate domain models from code
- Create context maps automatically
- Maintain up-to-date business documentation

## Configuration

### Enabling Domain Analysis

Domain analysis can be enabled in two ways:

1. **Per Repository** - Enable for specific repositories:
```yaml
repositories:
  - url: https://github.com/your/repo
    enable_domain_analysis: true
```

2. **Globally** - Enable by default for all repositories:
```yaml
domain_analysis:
  enabled: true  # Enable for all repos by default
  chunk_size: 1000
  chunk_overlap: 200
  min_confidence: 0.7
  max_entities_per_file: 50
  leiden_resolution: 1.0
  min_context_size: 3
```

### How It Works During Indexing

When domain analysis is enabled:

1. **During File Processing**: Each Python file is analyzed by the LLM to extract domain entities
2. **Entity Extraction**: Business concepts, rules, and relationships are identified
3. **Incremental Updates**: Only new or modified files are re-analyzed
4. **After Repository Scan**: Bounded contexts are detected using graph algorithms
5. **Automatic Context Maps**: Domain boundaries emerge from the semantic graph

### Performance Considerations

- Initial indexing with domain analysis is slower (requires LLM calls)
- Expect ~2-5 seconds per file for domain extraction
- Incremental updates are much faster
- Domain data is cached in the database

## Best Practices

1. **Start with Core Domain**: Index your most important business logic first
2. **Review Extracted Entities**: The LLM extraction improves with feedback
3. **Use Business Language**: Search using business terms, not technical jargon
4. **Iterate on Boundaries**: Adjust Leiden resolution to find appropriate context sizes
5. **Combine with Traditional Analysis**: Use both structural and semantic analysis

## Limitations

- Requires OpenAI API access for LLM analysis
- Initial indexing can be slow for large codebases
- Quality depends on code clarity and documentation
- Currently supports Python only

## Future Enhancements

- Support for more programming languages
- Custom domain ontologies
- Real-time collaboration on domain models
- Integration with architecture decision records
- Event storming diagram generation