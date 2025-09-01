# Domain Analysis Tools - Implementation Summary

## What Was Implemented

### 1. Pattern Analyzer Module (`src/domain/pattern_analyzer.py`)

A comprehensive analysis engine that provides value beyond basic domain extraction:

#### Key Features:
- **Cross-Context Coupling Analysis**: Measures and analyzes dependencies between bounded contexts
- **Anti-Pattern Detection**: Identifies common DDD anti-patterns in the codebase
- **Context Split Suggestions**: Uses graph algorithms to suggest how to split large contexts
- **Evolution Analysis**: Tracks how the domain model changes over time

#### Implementation Details:
- Uses SQL queries to analyze relationships stored during indexing
- Implements graph algorithms for community detection
- Provides actionable recommendations based on patterns found
- Calculates metrics like cohesion scores and coupling factors

### 2. MCP Tool Wrappers (`src/mcp_server/tools/analysis_tools.py`)

Exposes the pattern analyzer functionality through MCP tools:

#### New Tools:
1. **`analyze_coupling`**: Provides detailed coupling analysis with metrics
2. **`detect_anti_patterns`**: Finds 6 types of DDD anti-patterns
3. **`suggest_context_splits`**: Suggests how to break up large contexts
4. **`analyze_domain_evolution`**: Shows domain changes over time
5. **`get_domain_metrics`**: Provides overall domain health score

### 3. Integration Points

- Added pattern analyzer to domain module exports
- Registered analysis tools in the MCP server
- Created comprehensive documentation
- Added test coverage for key functionality

## How It Works

### During Indexing:
1. Domain entities and relationships are extracted by LLM
2. Data is stored in PostgreSQL with relationships intact
3. Bounded contexts are detected using Leiden algorithm

### During Analysis (Value-Add Tools):
1. **Coupling Analysis**: Queries relationship data to find inter-context dependencies
2. **Anti-Pattern Detection**: Analyzes entity metadata for problematic patterns
3. **Split Suggestions**: Uses graph algorithms on existing data
4. **Evolution Tracking**: Compares timestamps to show changes

## Key Benefits

1. **Beyond Indexing**: These tools provide insights that aren't captured during indexing
2. **Actionable Results**: Each tool provides specific recommendations
3. **Health Metrics**: Overall scoring helps track improvement
4. **Pattern Recognition**: Identifies issues that are hard to spot manually

## Example Usage

```bash
# Get overall domain health
mcp call get_domain_metrics

# Find architectural issues
mcp call detect_anti_patterns

# Analyze coupling problems
mcp call analyze_coupling

# Plan refactoring
mcp call suggest_context_splits --min_entities 20
```

## Architecture Decision

The analysis tools are separate from indexing because:
1. They need the full graph to provide meaningful insights
2. They're computationally different (analysis vs extraction)
3. They can be run on-demand without re-indexing
4. They provide cross-cutting concerns across the entire codebase

This approach allows users to:
- Index once, analyze many times
- Run different analyses for different purposes
- Track improvements over time
- Focus on specific areas of concern
