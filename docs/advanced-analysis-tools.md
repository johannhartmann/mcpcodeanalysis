# Advanced Domain Analysis Tools

The MCP Code Analysis Server provides advanced analysis tools that go beyond basic domain extraction to provide actionable insights about your codebase's architecture and health.

## Overview

These tools analyze the relationships and patterns in your domain model to:
- Detect architectural anti-patterns
- Measure coupling between bounded contexts
- Suggest improvements to your domain boundaries
- Track how your domain model evolves over time

## Available Tools

### 1. Analyze Coupling

Analyzes coupling between bounded contexts to identify architectural issues.

```json
{
  "tool": "analyze_coupling",
  "parameters": {
    "repository_id": 123  // Optional, analyzes all if not specified
  }
}
```

**Returns:**
- Context coupling scores and metrics
- High coupling pairs that need attention
- Specific recommendations for reducing coupling
- Distribution of coupling levels across contexts

**Example Output:**
```json
{
  "contexts": [
    {
      "name": "Sales",
      "coupling_score": 3.5,
      "outgoing_dependencies": 42,
      "coupled_with": 3
    }
  ],
  "high_coupling_pairs": [
    {
      "source": "Sales",
      "target": "Inventory",
      "relationship_count": 15,
      "relationship_types": ["depends_on", "orchestrates"],
      "recommendation": "Consider using events or a saga pattern to reduce orchestration coupling"
    }
  ],
  "metrics": {
    "average_coupling": 2.3,
    "max_coupling": 5.2,
    "coupling_distribution": {
      "low": 5,
      "medium": 8,
      "high": 2
    }
  }
}
```

### 2. Detect Anti-Patterns

Identifies common Domain-Driven Design anti-patterns in your codebase.

```json
{
  "tool": "detect_anti_patterns",
  "parameters": {
    "repository_id": 123  // Optional
  }
}
```

**Detects:**
- **Anemic Domain Models**: Entities with no business logic
- **God Objects**: Entities with too many responsibilities
- **Circular Dependencies**: Cyclic relationships between entities
- **Missing Aggregate Roots**: Contexts without consistency boundaries
- **Chatty Contexts**: Excessive inter-context communication
- **Shared Kernel Abuse**: Entities shared by too many contexts

**Example Output:**
```json
{
  "anemic_domain_models": [
    {
      "entity": "Customer",
      "type": "entity",
      "issue": "No business rules or invariants defined",
      "recommendation": "Add business logic to make this a rich domain model",
      "severity": "medium"
    }
  ],
  "god_objects": [
    {
      "entity": "OrderService",
      "responsibility_count": 12,
      "issue": "Too many responsibilities",
      "recommendation": "Extract some responsibilities into domain services",
      "severity": "high"
    }
  ]
}
```

### 3. Suggest Context Splits

Analyzes large bounded contexts and suggests how to split them based on cohesion patterns.

```json
{
  "tool": "suggest_context_splits",
  "parameters": {
    "min_entities": 20,
    "max_cohesion_threshold": 0.4
  }
}
```

**Returns:**
- Contexts that are candidates for splitting
- Suggested new context boundaries
- Estimated cohesion for each suggested split
- Key entities and aggregate roots for each split

**Example Output:**
```json
[
  {
    "context": "LargeOrderingContext",
    "current_size": 45,
    "cohesion_score": 0.35,
    "suggested_splits": [
      {
        "suggested_name": "Order Context",
        "entity_count": 18,
        "aggregate_roots": ["Order", "Cart"],
        "cohesion_estimate": 0.85
      },
      {
        "suggested_name": "Payment Context",
        "entity_count": 12,
        "aggregate_roots": ["Payment", "Invoice"],
        "cohesion_estimate": 0.78
      }
    ],
    "reasoning": "Low cohesion (0.35) indicates weak relationships between entity groups"
  }
]
```

### 4. Analyze Domain Evolution

Tracks how your domain model has evolved over time to identify trends and patterns.

```json
{
  "tool": "analyze_domain_evolution",
  "parameters": {
    "repository_id": 123,
    "days": 30
  }
}
```

**Provides:**
- New entities and contexts added
- Entities that were removed or modified
- Growth rate trends
- Insights about domain development patterns

**Example Output:**
```json
{
  "time_period": "Last 30 days",
  "entity_changes": {
    "added": [
      {
        "name": "RefundPolicy",
        "type": "value_object",
        "created": "2024-01-15T10:30:00Z"
      }
    ]
  },
  "trends": {
    "entity_growth_rate": 15.5,
    "context_stability": 0.85,
    "coupling_trend": "increasing"
  },
  "insights": [
    "Rapid growth in domain entities indicates active feature development",
    "New bounded contexts suggest evolving domain understanding"
  ]
}
```

### 5. Get Domain Metrics

Provides a comprehensive health report for your domain model.

```json
{
  "tool": "get_domain_metrics",
  "parameters": {
    "repository_id": 123  // Optional
  }
}
```

**Returns:**
- Overall health score (0-100)
- Key metrics about coupling and anti-patterns
- Top issues to address
- Actionable insights for improvement

**Example Output:**
```json
{
  "health_score": 72.5,
  "metrics": {
    "average_context_coupling": 2.3,
    "anti_pattern_counts": {
      "high": 3,
      "medium": 8,
      "low": 12
    },
    "total_contexts": 12,
    "high_coupling_pairs": 2
  },
  "insights": [
    {
      "type": "high_coupling",
      "message": "High average coupling between contexts indicates potential architectural issues",
      "recommendation": "Consider introducing anti-corruption layers or event-driven communication"
    }
  ],
  "top_issues": [
    {
      "type": "missing_aggregate_roots",
      "severity": "high",
      "description": "Context has entities but no aggregate root",
      "entity": "UserManagement"
    }
  ]
}
```

## Use Cases

### 1. Architecture Review

Use these tools during architecture reviews to:
- Identify architectural debt
- Find opportunities for microservice extraction
- Validate bounded context boundaries

### 2. Refactoring Planning

Before major refactoring:
- Detect which anti-patterns to address first
- Understand coupling that needs to be broken
- Plan context splits based on cohesion analysis

### 3. Team Education

Help teams understand DDD principles by:
- Showing concrete examples of anti-patterns in their code
- Demonstrating the impact of high coupling
- Tracking improvement over time

### 4. Continuous Monitoring

Set up regular analysis to:
- Track domain health metrics over time
- Alert on new anti-patterns
- Monitor coupling trends

## Best Practices

1. **Regular Analysis**: Run these tools regularly, not just during crises
2. **Focus on High Severity**: Address high-severity issues first
3. **Track Progress**: Use evolution analysis to ensure improvements
4. **Team Discussion**: Use results as input for architecture discussions
5. **Iterative Improvement**: Don't try to fix everything at once

## Integration with CI/CD

These tools can be integrated into your CI/CD pipeline:

```bash
# Example GitHub Action
- name: Analyze Domain Health
  run: |
    mcp call get_domain_metrics --repository_id $REPO_ID
    
- name: Check Anti-Patterns
  run: |
    mcp call detect_anti_patterns --repository_id $REPO_ID
```

## Limitations

- Analysis quality depends on having domain entities already extracted
- Suggestions are heuristic-based and should be reviewed by architects
- Some patterns may be intentional architectural decisions
- Currently supports only Python codebases