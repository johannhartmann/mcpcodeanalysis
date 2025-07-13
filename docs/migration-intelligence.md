# Migration Intelligence Guide

## Overview

The MCP Code Analysis Server now includes comprehensive migration intelligence capabilities to help transform monolithic codebases into modular architectures. This guide covers how to use these features to analyze, plan, and execute migrations.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Migration Analysis](#migration-analysis)
3. [Migration Planning](#migration-planning)
4. [Risk Assessment](#risk-assessment)
5. [Resource Planning](#resource-planning)
6. [Interface Design](#interface-design)
7. [Migration Execution](#migration-execution)
8. [Monitoring & Tracking](#monitoring--tracking)
9. [Complete Workflow Example](#complete-workflow-example)
10. [Migration Strategies](#migration-strategies)
11. [Best Practices](#best-practices)

## Getting Started

### Prerequisites

Before starting a migration analysis:

1. Ensure your repository is synced with the MCP server
2. Run a full code scan with embeddings enabled
3. Allow domain analysis to complete

```bash
# Add and scan repository
add_repository(
    url="https://github.com/your-org/monolith",
    scan_immediately=True,
    generate_embeddings=True
)
```

## Migration Analysis

### Analyze Migration Readiness

The first step is to analyze your codebase for migration opportunities:

```python
result = analyze_migration_readiness(
    repository_url="https://github.com/your-org/monolith"
)
```

This returns:
- **Bounded Contexts**: Natural module boundaries discovered in your code
- **Migration Candidates**: Packages/modules suitable for extraction
- **Dependency Analysis**: Circular dependencies, coupling hotspots
- **Complexity Metrics**: Overall codebase complexity assessment
- **Recommended Strategy**: Suggested migration approach

Example output:
```json
{
  "analysis_summary": {
    "bounded_contexts_found": 5,
    "migration_candidates": 23,
    "complexity_rating": "medium",
    "recommended_strategy": "strangler_fig"
  },
  "top_contexts": [
    {
      "name": "BillingContext",
      "migration_readiness": 0.85,
      "cohesion_score": 0.78,
      "coupling_score": 0.32
    }
  ],
  "dependency_issues": {
    "circular_dependencies": 3,
    "high_coupling_packages": 7,
    "bottlenecks": 2
  }
}
```

### Identify Migration Patterns

Find proven patterns applicable to your codebase:

```python
patterns = identify_migration_patterns(
    repository_url="https://github.com/your-org/monolith"
)
```

Returns patterns from the library that match your codebase characteristics.

## Migration Planning

### Create a Migration Plan

Based on the analysis, create a detailed migration plan:

```python
plan = create_migration_plan(
    repository_url="https://github.com/your-org/monolith",
    plan_name="Modular Monolith Transformation",
    strategy="strangler_fig",  # or: gradual, branch_by_abstraction, parallel_run
    target_architecture="modular_monolith",  # or: microservices, event_driven
    team_size=5,
    timeline_weeks=16,  # Optional: desired timeline
    risk_tolerance="medium"  # low, medium, high
)
```

The plan includes:
- Step-by-step migration tasks
- Dependencies between steps
- Effort estimates
- Risk assessments
- Success metrics

### Generate Migration Roadmap

Visualize the migration plan:

```python
roadmap = generate_migration_roadmap(plan_id=plan["plan_id"])
```

Returns:
- Phase breakdown with timelines
- Critical path identification
- Resource allocation by phase
- Milestone tracking
- Risk mitigation schedule

### Optimize Migration Plan

Optimize an existing plan based on constraints:

```python
optimized = optimize_migration_plan(
    plan_id=plan["plan_id"],
    minimize_time=True,      # Reduce timeline
    minimize_risk=False,     # Add validation steps
    maximize_quality=True    # Increase test coverage
)
```

## Risk Assessment

### Assess Migration Risks

Identify and assess risks:

```python
risks = assess_migration_risks(
    repository_url="https://github.com/your-org/monolith",
    plan_id=plan["plan_id"]  # Optional: assess specific plan
)
```

Risk categories:
- **Technical**: Code complexity, dependencies, technical debt
- **Operational**: Team skills, coordination, timeline
- **Business**: Downtime, performance, functionality

Each risk includes:
- Probability (0-1)
- Impact (0-1)
- Risk level (low/medium/high/critical)
- Mitigation strategy

### Analyze Migration Impact

Analyze the impact of migrating specific modules:

```python
impact = analyze_migration_impact(
    repository_url="https://github.com/your-org/monolith",
    module_path="src/billing",
    impact_type="extraction"  # or: refactoring, interface_change
)
```

Returns:
- Affected modules (dependents and dependencies)
- Required interface changes
- Testing requirements
- Estimated effort

## Resource Planning

### Estimate Migration Effort

Get detailed effort estimates:

```python
effort = estimate_migration_effort(
    plan_id=plan["plan_id"],
    include_testing=True,
    include_documentation=True,
    productivity_factor=0.8  # Team productivity (0.5-1.5)
)
```

Returns:
- Total effort breakdown (dev, test, docs, overhead)
- Phase-by-phase estimates
- Risk buffer calculations
- Assumptions and constraints

### Plan Migration Resources

Create resource allocation plan:

```python
resources = plan_migration_resources(
    plan_id=plan["plan_id"],
    available_developers=5,
    available_architects=1,
    available_qa=2,
    sprint_weeks=2
)
```

Returns:
- Sprint-by-sprint allocation
- Skill requirements
- Utilization metrics
- Resource risks

## Interface Design

### Design Module Interface

Create clean interfaces for modules:

```python
interface = design_module_interface(
    repository_url="https://github.com/your-org/monolith",
    package_path="src/billing",
    target_architecture="modular_monolith",
    include_events=True  # Include domain events
)
```

Returns:
- Public API specification
- Data contracts (DTOs)
- Domain events
- Dependency interfaces
- Implementation guidelines

### Generate Interface Documentation

Create comprehensive documentation:

```python
docs = generate_interface_documentation(
    interface_design=interface
)
```

Returns markdown documentation ready for publishing.

## Migration Execution

### Start Migration Step

Begin executing a specific migration step:

```python
execution = start_migration_step(
    step_id=123,
    executor_id="john.doe@company.com"  # Optional
)
```

Returns:
- Execution ID for tracking
- Step details and dependencies
- Estimated duration

### Complete Migration Step

Mark a step as completed:

```python
result = complete_migration_step(
    step_id=123,
    success=True,
    notes="Successfully extracted billing module",
    validation_results={
        "type": "automated",
        "passed": True,
        "results": {
            "tests_passed": 150,
            "coverage": "95%",
            "performance": "baseline"
        }
    }
)
```

### Validate Migration Step

Perform validation on completed steps:

```python
validation = validate_migration_step(
    step_id=123,
    validation_type="automated",  # or: manual, performance
    validation_criteria={
        "test_coverage_required": 90,
        "performance_baseline": {
            "response_time_ms": 100,
            "throughput_rps": 1000
        }
    }
)
```

### Rollback Migration Step

Rollback a failed or problematic step:

```python
rollback = rollback_migration_step(
    step_id=123,
    reason="Performance regression detected in production"
)
```

## Monitoring & Tracking

### Migration Dashboard

Get real-time dashboard view:

```python
dashboard = get_migration_dashboard(
    repository_url="https://github.com/acme/monolith"  # Optional filter
)
```

Returns:
- Summary metrics (active/completed/failed plans)
- Recent activity timeline
- Active alerts requiring attention
- Risk overview
- Performance metrics

### Track Migration Progress

Track detailed progress of a plan:

```python
progress = track_migration_progress(plan_id=plan["plan_id"])
```

Returns:
- Completion percentage
- Time tracking (estimated vs actual)
- Current phase
- Blockers and issues
- Health score

### Monitor Step Execution

Real-time monitoring of executing steps:

```python
status = monitor_step_execution(step_id=123)
```

Returns:
- Live progress percentage
- Elapsed time vs estimate
- Recent logs
- Detected issues

### Migration Timeline

Visualize migration timeline:

```python
timeline = get_migration_timeline(plan_id=plan["plan_id"])
```

Returns:
- Chronological event list
- Critical path identification
- Estimated completion date
- Phase transitions

### Health Checks

Comprehensive health assessment:

```python
health = check_migration_health(plan_id=plan["plan_id"])
```

Returns:
- Overall health score (0-100)
- Category scores (schedule, resources, risks, validation, dependencies)
- Identified issues with severity
- Recommendations

### Anomaly Detection

Detect execution anomalies:

```python
anomalies = detect_anomalies(plan_id=plan["plan_id"])
```

Detects:
- Repeated failures
- Excessive time overruns
- Validation failures after success
- Stalled progress

### Status Reports

Generate different types of reports:

```python
# Executive summary
exec_report = generate_status_report(
    plan_id=plan["plan_id"],
    report_type="executive"
)

# Detailed technical report
detailed_report = generate_status_report(
    plan_id=plan["plan_id"],
    report_type="detailed"
)

# Standard summary
summary_report = generate_status_report(
    plan_id=plan["plan_id"],
    report_type="summary"
)
```

### Execution Report

Generate comprehensive execution history:

```python
exec_report = generate_execution_report(plan_id=plan["plan_id"])
```

Returns:
- All execution attempts
- Success/failure rates
- Validation results
- Phase-wise breakdown
- Issues encountered

## Complete Workflow Example

Here's a complete migration workflow:

```python
# 1. Initial Analysis
print("🔍 Analyzing codebase...")
analysis = analyze_migration_readiness(
    repository_url="https://github.com/acme/monolith"
)

print(f"Found {analysis['analysis_summary']['bounded_contexts_found']} bounded contexts")
print(f"Identified {analysis['analysis_summary']['migration_candidates']} migration candidates")
print(f"Recommended strategy: {analysis['analysis_summary']['recommended_strategy']}")

# 2. Create Migration Plan
print("\n📋 Creating migration plan...")
plan = create_migration_plan(
    repository_url="https://github.com/acme/monolith",
    plan_name="ACME Modular Migration",
    strategy=analysis['analysis_summary']['recommended_strategy'],
    target_architecture="modular_monolith",
    team_size=6,
    risk_tolerance="medium"
)

print(f"Created plan with {plan['summary']['total_steps']} steps")
print(f"Estimated timeline: {plan['summary']['timeline_weeks']} weeks")
print(f"Confidence level: {plan['summary']['confidence_level']}")

# 3. Assess Risks
print("\n⚠️ Assessing risks...")
risks = assess_migration_risks(
    repository_url="https://github.com/acme/monolith",
    plan_id=plan['plan_id']
)

print(f"Identified {risks['risk_summary']['total_risks']} risks")
print(f"Critical risks: {risks['risk_summary']['critical_risks']}")
print(f"High risks: {risks['risk_summary']['high_risks']}")

# 4. Plan Resources
print("\n👥 Planning resources...")
resources = plan_migration_resources(
    plan_id=plan['plan_id'],
    available_developers=6,
    available_architects=2,
    available_qa=2,
    sprint_weeks=2
)

print(f"Sprints needed: {resources['resource_summary']['sprints_needed']}")
print(f"Total duration: {resources['resource_summary']['total_weeks']} weeks")
print(f"Team utilization: {resources['resource_summary']['utilization_percent']}")

# 5. Design First Module Interface
print("\n🏗️ Designing module interfaces...")
first_module = plan['top_candidates'][0]
interface = design_module_interface(
    repository_url="https://github.com/acme/monolith",
    package_path=first_module['path'],
    target_architecture="modular_monolith"
)

print(f"Designed interface for {first_module['name']}")
print(f"Public API methods: {len(interface['public_api']['services'])}")
print(f"Data contracts: {len(interface['data_contracts'])}")

# 6. Generate Documentation
print("\n📚 Generating documentation...")
docs = generate_interface_documentation(interface_design=interface)
print("Documentation generated successfully")

# 7. Generate Roadmap
print("\n🗺️ Generating migration roadmap...")
roadmap = generate_migration_roadmap(plan_id=plan['plan_id'])

for phase in roadmap['phases']:
    print(f"\nPhase: {phase['name']}")
    print(f"  Duration: {phase['duration_weeks']} weeks")
    print(f"  Activities: {', '.join(phase['key_activities'][:3])}")

# 8. Start Migration Execution
print("\n🚀 Starting migration execution...")
first_step = plan['steps'][0]  # Assuming steps are returned in order

# Start the first step
execution = start_migration_step(
    step_id=first_step['id'],
    executor_id="migration-team"
)
print(f"Started step: {first_step['name']}")
print(f"Execution ID: {execution['execution_id']}")

# 9. Monitor Progress
print("\n📊 Monitoring progress...")
progress = track_migration_progress(plan_id=plan['plan_id'])
print(f"Overall progress: {progress['progress_summary']['completion_percentage']:.1f}%")
print(f"Health score: {progress['health_score']:.1f}")

# Check step execution status
step_status = monitor_step_execution(step_id=first_step['id'])
print(f"Step progress: {step_status['progress_percentage']:.1f}%")

# 10. Complete Step with Validation
print("\n✅ Completing step with validation...")
completion = complete_migration_step(
    step_id=first_step['id'],
    success=True,
    notes="Module successfully extracted and tested",
    validation_results={
        "type": "automated",
        "passed": True,
        "results": {
            "tests_passed": 42,
            "coverage": "89%"
        }
    }
)
print(f"Step completed in {completion['duration_hours']:.1f} hours")

# 11. Get Dashboard Overview
print("\n📈 Getting migration dashboard...")
dashboard = get_migration_dashboard()
print(f"Active migrations: {dashboard['summary']['active_plans']}")
print(f"Recent alerts: {len(dashboard['alerts'])}")

# 12. Generate Status Report
print("\n📄 Generating status report...")
report = generate_status_report(
    plan_id=plan['plan_id'],
    report_type="summary"
)
print(f"Status: {report['status']}")
print(f"Next steps: {', '.join(report['next_steps'][:2])}")
```

## Migration Strategies

### Strangler Fig Pattern
- **Best for**: Large, complex monoliths
- **Approach**: Gradually replace functionality behind a proxy/facade
- **Benefits**: Low risk, can be paused/resumed
- **Timeline**: Longer but safer

### Gradual Migration
- **Best for**: Well-structured monoliths
- **Approach**: Extract modules one by one
- **Benefits**: Straightforward, easy to understand
- **Timeline**: Medium

### Branch by Abstraction
- **Best for**: When you need parallel development
- **Approach**: Create abstraction layer first, then migrate
- **Benefits**: Allows old/new to coexist
- **Timeline**: Medium to long

### Parallel Run
- **Best for**: Mission-critical systems
- **Approach**: Run old and new systems in parallel
- **Benefits**: Can verify correctness before switching
- **Timeline**: Long, higher cost

## Best Practices

### 1. Start with Analysis
- Always run `analyze_migration_readiness` first
- Review bounded contexts and their relationships
- Identify high-value, low-risk candidates

### 2. Incremental Approach
- Start with peripheral modules
- Build confidence with early wins
- Learn and adapt as you go

### 3. Test Coverage First
- Ensure comprehensive tests before extraction
- Use contract tests between modules
- Maintain performance benchmarks

### 4. Clear Interfaces
- Design interfaces before extraction
- Use DTOs for data transfer
- Version interfaces from the start

### 5. Monitor Progress
- Track quality metrics throughout
- Review decisions and outcomes
- Update patterns library with learnings

### 6. Team Alignment
- Involve all stakeholders in planning
- Regular architecture reviews
- Knowledge sharing sessions

### 7. Risk Management
- Address critical risks first
- Have rollback procedures ready
- Maintain migration decision log

## Troubleshooting

### Common Issues

**High Coupling Detected**
- Use `analyze_migration_impact` to understand dependencies
- Consider introducing interfaces first
- May need to refactor before extraction

**Circular Dependencies**
- Identify cycles with dependency analysis
- Break cycles through interface abstraction
- Consider merging tightly coupled modules

**Low Migration Readiness**
- Improve test coverage first
- Refactor to improve cohesion
- Consider different module boundaries

**Resource Constraints**
- Use `optimize_migration_plan` to extend timeline
- Focus on highest value modules
- Consider external help for peak phases

## Next Steps

1. Run initial analysis on your monolith
2. Review the recommended strategy
3. Create a migration plan
4. Get stakeholder buy-in with the roadmap
5. Start with a pilot module
6. Iterate and improve

For more information, see:
- [Architecture Patterns](./architecture-patterns.md)
- [Testing Strategies](./testing-strategies.md)
- [Performance Monitoring](./performance-monitoring.md)
