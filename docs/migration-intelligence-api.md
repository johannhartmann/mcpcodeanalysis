# Migration Intelligence API Reference

## Overview

The Migration Intelligence feature provides comprehensive tools for analyzing, planning, executing, and monitoring code migrations from monolithic architectures to modular or microservice architectures.

## API Categories

### 1. Migration Analysis Tools

#### `analyze_migration_readiness`
Analyzes a repository to assess its readiness for migration.

**Input:**
- `repository_url` (str): Repository to analyze

**Output:**
- `bounded_contexts`: Discovered domain boundaries
- `migration_candidates`: Modules suitable for extraction
- `dependency_analysis`: Circular dependencies and coupling metrics
- `complexity_metrics`: Code complexity assessment
- `recommended_strategy`: Suggested migration approach
- `readiness_score`: Overall readiness (0-1)

---

#### `identify_migration_patterns`
Identifies applicable migration patterns from the pattern library.

**Input:**
- `repository_url` (str): Repository to analyze

**Output:**
- List of applicable patterns with confidence scores

---

#### `assess_migration_risks`
Performs comprehensive risk assessment for migration.

**Input:**
- `repository_url` (str): Repository to assess
- `plan_id` (int, optional): Specific plan to assess

**Output:**
- List of risks with:
  - `type`: Risk category (technical/operational/business)
  - `probability`: Likelihood (0-1)
  - `impact`: Severity (0-1)
  - `level`: Risk level (low/medium/high/critical)
  - `mitigation`: Suggested mitigation strategy

---

#### `analyze_migration_impact`
Analyzes the impact of migrating specific modules.

**Input:**
- `repository_url` (str): Repository URL
- `module_path` (str): Module to analyze
- `impact_type` (str): Type of impact (extraction/refactoring/interface_change)

**Output:**
- `affected_modules`: List of dependent modules
- `required_changes`: Necessary code changes
- `testing_requirements`: Required tests
- `estimated_effort`: Effort estimation

### 2. Migration Planning Tools

#### `create_migration_plan`
Creates a comprehensive migration plan.

**Input:**
- `repository_url` (str): Repository to migrate
- `plan_name` (str): Name for the plan
- `strategy` (str): Migration strategy (strangler_fig/gradual/branch_by_abstraction/parallel_run)
- `target_architecture` (str): Target architecture (modular_monolith/microservices/event_driven)
- `team_size` (int): Available team members
- `timeline_weeks` (int, optional): Desired timeline
- `risk_tolerance` (str, optional): Risk tolerance (low/medium/high)

**Output:**
- `plan_id`: Created plan ID
- `steps`: List of migration steps
- `timeline`: Estimated timeline
- `confidence_level`: Plan confidence

---

#### `generate_migration_roadmap`
Generates visual roadmap for migration plan.

**Input:**
- `plan_id` (int): Migration plan ID

**Output:**
- `phases`: Phase breakdown with timelines
- `milestones`: Key milestones
- `critical_path`: Critical path steps
- `resource_allocation`: Resource needs by phase

---

#### `optimize_migration_plan`
Optimizes existing plan based on constraints.

**Input:**
- `plan_id` (int): Plan to optimize
- `minimize_time` (bool): Reduce timeline
- `minimize_risk` (bool): Add safety measures
- `maximize_quality` (bool): Increase quality checks

**Output:**
- Optimized plan with updated steps and timeline

---

#### `estimate_migration_effort`
Provides detailed effort estimates.

**Input:**
- `plan_id` (int): Plan to estimate
- `include_testing` (bool): Include test effort
- `include_documentation` (bool): Include docs effort
- `productivity_factor` (float): Team productivity (0.5-1.5)

**Output:**
- `total_effort_hours`: Total estimated hours
- `breakdown`: Effort by category
- `assumptions`: Estimation assumptions

---

#### `plan_migration_resources`
Creates resource allocation plan.

**Input:**
- `plan_id` (int): Plan ID
- `available_developers` (int): Developer count
- `available_architects` (int): Architect count
- `available_qa` (int): QA engineer count
- `sprint_weeks` (int): Sprint duration

**Output:**
- `allocation_plan`: Sprint-by-sprint allocation
- `resource_requirements`: Skill requirements
- `utilization_metrics`: Resource utilization

### 3. Interface Design Tools

#### `design_module_interface`
Designs clean interfaces for modules.

**Input:**
- `repository_url` (str): Repository URL
- `package_path` (str): Package to design interface for
- `target_architecture` (str): Target architecture
- `include_events` (bool): Include domain events

**Output:**
- `public_api`: API specification
- `data_contracts`: DTOs and value objects
- `domain_events`: Event definitions
- `dependency_interfaces`: Required interfaces
- `implementation_notes`: Guidelines

---

#### `generate_interface_documentation`
Generates comprehensive interface documentation.

**Input:**
- `interface_design` (dict): Interface design from previous tool

**Output:**
- Markdown documentation with examples

### 4. Migration Execution Tools

#### `start_migration_step`
Begins execution of a migration step.

**Input:**
- `step_id` (int): Step to start
- `executor_id` (str, optional): Person/team executing

**Output:**
- `execution_id`: Execution tracking ID
- `step_details`: Step information
- `dependencies_status`: Dependency check results

---

#### `complete_migration_step`
Marks a step as completed.

**Input:**
- `step_id` (int): Step to complete
- `success` (bool): Success status
- `notes` (str): Completion notes
- `validation_results` (dict): Validation data

**Output:**
- `duration_hours`: Actual duration
- `next_steps`: Available next steps

---

#### `validate_migration_step`
Validates a completed migration step.

**Input:**
- `step_id` (int): Step to validate
- `validation_type` (str): Type (automated/manual/performance)
- `validation_criteria` (dict): Criteria to check

**Output:**
- `validation_id`: Validation record ID
- `passed` (bool): Pass/fail status
- `results`: Detailed results

---

#### `rollback_migration_step`
Rolls back a migration step.

**Input:**
- `step_id` (int): Step to rollback
- `reason` (str): Rollback reason

**Output:**
- `success` (bool): Rollback success
- `rollback_execution_id`: Rollback tracking ID

### 5. Monitoring & Tracking Tools

#### `get_migration_dashboard`
Real-time migration dashboard.

**Input:**
- `repository_url` (str, optional): Filter by repository

**Output:**
- `summary`: Overview metrics
- `active_migrations`: Currently active plans
- `recent_activity`: Recent events
- `alerts`: Issues requiring attention
- `risk_overview`: Current risk status

---

#### `track_migration_progress`
Tracks detailed progress of a plan.

**Input:**
- `plan_id` (int): Plan to track

**Output:**
- `progress_summary`: Completion metrics
- `time_tracking`: Schedule adherence
- `current_phase`: Active phase
- `blockers`: Current issues
- `health_score`: Overall health (0-100)

---

#### `monitor_step_execution`
Real-time monitoring of executing step.

**Input:**
- `step_id` (int): Step to monitor

**Output:**
- `status`: Current status
- `progress_percentage`: Completion percentage
- `elapsed_time`: Time elapsed
- `estimated_remaining`: Time remaining
- `recent_logs`: Recent activity

---

#### `get_migration_timeline`
Visualizes migration timeline.

**Input:**
- `plan_id` (int): Plan ID

**Output:**
- `timeline_events`: Chronological events
- `critical_path`: Critical path visualization
- `estimated_completion`: Completion date
- `phase_transitions`: Phase boundaries

---

#### `check_migration_health`
Comprehensive health assessment.

**Input:**
- `plan_id` (int): Plan to check

**Output:**
- `overall_health_score`: Score (0-100)
- `health_checks`: Individual check results
- `issues`: Identified problems
- `recommendations`: Improvement suggestions

---

#### `detect_anomalies`
Detects execution anomalies.

**Input:**
- `plan_id` (int): Plan to analyze

**Output:**
- List of anomalies:
  - `type`: Anomaly type
  - `severity`: Impact level
  - `affected_steps`: Impacted steps
  - `recommended_action`: Suggested response

---

#### `generate_status_report`
Generates migration status reports.

**Input:**
- `plan_id` (int): Plan ID
- `report_type` (str): Type (executive/summary/detailed)

**Output:**
- Formatted report with appropriate detail level

---

#### `generate_execution_report`
Comprehensive execution history report.

**Input:**
- `plan_id` (int): Plan ID

**Output:**
- `execution_summary`: Overall execution metrics
- `phase_breakdown`: Phase-wise analysis
- `issues_encountered`: Problems and resolutions
- `lessons_learned`: Key takeaways

### 6. Knowledge Management Tools

#### `extract_migration_patterns`
Extracts reusable patterns from completed migrations.

**Input:**
- `plan_id` (int): Completed plan ID

**Output:**
- `patterns`: Extracted patterns
- `categories`: Pattern categories
- `confidence_scores`: Pattern confidence

---

#### `add_pattern_to_library`
Adds custom pattern to knowledge library.

**Input:**
- `name` (str): Pattern name
- `category` (str): Category
- `description` (str): Description
- `implementation_steps` (list): Step-by-step guide
- `prerequisites` (list): Requirements
- `best_practices` (list): Recommendations
- `avg_effort_hours` (float, optional): Typical effort

**Output:**
- `pattern_id`: Created pattern ID

---

#### `search_patterns`
Searches pattern library.

**Input:**
- `query` (str, optional): Text search
- `category` (str, optional): Category filter
- `min_success_rate` (float, optional): Minimum success rate
- `applicable_to` (dict, optional): Context filter

**Output:**
- List of matching patterns with scores

---

#### `get_pattern_recommendations`
AI-powered pattern recommendations.

**Input:**
- `repository_url` (str): Repository URL
- `context` (dict): Migration context

**Output:**
- Ranked pattern recommendations

---

#### `update_pattern_from_execution`
Updates pattern statistics from execution.

**Input:**
- `pattern_id` (int): Pattern ID
- `execution_data` (dict): Execution results

**Output:**
- Updated pattern statistics

---

#### `learn_from_failures`
Analyzes failures to extract lessons.

**Input:**
- `plan_id` (int): Failed plan ID

**Output:**
- `failure_analysis`: Failure breakdown
- `root_causes`: Identified causes
- `prevention_strategies`: Prevention recommendations

---

#### `generate_pattern_documentation`
Generates pattern documentation.

**Input:**
- `pattern_id` (int): Pattern ID

**Output:**
- Comprehensive markdown documentation

---

#### `share_migration_knowledge`
Exports knowledge for sharing.

**Input:**
- `knowledge_type` (str): Type (patterns/lessons/best_practices)
- `format` (str): Format (markdown/json)
- `filter_category` (str, optional): Category filter

**Output:**
- Formatted knowledge export

## Error Handling

All tools follow consistent error handling:

- **Invalid Input**: Returns error with descriptive message
- **Resource Not Found**: Returns 404-style error
- **Permission Denied**: Returns authorization error
- **Server Error**: Returns 500-style error with details

## Rate Limiting

- Analysis tools: 10 requests per minute
- Planning tools: 20 requests per minute
- Execution tools: 50 requests per minute
- Monitoring tools: 100 requests per minute
- Knowledge tools: 30 requests per minute

## Best Practices

1. **Always start with analysis** before creating plans
2. **Use risk assessment** to identify critical issues early
3. **Monitor health scores** throughout execution
4. **Extract patterns** from successful migrations
5. **Learn from failures** to improve future migrations
6. **Share knowledge** across teams

## Example Workflows

### Basic Migration Workflow
```python
# 1. Analyze
readiness = analyze_migration_readiness(repo_url)

# 2. Plan
plan = create_migration_plan(
    repository_url=repo_url,
    strategy=readiness["recommended_strategy"],
    ...
)

# 3. Execute
for step in plan["steps"]:
    start_migration_step(step["id"])
    # ... work ...
    complete_migration_step(step["id"], success=True)

# 4. Learn
patterns = extract_migration_patterns(plan["plan_id"])
```

### Advanced Monitoring Workflow
```python
# Real-time monitoring
dashboard = get_migration_dashboard()
health = check_migration_health(plan_id)
anomalies = detect_anomalies(plan_id)

# Generate reports
exec_report = generate_execution_report(plan_id)
status_report = generate_status_report(plan_id, "executive")
```

## Integration Points

- **Version Control**: Integrates with Git for tracking changes
- **CI/CD**: Can trigger validations in pipelines
- **Issue Tracking**: Can create issues for failures
- **Documentation**: Auto-generates migration docs
- **Monitoring**: Exports metrics to monitoring systems
