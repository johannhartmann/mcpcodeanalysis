"""Database models for migration intelligence functionality."""

from enum import Enum

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from src.database.models import Base


class MigrationStrategy(str, Enum):
    """Migration strategy types."""

    STRANGLER_FIG = "strangler_fig"
    BIG_BANG = "big_bang"
    GRADUAL = "gradual"
    BRANCH_BY_ABSTRACTION = "branch_by_abstraction"
    PARALLEL_RUN = "parallel_run"


class MigrationPhase(str, Enum):
    """Migration phase types."""

    PLANNING = "planning"
    PREPARATION = "preparation"
    EXECUTION = "execution"
    VALIDATION = "validation"
    COMPLETION = "completion"


class MigrationStepStatus(str, Enum):
    """Migration step status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"


class RiskLevel(str, Enum):
    """Risk level categories."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResourceType(str, Enum):
    """Types of resources required for migration."""

    DEVELOPER = "developer"
    ARCHITECT = "architect"
    DBA = "dba"
    DEVOPS = "devops"
    QA = "qa"
    BUSINESS_ANALYST = "business_analyst"


class MigrationPlan(Base):
    """Represents a complete migration plan for transforming a monolith."""

    __tablename__ = "migration_plans"

    id = Column(Integer, primary_key=True)
    repository_id = Column(
        Integer, ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False
    )

    # Plan metadata
    name = Column(String(255), nullable=False)
    description = Column(Text)
    strategy = Column(String(50), nullable=False)
    target_architecture = Column(
        String(100), nullable=False
    )  # modular_monolith, microservices, etc.

    # Planning parameters
    risk_tolerance = Column(String(20), nullable=False, default="medium")
    timeline_weeks = Column(Integer)
    team_size = Column(Integer)

    # Computed metrics
    total_effort_hours = Column(Float)
    complexity_score = Column(Float)
    risk_score = Column(Float)
    confidence_level = Column(Float)  # 0.0 to 1.0

    # Status tracking
    current_phase = Column(String(20), default=MigrationPhase.PLANNING.value)
    progress_percentage = Column(Float, default=0.0)
    created_by = Column(String(255))  # User or system that created the plan

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    # JSON fields for flexible data
    constraints = Column(JSON)  # Business constraints, downtime windows, etc.
    success_metrics = Column(JSON)  # KPIs and success criteria

    # Relationships
    repository = relationship("Repository", back_populates="migration_plans")
    steps = relationship(
        "MigrationStep", back_populates="plan", cascade="all, delete-orphan"
    )
    risks = relationship(
        "MigrationRisk", back_populates="plan", cascade="all, delete-orphan"
    )
    decisions = relationship(
        "MigrationDecision", back_populates="plan", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "confidence_level >= 0 AND confidence_level <= 1",
            name="check_confidence_level",
        ),
        CheckConstraint(
            "progress_percentage >= 0 AND progress_percentage <= 100",
            name="check_progress_percentage",
        ),
    )


class MigrationStep(Base):
    """Individual step in a migration plan."""

    __tablename__ = "migration_steps"

    id = Column(Integer, primary_key=True)
    plan_id = Column(
        Integer, ForeignKey("migration_plans.id", ondelete="CASCADE"), nullable=False
    )

    # Step definition
    sequence_number = Column(Integer, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    step_type = Column(
        String(50), nullable=False
    )  # extract_module, refactor, create_interface, etc.

    # Target entities
    bounded_context_id = Column(
        Integer, ForeignKey("bounded_contexts.id", ondelete="SET NULL")
    )
    target_module_id = Column(Integer, ForeignKey("modules.id", ondelete="SET NULL"))
    target_package_id = Column(Integer, ForeignKey("packages.id", ondelete="SET NULL"))

    # Execution details
    status = Column(String(20), default=MigrationStepStatus.PENDING.value)
    estimated_hours = Column(Float)
    actual_hours = Column(Float)
    assigned_to = Column(String(255))  # Team or individual assignment

    # Validation criteria
    validation_criteria = Column(JSON)  # List of checks to perform
    rollback_procedure = Column(JSON)  # Steps to rollback if needed

    # Execution tracking
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    validated_at = Column(DateTime(timezone=True))

    # Additional data
    prerequisites = Column(JSON)  # Conditions that must be met
    outputs = Column(JSON)  # Expected outputs/artifacts
    notes = Column(Text)  # Execution notes and learnings

    # Relationships
    plan = relationship("MigrationPlan", back_populates="steps")
    dependencies = relationship(
        "MigrationDependency",
        foreign_keys="MigrationDependency.dependent_step_id",
        back_populates="dependent_step",
        cascade="all, delete-orphan",
    )
    dependents = relationship(
        "MigrationDependency",
        foreign_keys="MigrationDependency.prerequisite_step_id",
        back_populates="prerequisite_step",
        cascade="all, delete-orphan",
    )
    progress_updates = relationship(
        "MigrationProgress", back_populates="step", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("plan_id", "sequence_number", name="uq_plan_sequence"),
    )


class MigrationDependency(Base):
    """Dependencies between migration steps."""

    __tablename__ = "migration_dependencies"

    id = Column(Integer, primary_key=True)
    dependent_step_id = Column(
        Integer, ForeignKey("migration_steps.id", ondelete="CASCADE"), nullable=False
    )
    prerequisite_step_id = Column(
        Integer, ForeignKey("migration_steps.id", ondelete="CASCADE"), nullable=False
    )

    # Dependency metadata
    dependency_type = Column(String(50), nullable=False)  # hard, soft, optional
    description = Column(Text)

    # Relationships
    dependent_step = relationship(
        "MigrationStep", foreign_keys=[dependent_step_id], back_populates="dependencies"
    )
    prerequisite_step = relationship(
        "MigrationStep",
        foreign_keys=[prerequisite_step_id],
        back_populates="dependents",
    )

    __table_args__ = (
        UniqueConstraint(
            "dependent_step_id", "prerequisite_step_id", name="uq_step_dependency"
        ),
        CheckConstraint(
            "dependent_step_id != prerequisite_step_id", name="check_no_self_dependency"
        ),
    )


class MigrationRisk(Base):
    """Risk assessment for migration activities."""

    __tablename__ = "migration_risks"

    id = Column(Integer, primary_key=True)
    plan_id = Column(Integer, ForeignKey("migration_plans.id", ondelete="CASCADE"))
    step_id = Column(Integer, ForeignKey("migration_steps.id", ondelete="CASCADE"))

    # Risk details
    risk_type = Column(String(50), nullable=False)  # technical, operational, business
    name = Column(String(255), nullable=False)
    description = Column(Text)

    # Risk assessment
    probability = Column(Float, nullable=False)  # 0.0 to 1.0
    impact = Column(Float, nullable=False)  # 0.0 to 1.0
    risk_level = Column(
        String(20), nullable=False
    )  # Calculated from probability * impact

    # Mitigation
    mitigation_strategy = Column(Text)
    contingency_plan = Column(Text)
    owner = Column(String(255))

    # Status tracking
    identified_at = Column(DateTime(timezone=True), server_default=func.now())
    mitigated_at = Column(DateTime(timezone=True))
    realized_at = Column(DateTime(timezone=True))  # If risk actually occurred

    # Relationships
    plan = relationship("MigrationPlan", back_populates="risks")
    step = relationship("MigrationStep")

    __table_args__ = (
        CheckConstraint(
            "probability >= 0 AND probability <= 1", name="check_risk_probability"
        ),
        CheckConstraint("impact >= 0 AND impact <= 1", name="check_risk_impact"),
        CheckConstraint(
            "plan_id IS NOT NULL OR step_id IS NOT NULL", name="check_risk_association"
        ),
    )


class MigrationProgress(Base):
    """Track progress updates for migration steps."""

    __tablename__ = "migration_progress"

    id = Column(Integer, primary_key=True)
    step_id = Column(
        Integer, ForeignKey("migration_steps.id", ondelete="CASCADE"), nullable=False
    )

    # Progress details
    update_type = Column(
        String(50), nullable=False
    )  # status_change, metric_update, validation_result
    description = Column(Text)
    percentage_complete = Column(Float)

    # Metrics at time of update
    metrics = Column(JSON)  # Quality metrics, performance metrics, etc.
    issues_found = Column(JSON)  # Any issues discovered

    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    recorded_by = Column(String(255))

    # Relationships
    step = relationship("MigrationStep", back_populates="progress_updates")


class MigrationPattern(Base):
    """Library of reusable migration patterns."""

    __tablename__ = "migration_patterns"

    id = Column(Integer, primary_key=True)

    # Pattern identification
    name = Column(String(255), nullable=False, unique=True)
    category = Column(
        String(100), nullable=False
    )  # extraction, refactoring, interface_design, etc.
    description = Column(Text)

    # Pattern details
    context = Column(JSON)  # When to use this pattern
    solution = Column(JSON)  # How to apply the pattern
    consequences = Column(JSON)  # Trade-offs and impacts
    implementation_steps = Column(JSON)  # Step-by-step implementation
    prerequisites = Column(JSON)  # Prerequisites for using this pattern
    best_practices = Column(JSON)  # Best practices when applying pattern
    anti_patterns = Column(JSON)  # Anti-patterns to avoid
    risks = Column(JSON)  # Known risks when applying this pattern
    example_code = Column(Text)  # Example code demonstrating the pattern

    # Success metrics
    success_rate = Column(Float)  # Historical success rate
    avg_effort_hours = Column(Float)
    applicable_scenarios = Column(JSON)

    # Versioning
    version = Column(String(20), nullable=False, default="1.0.0")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime(timezone=True))

    # Relationships
    pattern_applications = relationship(
        "MigrationPatternApplication", back_populates="pattern"
    )


class MigrationPatternApplication(Base):
    """Track applications of migration patterns."""

    __tablename__ = "migration_pattern_applications"

    id = Column(Integer, primary_key=True)
    pattern_id = Column(
        Integer, ForeignKey("migration_patterns.id", ondelete="CASCADE"), nullable=False
    )
    plan_id = Column(
        Integer, ForeignKey("migration_plans.id", ondelete="CASCADE"), nullable=False
    )
    step_id = Column(Integer, ForeignKey("migration_steps.id", ondelete="CASCADE"))

    # Application details
    applied_at = Column(DateTime(timezone=True), server_default=func.now())
    success = Column(Integer)  # 1 for success, 0 for failure, NULL for in-progress
    modifications = Column(JSON)  # Any adaptations made to the pattern
    lessons_learned = Column(Text)

    # Relationships
    pattern = relationship("MigrationPattern", back_populates="pattern_applications")
    plan = relationship("MigrationPlan")
    step = relationship("MigrationStep")


class MigrationDecision(Base):
    """Track important decisions made during migration planning and execution."""

    __tablename__ = "migration_decisions"

    id = Column(Integer, primary_key=True)
    plan_id = Column(
        Integer, ForeignKey("migration_plans.id", ondelete="CASCADE"), nullable=False
    )
    step_id = Column(Integer, ForeignKey("migration_steps.id", ondelete="CASCADE"))

    # Decision details
    decision_type = Column(
        String(100), nullable=False
    )  # strategy_selection, boundary_definition, etc.
    title = Column(String(255), nullable=False)
    description = Column(Text)

    # Decision process
    alternatives_considered = Column(JSON)  # List of options evaluated
    evaluation_criteria = Column(JSON)  # Criteria used for evaluation
    rationale = Column(Text, nullable=False)  # Why this decision was made

    # Decision metadata
    made_by = Column(String(255))
    made_at = Column(DateTime(timezone=True), server_default=func.now())

    # Outcome tracking
    outcome_assessment = Column(Text)  # How the decision worked out
    assessed_at = Column(DateTime(timezone=True))
    success_rating = Column(Integer)  # 1-5 scale

    # Relationships
    plan = relationship("MigrationPlan", back_populates="decisions")
    step = relationship("MigrationStep")

    __table_args__ = (
        CheckConstraint(
            "success_rating >= 1 AND success_rating <= 5", name="check_success_rating"
        ),
    )


class MigrationResourceRequirement(Base):
    """Resource requirements for migration activities."""

    __tablename__ = "migration_resource_requirements"

    id = Column(Integer, primary_key=True)
    plan_id = Column(Integer, ForeignKey("migration_plans.id", ondelete="CASCADE"))
    step_id = Column(Integer, ForeignKey("migration_steps.id", ondelete="CASCADE"))

    # Resource specification
    resource_type = Column(String(50), nullable=False)
    skill_level = Column(String(50), nullable=False)  # junior, mid, senior, expert
    quantity = Column(Float, nullable=False)  # FTE or hours

    # Timing
    start_date = Column(DateTime(timezone=True))
    end_date = Column(DateTime(timezone=True))

    # Additional requirements
    specific_skills = Column(JSON)  # List of specific skills needed
    tools_required = Column(JSON)  # Tools, licenses, infrastructure

    # Relationships
    plan = relationship("MigrationPlan")
    step = relationship("MigrationStep")

    __table_args__ = (
        CheckConstraint(
            "plan_id IS NOT NULL OR step_id IS NOT NULL",
            name="check_resource_association",
        ),
    )


class MigrationQualityMetric(Base):
    """Track quality metrics throughout migration."""

    __tablename__ = "migration_quality_metrics"

    id = Column(Integer, primary_key=True)
    plan_id = Column(
        Integer, ForeignKey("migration_plans.id", ondelete="CASCADE"), nullable=False
    )
    step_id = Column(Integer, ForeignKey("migration_steps.id", ondelete="CASCADE"))

    # Metric details
    metric_type = Column(
        String(100), nullable=False
    )  # complexity, coupling, test_coverage, etc.
    metric_name = Column(String(255), nullable=False)

    # Values
    baseline_value = Column(Float)
    current_value = Column(Float, nullable=False)
    target_value = Column(Float)

    # Assessment
    trend = Column(String(20))  # improving, degrading, stable
    within_threshold = Column(Integer)  # 1 for yes, 0 for no

    # Timestamp
    measured_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    plan = relationship("MigrationPlan")
    step = relationship("MigrationStep")


class ValidationStatus(str, Enum):
    """Validation status types."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MigrationExecution(Base):
    """Track execution attempts of migration steps."""

    __tablename__ = "migration_executions"

    id = Column(Integer, primary_key=True)
    step_id = Column(
        Integer, ForeignKey("migration_steps.id", ondelete="CASCADE"), nullable=False
    )

    # Execution details
    execution_number = Column(Integer, nullable=False)  # Attempt number
    executor_id = Column(String(255))  # Who/what executed
    status = Column(String(20), nullable=False)  # started, completed, failed

    # Timing
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    duration_minutes = Column(Float)

    # Results
    success = Column(Integer)  # 1 for success, 0 for failure
    error_message = Column(Text)
    rollback_executed = Column(Integer, default=0)

    # Logs and artifacts
    execution_logs = Column(JSON)  # Structured logs
    artifacts_produced = Column(JSON)  # Files, configs, etc.

    # Relationships
    step = relationship("MigrationStep")


class MigrationValidation(Base):
    """Track validation results for migration steps."""

    __tablename__ = "migration_validations"

    id = Column(Integer, primary_key=True)
    step_id = Column(
        Integer, ForeignKey("migration_steps.id", ondelete="CASCADE"), nullable=False
    )
    execution_id = Column(
        Integer, ForeignKey("migration_executions.id", ondelete="CASCADE")
    )

    # Validation details
    validation_type = Column(
        String(50), nullable=False
    )  # automated, manual, performance
    status = Column(String(20), nullable=False, default=ValidationStatus.PENDING.value)

    # Criteria and results
    criteria = Column(JSON, nullable=False)  # What to validate
    results = Column(JSON)  # Validation results
    passed = Column(Integer)  # 1 for pass, 0 for fail

    # Timing
    validated_at = Column(DateTime(timezone=True), server_default=func.now())
    validated_by = Column(String(255))

    # Notes
    notes = Column(Text)

    # Relationships
    step = relationship("MigrationStep")
    execution = relationship("MigrationExecution")
