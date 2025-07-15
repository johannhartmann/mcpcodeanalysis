"""Database models for MCP Code Analysis Server."""

from __future__ import annotations

from typing import Any

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # For SQLite tests, use JSON instead of Vector
    def Vector(dim: int) -> type[JSON]:  # noqa: ARG001, N802
        _ = dim  # Mark as intentionally unused
        return JSON


from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)

# SQLAlchemy compatibility
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Create base class
Base: Any = declarative_base()

# For type annotations (using Any to avoid import errors)
Mapped = Any
mapped_column = Column


class Repository(Base):
    """GitHub repository model."""

    __tablename__ = "repositories"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    github_url = Column(String(500), unique=True, nullable=False)
    owner = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    default_branch = Column(String(255), default="main")
    access_token_id = Column(String(255), nullable=True)
    last_synced = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    webhook_id = Column(String(255), nullable=True)
    repo_metadata = Column(JSON, default=dict)

    # Relationships
    files: Any = relationship(
        "File",
        back_populates="repository",
        cascade="all, delete-orphan",
    )
    commits: Any = relationship(
        "Commit",
        back_populates="repository",
        cascade="all, delete-orphan",
    )
    migration_plans: Any = relationship(
        "MigrationPlan",
        back_populates="repository",
        cascade="all, delete-orphan",
    )

    __table_args__ = (Index("idx_repository_owner_name", "owner", "name"),)


class File(Base):
    """Source code file model."""

    __tablename__ = "files"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    path = Column(String(1000), nullable=False)
    content_hash = Column(String(64))  # SHA-256 hash
    git_hash = Column(String(40))  # Git blob hash
    branch = Column(String(255), default="main")
    size = Column(Integer)
    language = Column(String(50))
    last_modified = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    is_deleted = Column(Boolean, default=False)

    # Relationships
    repository: Any = relationship("Repository", back_populates="files")
    modules: Any = relationship(
        "Module",
        back_populates="file",
        cascade="all, delete-orphan",
    )
    imports: Any = relationship(
        "Import",
        back_populates="file",
        cascade="all, delete-orphan",
    )
    embeddings: Any = relationship(
        "CodeEmbedding",
        back_populates="file",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("repository_id", "path", "branch", name="uq_file_path"),
        Index("idx_file_repository", "repository_id"),
        Index("idx_file_path", "path"),
        Index("idx_file_language", "language"),
    )


class Module(Base):
    """Python module model."""

    __tablename__ = "modules"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    name = Column(String(255), nullable=False)
    docstring = Column(Text)
    start_line = Column(Integer)
    end_line = Column(Integer)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    file: Any = relationship("File", back_populates="modules")
    classes: Any = relationship(
        "Class",
        back_populates="module",
        cascade="all, delete-orphan",
    )
    functions: Any = relationship(
        "Function",
        primaryjoin="and_(Module.id==Function.module_id, Function.class_id==None)",
        back_populates="module",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_module_file", "file_id"),
        Index("idx_module_name", "name"),
    )


class Class(Base):
    """Class definition model."""

    __tablename__ = "classes"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    module_id = Column(Integer, ForeignKey("modules.id"), nullable=False)
    name = Column(String(255), nullable=False)
    docstring = Column(Text)
    base_classes = Column(JSON, default=[])  # List of base class names
    decorators = Column(JSON, default=[])  # List of decorator names
    start_line = Column(Integer)
    end_line = Column(Integer)
    is_abstract = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    module: Any = relationship("Module", back_populates="classes")
    methods: Any = relationship(
        "Function",
        primaryjoin="Class.id==Function.class_id",
        back_populates="parent_class",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_class_module", "module_id"),
        Index("idx_class_name", "name"),
    )


class Function(Base):
    """Function/method definition model."""

    __tablename__ = "functions"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    module_id = Column(Integer, ForeignKey("modules.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=True)
    name = Column(String(255), nullable=False)
    docstring = Column(Text)
    parameters = Column(JSON, default=[])  # List of parameter info
    return_type = Column(String(255))
    decorators = Column(JSON, default=[])  # List of decorator names
    start_line = Column(Integer)
    end_line = Column(Integer)
    is_async = Column(Boolean, default=False)
    is_generator = Column(Boolean, default=False)
    is_property = Column(Boolean, default=False)
    is_static = Column(Boolean, default=False)
    is_classmethod = Column(Boolean, default=False)
    complexity = Column(Integer)  # Cyclomatic complexity
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    module: Any = relationship(
        "Module",
        back_populates="functions",
        foreign_keys=[module_id],
    )
    parent_class: Any = relationship(
        "Class",
        back_populates="methods",
        foreign_keys=[class_id],
    )

    __table_args__ = (
        Index("idx_function_module", "module_id"),
        Index("idx_function_class", "class_id"),
        Index("idx_function_name", "name"),
    )


class Import(Base):
    """Import statement model."""

    __tablename__ = "imports"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    import_statement = Column(String(500), nullable=False)
    module_name = Column(String(255))
    imported_names = Column(JSON, default=[])  # List of imported names
    is_relative = Column(Boolean, default=False)
    level = Column(Integer, default=0)  # Relative import level
    line_number = Column(Integer)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    file: Any = relationship("File", back_populates="imports")

    __table_args__ = (
        Index("idx_import_file", "file_id"),
        Index("idx_import_module", "module_name"),
    )


class Commit(Base):
    """Git commit model."""

    __tablename__ = "commits"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    sha = Column(String(40), unique=True, nullable=False)
    message = Column(Text)
    author = Column(String(255))
    author_email = Column(String(255))
    timestamp = Column(DateTime, nullable=False)
    files_changed = Column(JSON, default=[])  # List of file paths
    additions = Column(Integer, default=0)
    deletions = Column(Integer, default=0)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    repository: Any = relationship("Repository", back_populates="commits")

    __table_args__ = (
        Index("idx_commit_repository", "repository_id"),
        Index("idx_commit_timestamp", "timestamp"),
        Index("idx_commit_processed", "processed"),
    )


class CodeReference(Base):
    """Track references between code entities."""

    __tablename__ = "code_references"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    # Source entity (the one making the reference)
    source_type = Column(String(50), nullable=False)  # module, class, function
    source_id = Column(Integer, nullable=False)
    source_file_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    source_line = Column(Integer)  # Line where reference occurs

    # Target entity (the one being referenced)
    target_type = Column(String(50), nullable=False)  # module, class, function
    target_id = Column(Integer, nullable=False)
    target_file_id = Column(Integer, ForeignKey("files.id"), nullable=False)

    # Reference details
    reference_type = Column(String(50))  # import, call, inherit, instantiate, type_hint
    context = Column(Text)  # Code context around the reference
    is_dynamic = Column(Boolean, default=False)  # Dynamic imports/calls

    created_at = Column(DateTime, default=func.now())

    # Relationships
    source_file: Any = relationship("File", foreign_keys=[source_file_id])
    target_file: Any = relationship("File", foreign_keys=[target_file_id])

    __table_args__ = (
        Index("idx_ref_source", "source_type", "source_id"),
        Index("idx_ref_target", "target_type", "target_id"),
        Index("idx_ref_files", "source_file_id", "target_file_id"),
        UniqueConstraint(
            "source_type",
            "source_id",
            "target_type",
            "target_id",
            "source_line",
            name="uq_code_reference",
        ),
    )


class CodeEmbedding(Base):
    """Code embedding model with pgvector."""

    __tablename__ = "code_embeddings"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    entity_type = Column(
        Enum("file", "module", "class", "function", name="code_entity_type"),
        nullable=False,
    )
    entity_id = Column(Integer, nullable=False)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    embedding_type = Column(
        Enum("raw", "interpreted", name="embedding_type"),
        nullable=False,
    )
    # Use Vector for PostgreSQL, JSON for SQLite
    embedding = Column(Vector(1536), nullable=False)  # OpenAI ada-002 dimension
    content = Column(Text, nullable=False)
    tokens = Column(Integer)
    repo_metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    file: Any = relationship("File", back_populates="embeddings")

    __table_args__ = (
        Index("idx_embedding_entity", "entity_type", "entity_id"),
        Index("idx_embedding_file", "file_id"),
        Index("idx_embedding_type", "embedding_type"),
        # Vector similarity index will be created separately with custom SQL
    )


class SearchHistory(Base):
    """Search query history for analytics."""

    __tablename__ = "search_history"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    query_type = Column(String(50))  # search_code, find_definition, etc.
    results_count = Column(Integer, default=0)
    response_time_ms = Column(Float)
    user_id = Column(String(255))  # Optional user tracking
    session_id = Column(String(255))
    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("idx_search_created", "created_at"),
        Index("idx_search_user", "user_id"),
    )
