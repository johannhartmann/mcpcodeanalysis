"""Database models for MCP Code Analysis Server."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # For SQLite tests, use JSON instead of Vector
    def Vector(_dim: int) -> type[JSON]:  # noqa: N802
        return JSON


from sqlalchemy import (
    JSON,
    Boolean,
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
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Typed declarative base for all ORM models."""

    pass


class Repository(Base):
    """GitHub repository model."""

    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(primary_key=True)
    github_url: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    owner: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    default_branch: Mapped[str] = mapped_column(String(255), default="main")
    access_token_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    @property
    def access_token(self) -> str | None:
        """Convenience property for tests and callers that expect an access_token attribute.

        Maps to the underlying access_token_id column used for storage.
        """
        return cast("str | None", self.access_token_id)

    @access_token.setter
    def access_token(self, value: Any) -> None:
        # None -> clear
        if value is None:
            # mypy: Column[...] type at class-level; cast to Any for runtime assignment of None
            self.access_token_id = cast("Any", None)
            return

        # SecretStr-like objects expose get_secret_value
        if hasattr(value, "get_secret_value"):
            token = value.get_secret_value()
        else:
            token = str(value)

        self.access_token_id = cast("Any", token)

    last_synced: Mapped[datetime | None] = mapped_column(DateTime, default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    webhook_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    repo_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relationships
    files: Mapped[list[File]] = relationship(
        "File",
        back_populates="repository",
        cascade="all, delete-orphan",
    )
    commits: Mapped[list[Commit]] = relationship(
        "Commit",
        back_populates="repository",
        cascade="all, delete-orphan",
    )

    __table_args__ = (Index("idx_repository_owner_name", "owner", "name"),)


class File(Base):
    """Source code file model."""

    __tablename__ = "files"

    id: Mapped[int] = mapped_column(primary_key=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id"), nullable=False
    )
    path: Mapped[str] = mapped_column(String(1000), nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64))  # SHA-256 hash
    git_hash: Mapped[str | None] = mapped_column(String(40))  # Git blob hash
    branch: Mapped[str] = mapped_column(String(255), default="main")
    size: Mapped[int | None] = mapped_column(Integer)
    language: Mapped[str | None] = mapped_column(String(50))
    last_modified: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    repository: Mapped[Repository] = relationship("Repository", back_populates="files")
    modules: Mapped[list[Module]] = relationship(
        "Module",
        back_populates="file",
        cascade="all, delete-orphan",
    )
    imports: Mapped[list[Import]] = relationship(
        "Import",
        back_populates="file",
        cascade="all, delete-orphan",
    )
    embeddings: Mapped[list[CodeEmbedding]] = relationship(
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

    id: Mapped[int] = mapped_column(primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    docstring: Mapped[str | None] = mapped_column(Text)
    start_line: Mapped[int | None] = mapped_column(Integer)
    end_line: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    file: Mapped[File] = relationship("File", back_populates="modules")
    classes: Mapped[list[Class]] = relationship(
        "Class",
        back_populates="module",
        cascade="all, delete-orphan",
    )
    functions: Mapped[list[Function]] = relationship(
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

    id: Mapped[int] = mapped_column(primary_key=True)
    module_id: Mapped[int] = mapped_column(ForeignKey("modules.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    docstring: Mapped[str | None] = mapped_column(Text)
    base_classes: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # List of base class names
    decorators: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # List of decorator names
    start_line: Mapped[int | None] = mapped_column(Integer)
    end_line: Mapped[int | None] = mapped_column(Integer)
    is_abstract: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    module: Mapped[Module] = relationship("Module", back_populates="classes")
    methods: Mapped[list[Function]] = relationship(
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

    id: Mapped[int] = mapped_column(primary_key=True)
    module_id: Mapped[int] = mapped_column(ForeignKey("modules.id"), nullable=False)
    class_id: Mapped[int | None] = mapped_column(
        ForeignKey("classes.id"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    docstring: Mapped[str | None] = mapped_column(Text)
    parameters: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list
    )  # List of parameter info
    return_type: Mapped[str | None] = mapped_column(String(255))
    decorators: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # List of decorator names
    start_line: Mapped[int | None] = mapped_column(Integer)
    end_line: Mapped[int | None] = mapped_column(Integer)
    is_async: Mapped[bool] = mapped_column(Boolean, default=False)
    is_generator: Mapped[bool] = mapped_column(Boolean, default=False)
    is_property: Mapped[bool] = mapped_column(Boolean, default=False)
    is_static: Mapped[bool] = mapped_column(Boolean, default=False)
    is_classmethod: Mapped[bool] = mapped_column(Boolean, default=False)
    complexity: Mapped[int | None] = mapped_column(Integer)  # Cyclomatic complexity
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    module: Mapped[Module] = relationship(
        "Module",
        back_populates="functions",
        foreign_keys=[module_id],
    )
    parent_class: Mapped[Class | None] = relationship(
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

    id: Mapped[int] = mapped_column(primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id"), nullable=False)
    import_statement: Mapped[str] = mapped_column(String(500), nullable=False)
    module_name: Mapped[str | None] = mapped_column(String(255))
    imported_names: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # List of imported names
    is_relative: Mapped[bool] = mapped_column(Boolean, default=False)
    level: Mapped[int] = mapped_column(Integer, default=0)  # Relative import level
    line_number: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    file: Mapped[File] = relationship("File", back_populates="imports")

    __table_args__ = (
        Index("idx_import_file", "file_id"),
        Index("idx_import_module", "module_name"),
    )


class Commit(Base):
    """Git commit model."""

    __tablename__ = "commits"

    id: Mapped[int] = mapped_column(primary_key=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id"), nullable=False
    )
    sha: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(String(255))
    author_email: Mapped[str | None] = mapped_column(String(255))
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    files_changed: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # List of file paths
    additions: Mapped[int] = mapped_column(Integer, default=0)
    deletions: Mapped[int] = mapped_column(Integer, default=0)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    repository: Mapped[Repository] = relationship(
        "Repository", back_populates="commits"
    )

    __table_args__ = (
        Index("idx_commit_repository", "repository_id"),
        Index("idx_commit_timestamp", "timestamp"),
        Index("idx_commit_processed", "processed"),
    )


class CodeReference(Base):
    """Track references between code entities."""

    __tablename__ = "code_references"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Source entity (the one making the reference)
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # module, class, function
    source_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source_file_id: Mapped[int] = mapped_column(ForeignKey("files.id"), nullable=False)
    source_line: Mapped[int | None] = mapped_column(
        Integer
    )  # Line where reference occurs

    # Target entity (the one being referenced)
    target_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # module, class, function
    target_id: Mapped[int] = mapped_column(Integer, nullable=False)
    target_file_id: Mapped[int] = mapped_column(ForeignKey("files.id"), nullable=False)

    # Reference details
    reference_type: Mapped[str | None] = mapped_column(
        String(50)
    )  # import, call, inherit, instantiate, type_hint
    context: Mapped[str | None] = mapped_column(
        Text
    )  # Code context around the reference
    is_dynamic: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # Dynamic imports/calls

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    source_file: Mapped[File] = relationship("File", foreign_keys=[source_file_id])
    target_file: Mapped[File] = relationship("File", foreign_keys=[target_file_id])

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

    id: Mapped[int] = mapped_column(primary_key=True)
    # NOTE: The Enum names are SQLAlchemy Enum types; mypy needs explicit Python types
    entity_type: Mapped[str] = mapped_column(
        Enum("file", "module", "class", "function", name="code_entity_type"),
        nullable=False,
    )
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id"), nullable=False)
    embedding_type: Mapped[str] = mapped_column(
        Enum("raw", "interpreted", name="embedding_type"),
        nullable=False,
    )
    # Use Vector for PostgreSQL, JSON for SQLite
    embedding: Mapped[list[float]] = mapped_column(
        Vector(1536), nullable=False
    )  # OpenAI ada-002 dimension
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens: Mapped[int | None] = mapped_column(Integer)
    repo_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    file: Mapped[File] = relationship("File", back_populates="embeddings")

    __table_args__ = (
        Index("idx_embedding_entity", "entity_type", "entity_id"),
        Index("idx_embedding_file", "file_id"),
        Index("idx_embedding_type", "embedding_type"),
        # Vector similarity index will be created separately with custom SQL
    )


class SearchHistory(Base):
    """Search query history for analytics."""

    __tablename__ = "search_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str | None] = mapped_column(
        String(50)
    )  # search_code, find_definition, etc.
    results_count: Mapped[int] = mapped_column(Integer, default=0)
    response_time_ms: Mapped[float | None] = mapped_column(Float)
    user_id: Mapped[str | None] = mapped_column(String(255))  # Optional user tracking
    session_id: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (
        Index("idx_search_created", "created_at"),
        Index("idx_search_user", "user_id"),
    )
