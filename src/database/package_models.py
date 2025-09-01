"""Database models for package structure analysis."""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.models import Base


class Package(Base):
    """Python package model - represents a directory with __init__.py."""

    __tablename__ = "packages"

    id: Mapped[int] = mapped_column(primary_key=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id"), nullable=False
    )
    path: Mapped[str] = mapped_column(
        String(1000), nullable=False
    )  # Relative path from repo root
    name: Mapped[str] = mapped_column(
        String(255), nullable=False
    )  # Package name (last part of path)
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("packages.id"), nullable=True
    )

    # Package metadata
    init_file_id: Mapped[int | None] = mapped_column(
        ForeignKey("files.id"), nullable=True
    )
    has_init: Mapped[bool] = mapped_column(
        Boolean, default=True
    )  # False for namespace packages
    is_namespace: Mapped[bool] = mapped_column(Boolean, default=False)

    # Statistics
    module_count: Mapped[int] = mapped_column(Integer, default=0)
    subpackage_count: Mapped[int] = mapped_column(Integer, default=0)
    total_lines: Mapped[int] = mapped_column(Integer, default=0)
    total_functions: Mapped[int] = mapped_column(Integer, default=0)
    total_classes: Mapped[int] = mapped_column(Integer, default=0)

    # Documentation
    readme_file_id: Mapped[int | None] = mapped_column(
        ForeignKey("files.id"), nullable=True
    )
    docstring: Mapped[str | None] = mapped_column(Text)  # From __init__.py

    created_at: Mapped[Any] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[Any] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    repository: Mapped[Repository] = relationship("Repository")
    parent: Mapped[Package | None] = relationship(
        "Package", remote_side=[id], backref="subpackages"
    )
    init_file: Mapped[File | None] = relationship("File", foreign_keys=[init_file_id])
    readme_file: Mapped[File | None] = relationship(
        "File", foreign_keys=[readme_file_id]
    )

    # Modules in this package (direct children only)
    modules: Mapped[list[PackageModule]] = relationship(
        "PackageModule", back_populates="package", cascade="all, delete-orphan"
    )

    # Dependencies this package has
    dependencies: Mapped[list[PackageDependency]] = relationship(
        "PackageDependency",
        foreign_keys="PackageDependency.source_package_id",
        back_populates="source_package",
        cascade="all, delete-orphan",
    )

    # Other packages that depend on this one
    dependents: Mapped[list[PackageDependency]] = relationship(
        "PackageDependency",
        foreign_keys="PackageDependency.target_package_id",
        back_populates="target_package",
    )

    __table_args__ = (
        UniqueConstraint("repository_id", "path", name="uq_package_path"),
        Index("idx_package_repository", "repository_id"),
        Index("idx_package_parent", "parent_id"),
        Index("idx_package_name", "name"),
    )


class PackageModule(Base):
    """Association between packages and their direct module files."""

    __tablename__ = "package_modules"

    id: Mapped[int] = mapped_column(primary_key=True)
    package_id: Mapped[int] = mapped_column(ForeignKey("packages.id"), nullable=False)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id"), nullable=False)
    module_name: Mapped[str] = mapped_column(
        String(255), nullable=False
    )  # Without .py extension

    # Module characteristics
    is_public: Mapped[bool] = mapped_column(
        Boolean, default=True
    )  # Not starting with _
    has_main: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # Has if __name__ == "__main__"
    exports: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # List of exported names from __all__

    created_at: Mapped[Any] = mapped_column(DateTime, default=func.now())

    # Relationships
    package: Mapped[Package] = relationship("Package", back_populates="modules")
    file: Mapped[File] = relationship("File")

    __table_args__ = (
        UniqueConstraint("package_id", "file_id", name="uq_package_module"),
        Index("idx_package_module_package", "package_id"),
        Index("idx_package_module_file", "file_id"),
    )


class PackageDependency(Base):
    """Dependencies between packages within the repository."""

    __tablename__ = "package_dependencies"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_package_id: Mapped[int] = mapped_column(
        ForeignKey("packages.id"), nullable=False
    )
    target_package_id: Mapped[int] = mapped_column(
        ForeignKey("packages.id"), nullable=False
    )

    # Dependency strength
    import_count: Mapped[int] = mapped_column(Integer, default=1)  # Number of imports
    dependency_type: Mapped[str | None] = mapped_column(
        String(50)
    )  # direct, transitive

    # Import details
    import_details: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )  # List of specific imports

    created_at: Mapped[Any] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[Any] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    source_package: Mapped[Package] = relationship(
        "Package", foreign_keys=[source_package_id], back_populates="dependencies"
    )
    target_package: Mapped[Package] = relationship(
        "Package", foreign_keys=[target_package_id], back_populates="dependents"
    )

    __table_args__ = (
        UniqueConstraint(
            "source_package_id", "target_package_id", name="uq_package_dependency"
        ),
        Index("idx_package_dep_source", "source_package_id"),
        Index("idx_package_dep_target", "target_package_id"),
    )


class PackageMetrics(Base):
    """Aggregated metrics for packages."""

    __tablename__ = "package_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    package_id: Mapped[int] = mapped_column(
        ForeignKey("packages.id"), unique=True, nullable=False
    )

    # Complexity metrics
    total_complexity: Mapped[int] = mapped_column(Integer, default=0)
    avg_complexity: Mapped[int] = mapped_column(Integer, default=0)
    max_complexity: Mapped[int] = mapped_column(Integer, default=0)

    # Cohesion metrics
    cohesion_score: Mapped[int] = mapped_column(Integer, default=0)  # 0-100
    coupling_score: Mapped[int] = mapped_column(Integer, default=0)  # 0-100

    # Size metrics
    total_loc: Mapped[int] = mapped_column(Integer, default=0)  # Lines of code
    total_comments: Mapped[int] = mapped_column(Integer, default=0)
    total_docstrings: Mapped[int] = mapped_column(Integer, default=0)

    # Quality indicators
    test_coverage: Mapped[int | None] = mapped_column(
        Integer
    )  # Percentage if available
    has_tests: Mapped[bool] = mapped_column(Boolean, default=False)
    has_docs: Mapped[bool] = mapped_column(Boolean, default=False)

    # API surface
    public_classes: Mapped[int] = mapped_column(Integer, default=0)
    public_functions: Mapped[int] = mapped_column(Integer, default=0)
    public_constants: Mapped[int] = mapped_column(Integer, default=0)

    calculated_at: Mapped[Any] = mapped_column(DateTime, default=func.now())

    # Relationships
    package: Mapped[Package] = relationship("Package", backref="metrics", uselist=False)

    __table_args__ = (Index("idx_package_metrics_package", "package_id"),)
