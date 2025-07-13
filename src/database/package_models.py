"""Database models for package structure analysis."""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from src.database.models import Base


class Package(Base):
    """Python package model - represents a directory with __init__.py."""

    __tablename__ = "packages"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    path = Column(String(1000), nullable=False)  # Relative path from repo root
    name = Column(String(255), nullable=False)  # Package name (last part of path)
    parent_id = Column(Integer, ForeignKey("packages.id"), nullable=True)

    # Package metadata
    init_file_id = Column(Integer, ForeignKey("files.id"), nullable=True)
    has_init = Column(Boolean, default=True)  # False for namespace packages
    is_namespace = Column(Boolean, default=False)

    # Statistics
    module_count = Column(Integer, default=0)
    subpackage_count = Column(Integer, default=0)
    total_lines = Column(Integer, default=0)
    total_functions = Column(Integer, default=0)
    total_classes = Column(Integer, default=0)

    # Documentation
    readme_file_id = Column(Integer, ForeignKey("files.id"), nullable=True)
    docstring = Column(Text)  # From __init__.py

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    repository: Any = relationship("Repository")
    parent: Any = relationship("Package", remote_side=[id], backref="subpackages")
    init_file: Any = relationship("File", foreign_keys=[init_file_id])
    readme_file: Any = relationship("File", foreign_keys=[readme_file_id])

    # Modules in this package (direct children only)
    modules: Any = relationship(
        "PackageModule", back_populates="package", cascade="all, delete-orphan"
    )

    # Dependencies this package has
    dependencies: Any = relationship(
        "PackageDependency",
        foreign_keys="PackageDependency.source_package_id",
        back_populates="source_package",
        cascade="all, delete-orphan",
    )

    # Other packages that depend on this one
    dependents: Any = relationship(
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
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    package_id = Column(Integer, ForeignKey("packages.id"), nullable=False)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    module_name = Column(String(255), nullable=False)  # Without .py extension

    # Module characteristics
    is_public = Column(Boolean, default=True)  # Not starting with _
    has_main = Column(Boolean, default=False)  # Has if __name__ == "__main__"
    exports = Column(JSON, default=[])  # List of exported names from __all__

    created_at = Column(DateTime, default=func.now())

    # Relationships
    package: Any = relationship("Package", back_populates="modules")
    file: Any = relationship("File")

    __table_args__ = (
        UniqueConstraint("package_id", "file_id", name="uq_package_module"),
        Index("idx_package_module_package", "package_id"),
        Index("idx_package_module_file", "file_id"),
    )


class PackageDependency(Base):
    """Dependencies between packages within the repository."""

    __tablename__ = "package_dependencies"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    source_package_id = Column(Integer, ForeignKey("packages.id"), nullable=False)
    target_package_id = Column(Integer, ForeignKey("packages.id"), nullable=False)

    # Dependency strength
    import_count = Column(Integer, default=1)  # Number of imports
    dependency_type = Column(String(50))  # direct, transitive

    # Import details
    import_details = Column(JSON, default=[])  # List of specific imports

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    source_package: Any = relationship(
        "Package", foreign_keys=[source_package_id], back_populates="dependencies"
    )
    target_package: Any = relationship(
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
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True)
    package_id = Column(Integer, ForeignKey("packages.id"), unique=True, nullable=False)

    # Complexity metrics
    total_complexity = Column(Integer, default=0)
    avg_complexity = Column(Integer, default=0)
    max_complexity = Column(Integer, default=0)

    # Cohesion metrics
    cohesion_score = Column(Integer, default=0)  # 0-100
    coupling_score = Column(Integer, default=0)  # 0-100

    # Size metrics
    total_loc = Column(Integer, default=0)  # Lines of code
    total_comments = Column(Integer, default=0)
    total_docstrings = Column(Integer, default=0)

    # Quality indicators
    test_coverage = Column(Integer)  # Percentage if available
    has_tests = Column(Boolean, default=False)
    has_docs = Column(Boolean, default=False)

    # API surface
    public_classes = Column(Integer, default=0)
    public_functions = Column(Integer, default=0)
    public_constants = Column(Integer, default=0)

    calculated_at = Column(DateTime, default=func.now())

    # Relationships
    package: Any = relationship("Package", backref="metrics", uselist=False)

    __table_args__ = (Index("idx_package_metrics_package", "package_id"),)
