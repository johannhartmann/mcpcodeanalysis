"""Database repositories for the MCP Code Analysis Server."""

from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.models import (
    Class,
    CodeEmbedding,
    Commit,
    File,
    Function,
    Import,
    Module,
    Repository,
    SearchQuery,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RepositoryRepo:
    """Repository for GitHub repositories."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **kwargs: Any) -> Repository:
        """Create a new repository."""
        repo = Repository(**kwargs)
        self.session.add(repo)
        await self.session.commit()
        await self.session.refresh(repo)
        return repo

    async def get_by_url(self, url: str) -> Repository | None:
        """Get repository by URL."""
        result = await self.session.execute(
            select(Repository).where(Repository.github_url == url),
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, repo_id: int) -> Repository | None:
        """Get repository by ID."""
        result = await self.session.execute(
            select(Repository).where(Repository.id == repo_id),
        )
        return result.scalar_one_or_none()

    async def list_all(self) -> list[Repository]:
        """List all repositories."""
        result = await self.session.execute(select(Repository))
        return list(result.scalars().all())

    async def update_last_synced(self, repo_id: int) -> None:
        """Update last synced timestamp."""
        await self.session.execute(
            select(Repository).where(Repository.id == repo_id).with_for_update(),
        )
        repo = await self.get_by_id(repo_id)
        if repo:
            repo.last_synced = datetime.utcnow()
            await self.session.commit()


class FileRepo:
    """Repository for source files."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **kwargs: Any) -> File:
        """Create a new file."""
        file = File(**kwargs)
        self.session.add(file)
        await self.session.commit()
        await self.session.refresh(file)
        return file

    async def get_by_path(
        self, repo_id: int, path: str, branch: str = "main",
    ) -> File | None:
        """Get file by path."""
        result = await self.session.execute(
            select(File).where(
                and_(
                    File.repository_id == repo_id,
                    File.path == path,
                    File.branch == branch,
                ),
            ),
        )
        return result.scalar_one_or_none()

    async def update_or_create(
        self, repo_id: int, path: str, **kwargs: Any,
    ) -> tuple[File, bool]:
        """Update existing file or create new one."""
        file = await self.get_by_path(repo_id, path, kwargs.get("branch", "main"))
        created = False

        if file:
            for key, value in kwargs.items():
                setattr(file, key, value)
        else:
            file = File(repository_id=repo_id, path=path, **kwargs)
            self.session.add(file)
            created = True

        await self.session.commit()
        await self.session.refresh(file)
        return file, created

    async def get_modified_since(self, repo_id: int, since: datetime) -> list[File]:
        """Get files modified since given timestamp."""
        result = await self.session.execute(
            select(File).where(
                and_(File.repository_id == repo_id, File.last_modified > since),
            ),
        )
        return list(result.scalars().all())

    async def delete_by_id(self, file_id: int) -> None:
        """Delete a file and all related entities."""
        file = await self.session.get(File, file_id)
        if file:
            await self.session.delete(file)
            await self.session.commit()


class CodeEntityRepo:
    """Repository for code entities (modules, classes, functions)."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_module(self, **kwargs: Any) -> Module:
        """Create a new module."""
        module = Module(**kwargs)
        self.session.add(module)
        await self.session.commit()
        await self.session.refresh(module)
        return module

    async def create_class(self, **kwargs: Any) -> Class:
        """Create a new class."""
        cls = Class(**kwargs)
        self.session.add(cls)
        await self.session.commit()
        await self.session.refresh(cls)
        return cls

    async def create_function(self, **kwargs: Any) -> Function:
        """Create a new function."""
        func = Function(**kwargs)
        self.session.add(func)
        await self.session.commit()
        await self.session.refresh(func)
        return func

    async def create_import(self, **kwargs: Any) -> Import:
        """Create a new import."""
        imp = Import(**kwargs)
        self.session.add(imp)
        await self.session.commit()
        await self.session.refresh(imp)
        return imp

    async def clear_file_entities(self, file_id: int) -> None:
        """Clear all entities for a file before re-parsing."""
        # Delete modules (cascades to classes and functions)
        await self.session.execute(select(Module).where(Module.file_id == file_id))
        modules = await self.session.execute(
            select(Module).where(Module.file_id == file_id),
        )
        for module in modules.scalars():
            await self.session.delete(module)

        # Delete imports
        await self.session.execute(select(Import).where(Import.file_id == file_id))
        imports = await self.session.execute(
            select(Import).where(Import.file_id == file_id),
        )
        for imp in imports.scalars():
            await self.session.delete(imp)

        await self.session.commit()

    async def find_by_name(
        self, name: str, entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find entities by name."""
        results = []

        if entity_type in (None, "function"):
            funcs = await self.session.execute(
                select(Function)
                .options(selectinload(Function.module), selectinload(Function.class_))
                .where(Function.name == name),
            )
            for func in funcs.scalars():
                results.append(
                    {
                        "type": "function",
                        "entity": func,
                        "module": func.module,
                        "class": func.class_,
                    },
                )

        if entity_type in (None, "class"):
            classes = await self.session.execute(
                select(Class)
                .options(selectinload(Class.module))
                .where(Class.name == name),
            )
            for cls in classes.scalars():
                results.append(
                    {
                        "type": "class",
                        "entity": cls,
                        "module": cls.module,
                        "class": None,
                    },
                )

        if entity_type in (None, "module"):
            modules = await self.session.execute(
                select(Module)
                .options(selectinload(Module.file))
                .where(Module.name == name),
            )
            for module in modules.scalars():
                results.append(
                    {
                        "type": "module",
                        "entity": module,
                        "module": module,
                        "class": None,
                    },
                )

        return results


class EmbeddingRepo:
    """Repository for code embeddings."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **kwargs: Any) -> CodeEmbedding:
        """Create a new embedding."""
        embedding = CodeEmbedding(**kwargs)
        self.session.add(embedding)
        await self.session.commit()
        await self.session.refresh(embedding)
        return embedding

    async def create_batch(self, embeddings: list[dict[str, Any]]) -> None:
        """Create multiple embeddings in batch."""
        for emb_data in embeddings:
            embedding = CodeEmbedding(**emb_data)
            self.session.add(embedding)
        await self.session.commit()

    async def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        threshold: float = 0.7,
        embedding_type: str | None = None,
        entity_types: list[str] | None = None,
    ) -> list[tuple[CodeEmbedding, float]]:
        """Search for similar embeddings using cosine similarity."""
        # Build query
        query = select(
            CodeEmbedding,
            func.cosine_distance(CodeEmbedding.embedding, query_embedding).label(
                "distance",
            ),
        )

        # Apply filters
        filters = []
        if embedding_type:
            filters.append(CodeEmbedding.embedding_type == embedding_type)
        if entity_types:
            filters.append(CodeEmbedding.entity_type.in_(entity_types))

        if filters:
            query = query.where(and_(*filters))

        # Order by similarity and apply threshold
        query = (
            query.where(
                func.cosine_distance(CodeEmbedding.embedding, query_embedding)
                <= (1 - threshold),
            )
            .order_by("distance")
            .limit(limit)
        )

        # Execute query
        result = await self.session.execute(query)

        # Convert distance to similarity score
        return [(embedding, 1 - distance) for embedding, distance in result.all()]

    async def get_by_entity(
        self, entity_type: str, entity_id: int, embedding_type: str | None = None,
    ) -> list[CodeEmbedding]:
        """Get embeddings for a specific entity."""
        query = select(CodeEmbedding).where(
            and_(
                CodeEmbedding.entity_type == entity_type,
                CodeEmbedding.entity_id == entity_id,
            ),
        )

        if embedding_type:
            query = query.where(CodeEmbedding.embedding_type == embedding_type)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def delete_by_entity(self, entity_type: str, entity_id: int) -> None:
        """Delete all embeddings for an entity."""
        await self.session.execute(
            select(CodeEmbedding).where(
                and_(
                    CodeEmbedding.entity_type == entity_type,
                    CodeEmbedding.entity_id == entity_id,
                ),
            ),
        )
        embeddings = await self.session.execute(
            select(CodeEmbedding).where(
                and_(
                    CodeEmbedding.entity_type == entity_type,
                    CodeEmbedding.entity_id == entity_id,
                ),
            ),
        )
        for embedding in embeddings.scalars():
            await self.session.delete(embedding)
        await self.session.commit()


class CommitRepo:
    """Repository for git commits."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **kwargs: Any) -> Commit:
        """Create a new commit."""
        commit = Commit(**kwargs)
        self.session.add(commit)
        await self.session.commit()
        await self.session.refresh(commit)
        return commit

    async def get_by_sha(self, repo_id: int, sha: str) -> Commit | None:
        """Get commit by SHA."""
        result = await self.session.execute(
            select(Commit).where(
                and_(Commit.repository_id == repo_id, Commit.sha == sha),
            ),
        )
        return result.scalar_one_or_none()

    async def get_latest(self, repo_id: int) -> Commit | None:
        """Get latest commit for a repository."""
        result = await self.session.execute(
            select(Commit)
            .where(Commit.repository_id == repo_id)
            .order_by(Commit.timestamp.desc())
            .limit(1),
        )
        return result.scalar_one_or_none()

    async def create_batch(self, commits: list[dict[str, Any]]) -> None:
        """Create multiple commits in batch."""
        for commit_data in commits:
            # Check if commit already exists
            existing = await self.get_by_sha(
                commit_data["repository_id"], commit_data["sha"],
            )
            if not existing:
                commit = Commit(**commit_data)
                self.session.add(commit)
        await self.session.commit()


class SearchQueryRepo:
    """Repository for search query analytics."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def log_query(
        self,
        query: str,
        tool_name: str,
        results_count: int,
        execution_time_ms: float,
        **kwargs: Any,
    ) -> None:
        """Log a search query."""
        search_query = SearchQuery(
            query=query,
            tool_name=tool_name,
            results_count=results_count,
            execution_time_ms=execution_time_ms,
            **kwargs,
        )
        self.session.add(search_query)
        await self.session.commit()
