"""Service for managing embeddings in the database."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.domain_models import DomainEntity
from src.database.models import (
    Class,
    CodeEmbedding,
    File,
    Function,
    Module,
)
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.logger import get_logger
from src.utils.exceptions import EmbeddingError, NotFoundError

logger = get_logger(__name__)


class EmbeddingService:
    """Service for creating and managing code embeddings."""

    def __init__(
        self,
        db_session: AsyncSession,
    ) -> None:
        """Initialize embedding service.

        Args:
            db_session: Database session
        """
        self.db_session = db_session
        self.embedding_generator = EmbeddingGenerator()

    async def create_file_embeddings(self, file_id: int) -> dict[str, Any]:
        """Create embeddings for all entities in a file.

        Args:
            file_id: File ID to process

        Returns:
            Summary of created embeddings
        """
        logger.info("Creating embeddings for file %d", file_id)

        # Get file record
        result = await self.db_session.execute(select(File).where(File.id == file_id))
        file_record = result.scalar_one_or_none()

        if not file_record:
            msg = "File not found"
            raise NotFoundError(msg)

        stats = {
            "file_id": file_id,
            "file_path": file_record.path,
            "modules": 0,
            "classes": 0,
            "functions": 0,
            "total": 0,
            "errors": [],
        }

        try:
            # Process modules
            modules = await self._get_file_modules(file_id)
            for module in modules:
                try:
                    await self._create_module_embedding(module, file_record.path)
                    stats["modules"] += 1
                except Exception as e:
                    logger.exception(
                        "Failed to create embedding for module %s",
                        module.id,
                    )
                    stats["errors"].append(f"Module {module.name}: {e!s}")

            # Process classes
            classes = await self._get_file_classes(file_id)
            for cls in classes:
                try:
                    await self._create_class_embedding(cls, file_record.path)
                    stats["classes"] += 1
                except Exception as e:
                    logger.exception(
                        "Failed to create embedding for class %s",
                        cls.id,
                    )
                    stats["errors"].append(f"Class {cls.name}: {e!s}")

            # Process functions
            functions = await self._get_file_functions(file_id)
            for func in functions:
                try:
                    await self._create_function_embedding(func, file_record.path)
                    stats["functions"] += 1
                except Exception as e:
                    logger.exception(
                        "Failed to create embedding for function %s",
                        func.id,
                    )
                    stats["errors"].append(f"Function {func.name}: {e!s}")

            stats["total"] = stats["modules"] + stats["classes"] + stats["functions"]

            logger.info(
                "Created %s embeddings for file %s (%s errors)",
                stats["total"],
                file_id,
                stats["errors"],
            )

        except Exception as e:
            logger.exception("Failed to create embeddings for file %s", file_id)
            msg = "Failed to create embeddings"
            raise EmbeddingError(msg) from e

        return stats

    async def create_repository_embeddings(
        self,
        repository_id: int,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Create embeddings for all files in a repository.

        Args:
            repository_id: Repository ID to process
            limit: Optional limit on number of files to process

        Returns:
            Summary of created embeddings
        """
        logger.info("Creating embeddings for repository %d", repository_id)

        # Get repository files
        query = select(File).where(
            and_(
                File.repository_id == repository_id,
                not File.is_deleted,
            ),
        )

        if limit:
            query = query.limit(limit)

        result = await self.db_session.execute(query)
        files = result.scalars().all()

        stats = {
            "repository_id": repository_id,
            "files_processed": 0,
            "total_embeddings": 0,
            "errors": [],
        }

        for file_record in files:
            try:
                file_stats = await self.create_file_embeddings(file_record.id)
                stats["files_processed"] += 1
                stats["total_embeddings"] += file_stats["total"]
                if file_stats["errors"]:
                    stats["errors"].extend(
                        [
                            f"File {file_record.path}: {err}"
                            for err in file_stats["errors"]
                        ],
                    )
            except Exception as e:
                logger.exception("Failed to process file %s", file_record.id)
                stats["errors"].append(f"File {file_record.path}: {e!s}")

        logger.info(
            "Created %s embeddings for %s files in repository %s",
            stats["total_embeddings"],
            stats["files_processed"],
            repository_id,
        )

        return stats

    async def _create_module_embedding(self, module: Module, file_path: str) -> int:
        """Create embedding for a module.

        Args:
            module: Module record
            file_path: Path to the source file

        Returns:
            Created embedding ID
        """
        # Check if embedding already exists
        existing = await self._get_existing_embedding("module", module.id)
        if existing:
            logger.debug("Embedding already exists for module %d", module.id)
            return existing.id

        # Prepare module data
        module_data = {
            "name": module.name,
            "docstring": module.docstring,
            "start_line": module.start_line,
            "end_line": module.end_line,
        }

        # Get module statistics
        stats = await self._get_module_stats(module.id)

        # Generate embedding
        result = await self.embedding_generator.generate_module_embedding(
            module_data,
            file_path,
            stats,
        )

        # Store embedding
        embedding = CodeEmbedding(
            entity_type="module",
            entity_id=module.id,
            file_id=module.file_id,
            embedding_type="interpreted",
            embedding=result["embedding"],
            content=result["text"],
            tokens=result.get("tokens"),
            repo_metadata=result["metadata"],
            created_at=datetime.now(tz=datetime.UTC),
        )

        self.db_session.add(embedding)
        await self.db_session.commit()

        return embedding.id

    async def _create_class_embedding(self, cls: Class, file_path: str) -> int:
        """Create embedding for a class.

        Args:
            cls: Class record
            file_path: Path to the source file

        Returns:
            Created embedding ID
        """
        # Check if embedding already exists
        existing = await self._get_existing_embedding("class", cls.id)
        if existing:
            logger.debug("Embedding already exists for class %d", cls.id)
            return existing.id

        # Prepare class data with methods
        methods = await self._get_class_methods(cls.id)

        class_data = {
            "name": cls.name,
            "docstring": cls.docstring,
            "base_classes": cls.base_classes,
            "decorators": cls.decorators,
            "is_abstract": cls.is_abstract,
            "start_line": cls.start_line,
            "end_line": cls.end_line,
            "methods": [
                {
                    "name": method.name,
                    "parameters": method.parameters,
                    "return_type": method.return_type,
                }
                for method in methods
            ],
        }

        # Generate embedding
        results = await self.embedding_generator.generate_class_embeddings(
            [class_data],
            file_path,
        )

        if not results or not results[0].get("embedding"):
            msg = "Failed to generate class embedding"
            raise EmbeddingError(msg)

        result = results[0]

        # Store embedding
        embedding = CodeEmbedding(
            entity_type="class",
            entity_id=cls.id,
            file_id=cls.module.file_id,
            embedding_type="interpreted",
            embedding=result["embedding"],
            content=result["text"],
            tokens=result.get("tokens"),
            repo_metadata=result["metadata"],
            created_at=datetime.now(tz=datetime.UTC),
        )

        self.db_session.add(embedding)
        await self.db_session.commit()

        return embedding.id

    async def _create_function_embedding(self, func: Function, file_path: str) -> int:
        """Create embedding for a function.

        Args:
            func: Function record
            file_path: Path to the source file

        Returns:
            Created embedding ID
        """
        # Check if embedding already exists
        existing = await self._get_existing_embedding("function", func.id)
        if existing:
            logger.debug("Embedding already exists for function %d", func.id)
            return existing.id

        # Prepare function data
        func_data = {
            "name": func.name,
            "parameters": func.parameters,
            "return_type": func.return_type,
            "docstring": func.docstring,
            "decorators": func.decorators,
            "is_async": func.is_async,
            "is_generator": func.is_generator,
            "is_property": func.is_property,
            "is_staticmethod": func.is_static,
            "is_classmethod": func.is_classmethod,
            "start_line": func.start_line,
            "end_line": func.end_line,
            "class_name": func.parent_class.name if func.class_id else None,
        }

        # Generate embedding
        results = await self.embedding_generator.generate_function_embeddings(
            [func_data],
            file_path,
        )

        if not results or not results[0].get("embedding"):
            msg = "Failed to generate function embedding"
            raise EmbeddingError(msg)

        result = results[0]

        # Store embedding
        embedding = CodeEmbedding(
            entity_type="function",
            entity_id=func.id,
            file_id=func.module.file_id,
            embedding_type="interpreted",
            embedding=result["embedding"],
            content=result["text"],
            tokens=result.get("tokens"),
            repo_metadata=result["metadata"],
            created_at=datetime.now(tz=datetime.UTC),
        )

        self.db_session.add(embedding)
        await self.db_session.commit()

        return embedding.id

    async def _get_existing_embedding(
        self,
        entity_type: str,
        entity_id: int,
    ) -> CodeEmbedding | None:
        """Check if embedding already exists.

        Args:
            entity_type: Type of entity
            entity_id: Entity ID

        Returns:
            Existing embedding or None
        """
        result = await self.db_session.execute(
            select(CodeEmbedding).where(
                and_(
                    CodeEmbedding.entity_type == entity_type,
                    CodeEmbedding.entity_id == entity_id,
                ),
            ),
        )
        return result.scalar_one_or_none()

    async def _get_file_modules(self, file_id: int) -> list[Module]:
        """Get modules for a file."""
        result = await self.db_session.execute(
            select(Module)
            .where(Module.file_id == file_id)
            .options(selectinload(Module.file)),
        )
        return result.scalars().all()

    async def _get_file_classes(self, file_id: int) -> list[Class]:
        """Get classes for a file."""
        result = await self.db_session.execute(
            select(Class)
            .join(Module)
            .where(Module.file_id == file_id)
            .options(selectinload(Class.module)),
        )
        return result.scalars().all()

    async def _get_file_functions(self, file_id: int) -> list[Function]:
        """Get functions for a file."""
        result = await self.db_session.execute(
            select(Function)
            .join(Module)
            .where(Module.file_id == file_id)
            .options(
                selectinload(Function.module),
                selectinload(Function.parent_class),
            ),
        )
        return result.scalars().all()

    async def _get_class_methods(self, class_id: int) -> list[Function]:
        """Get methods for a class."""
        result = await self.db_session.execute(
            select(Function).where(Function.class_id == class_id),
        )
        return result.scalars().all()

    async def _get_module_stats(self, module_id: int) -> dict[str, int]:
        """Get statistics for a module."""
        # Count classes
        class_result = await self.db_session.execute(
            select(func.count()).select_from(Class).where(Class.module_id == module_id),
        )
        class_count = class_result.scalar() or 0

        # Count functions
        func_result = await self.db_session.execute(
            select(func.count())
            .select_from(Function)
            .where(
                and_(
                    Function.module_id == module_id,
                    Function.class_id.is_(None),
                ),
            ),
        )
        func_count = func_result.scalar() or 0

        return {
            "classes": class_count,
            "functions": func_count,
        }

    async def delete_entity_embeddings(
        self,
        entity_type: str,
        entity_id: int,
    ) -> int:
        """Delete embeddings for an entity.

        Args:
            entity_type: Type of entity
            entity_id: Entity ID

        Returns:
            Number of deleted embeddings
        """
        result = await self.db_session.execute(
            select(CodeEmbedding)
            .where(
                and_(
                    CodeEmbedding.entity_type == entity_type,
                    CodeEmbedding.entity_id == entity_id,
                ),
            )
            .delete(),
        )

        await self.db_session.commit()
        return result.rowcount

    async def update_file_embeddings(self, file_id: int) -> dict[str, Any]:
        """Update embeddings for a file (delete and recreate).

        Args:
            file_id: File ID to update

        Returns:
            Summary of updated embeddings
        """
        logger.info("Updating embeddings for file %d", file_id)

        # Delete existing embeddings
        deleted = await self.db_session.execute(
            select(CodeEmbedding).where(CodeEmbedding.file_id == file_id).delete(),
        )
        await self.db_session.commit()

        logger.info("Deleted %d existing embeddings", deleted.rowcount)

        # Create new embeddings
        stats = await self.create_file_embeddings(file_id)
        stats["deleted"] = deleted.rowcount

        return stats

    async def create_domain_entity_embedding(
        self,
        entity_id: int,
    ) -> dict[str, Any]:
        """Create embedding for a domain entity.

        Args:
            entity_id: Domain entity ID

        Returns:
            Embedding creation result
        """
        logger.info("Creating embedding for domain entity %d", entity_id)

        # Get domain entity with relationships
        result = await self.db_session.execute(
            select(DomainEntity)
            .where(DomainEntity.id == entity_id)
            .options(selectinload(DomainEntity.bounded_contexts))
        )
        entity = result.scalar_one_or_none()

        if not entity:
            msg = f"Domain entity {entity_id} not found"
            raise NotFoundError(msg)

        # Prepare entity data
        entity_data = {
            "name": entity.name,
            "entity_type": entity.entity_type,
            "description": entity.description,
            "business_rules": entity.business_rules,
            "invariants": entity.invariants,
            "responsibilities": entity.responsibilities,
            "module_path": entity.module_path,
            "class_name": entity.class_name,
        }

        # Add bounded context info if available
        if entity.bounded_contexts:
            context_names = [bc.name for bc in entity.bounded_contexts]
            entity_data["bounded_context"] = ", ".join(context_names)

        # Generate embedding
        try:
            result = await self.embedding_generator.generate_domain_entity_embedding(
                entity_data
            )

            # Store in database
            embedding_record = CodeEmbedding(
                entity_type="domain_entity",
                entity_id=entity.id,
                text=result["text"],
                embedding=result["embedding"],
                metadata=result["metadata"],
                tokens=result["tokens"],
                model=self.embedding_generator.embeddings.model,
                created_at=datetime.now(tz=UTC),
            )

            self.db_session.add(embedding_record)
            await self.db_session.commit()

            return {
                "entity_id": entity.id,
                "entity_name": entity.name,
                "entity_type": entity.entity_type,
                "embedding_id": embedding_record.id,
                "tokens": result["tokens"],
                "status": "success",
            }

        except Exception as e:
            logger.exception(
                "Failed to create embedding for domain entity %s",
                entity.name,
            )
            return {
                "entity_id": entity.id,
                "entity_name": entity.name,
                "entity_type": entity.entity_type,
                "status": "failed",
                "error": str(e),
            }

    async def create_all_domain_entity_embeddings(self) -> dict[str, Any]:
        """Create embeddings for all domain entities.

        Returns:
            Summary of created embeddings
        """
        logger.info("Creating embeddings for all domain entities")

        # Get all domain entities
        result = await self.db_session.execute(
            select(DomainEntity).options(selectinload(DomainEntity.bounded_contexts))
        )
        entities = result.scalars().all()

        stats = {
            "total": len(entities),
            "success": 0,
            "failed": 0,
            "errors": [],
        }

        for entity in entities:
            result = await self.create_domain_entity_embedding(entity.id)
            if result["status"] == "success":
                stats["success"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append(
                    f"{entity.name}: {result.get('error', 'Unknown error')}"
                )

        return stats
