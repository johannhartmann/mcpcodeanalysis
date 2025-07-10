"""Code processing integration between scanner and parser."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.database.models import Class, File, Function, Import, Module
from src.domain.indexer import DomainIndexer
from src.logger import get_logger
from src.parser.code_extractor import CodeExtractor
from src.parser.parser_factory import ParserFactory

logger = get_logger(__name__)


class CodeProcessor:
    """Process code files to extract and store code entities."""

    def __init__(
        self,
        db_session: AsyncSession,
        *,
        repository_path: Path | str | None = None,
        enable_domain_analysis: bool = False,
    ) -> None:
        self.db_session = db_session
        self.repository_path = Path(repository_path) if repository_path else None
        self.code_extractor = CodeExtractor()
        self.parser_factory = ParserFactory()
        self.enable_domain_analysis = enable_domain_analysis
        self.domain_indexer = None

        if enable_domain_analysis:
            self.domain_indexer = DomainIndexer(db_session)

    async def process_file(self, file_record: File) -> dict[str, Any]:
        """Process a file to extract code entities."""
        logger.info("Processing file: %s", file_record.path)

        # Check if file is supported
        file_path = Path(file_record.path)

        # Get absolute path if repository path is set
        if self.repository_path:
            absolute_path = self.repository_path / file_path
        else:
            absolute_path = file_path

        if not self.parser_factory.is_supported(file_path):
            logger.debug("Skipping unsupported file type: %s", file_path)
            return {
                "file_id": file_record.id,
                "status": "skipped",
                "reason": "unsupported_file_type",
            }

        try:
            # Extract entities
            entities = await self._extract_entities(absolute_path, file_record.id)
            if not entities:
                return {
                    "file_id": file_record.id,
                    "status": "failed",
                    "reason": "extraction_failed",
                }

            # Store entities in database
            stats = await self._store_entities(entities, file_record)

            # Update file processing status
            file_record.last_modified = datetime.now(tz=UTC)
            await self.db_session.commit()

            # Run domain analysis if enabled
            domain_stats = {}
            if self.domain_indexer and file_path.suffix == ".py":
                try:
                    logger.info("Running domain analysis for %s", file_record.path)
                    domain_result = await self.domain_indexer.index_file(file_record.id)
                    domain_stats = {
                        "domain_entities": domain_result.get("entities_extracted", 0),
                        "domain_relationships": domain_result.get(
                            "relationships_extracted",
                            0,
                        ),
                    }
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(
                        "Domain analysis failed for %s: %s",
                        file_record.path,
                        e,
                    )

            return {
                "file_id": file_record.id,
                "status": "success",
                "statistics": stats,
                "domain_statistics": domain_stats,
            }

        except Exception as e:
            logger.exception("Error processing file %s", file_record.path)
            import traceback

            logger.debug("Traceback: %s", traceback.format_exc())

            # Update file with error
            await self.db_session.commit()

            return {
                "file_id": file_record.id,
                "status": "failed",
                "reason": "processing_error",
                "error": str(e),
            }

    async def _extract_entities(
        self,
        file_path: Path,
        file_id: int,
    ) -> dict[str, list[Any]] | None:
        """Extract code entities from a file."""
        # Run extraction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.code_extractor.extract_from_file,
            file_path,
            file_id,
        )

    async def _store_entities(
        self,
        entities: dict[str, list[Any]],
        file_record: File,
    ) -> dict[str, int]:
        """Store extracted entities in the database."""
        stats = {
            "modules": 0,
            "classes": 0,
            "functions": 0,
            "imports": 0,
        }

        # Clear existing entities for this file
        await self._clear_file_entities(file_record.id)

        # Store modules
        module_map = {}
        for module_data in entities.get("modules", []):
            module = Module(
                file_id=file_record.id,
                name=module_data["name"],
                docstring=module_data.get("docstring"),
                start_line=module_data["start_line"],
                end_line=module_data["end_line"],
            )
            self.db_session.add(module)
            await self.db_session.flush()  # Get ID
            module_map[module_data["name"]] = module.id
            stats["modules"] += 1

        # Store classes
        class_map = {}
        for class_data in entities.get("classes", []):
            class_obj = Class(
                module_id=module_map.get(file_record.path.split("/")[-1].split(".")[0]),
                name=class_data["name"],
                docstring=class_data.get("docstring"),
                base_classes=class_data.get("base_classes", []),
                decorators=class_data.get("decorators", []),
                start_line=class_data["start_line"],
                end_line=class_data["end_line"],
                is_abstract=class_data.get("is_abstract", False),
            )
            self.db_session.add(class_obj)
            await self.db_session.flush()  # Get ID
            class_map[class_data["name"]] = class_obj.id
            stats["classes"] += 1

        # Store functions and methods
        for func_data in entities.get("functions", []):
            class_name = func_data.get("class_name")
            function = Function(
                module_id=module_map.get(file_record.path.split("/")[-1].split(".")[0]),
                class_id=class_map.get(class_name) if class_name else None,
                name=func_data["name"],
                parameters=func_data.get("parameters", []),
                return_type=func_data.get("return_type"),
                docstring=func_data.get("docstring"),
                decorators=func_data.get("decorators", []),
                is_async=func_data.get("is_async", False),
                is_generator=func_data.get("is_generator", False),
                is_property=func_data.get("is_property", False),
                is_static=func_data.get("is_staticmethod", False),
                is_classmethod=func_data.get("is_classmethod", False),
                start_line=func_data["start_line"],
                end_line=func_data["end_line"],
                complexity=func_data.get("complexity", 1),
            )
            self.db_session.add(function)
            stats["functions"] += 1

        # Store imports
        for import_data in entities.get("imports", []):
            import_obj = Import(
                file_id=file_record.id,
                import_statement=import_data["import_statement"],
                module_name=import_data.get("imported_from"),
                imported_names=import_data.get("imported_names", []),
                is_relative=import_data.get("is_relative", False),
                level=import_data.get("level", 0),
                line_number=import_data["line_number"],
            )
            self.db_session.add(import_obj)
            stats["imports"] += 1

        await self.db_session.commit()
        return stats

    async def _clear_file_entities(self, file_id: int) -> None:
        """Clear existing entities for a file before re-parsing."""
        # Delete in correct order to respect foreign key constraints
        await self.db_session.execute(delete(Import).where(Import.file_id == file_id))
        # Functions are linked to modules, not files directly
        # First get modules for this file
        modules = await self.db_session.execute(
            select(Module).where(Module.file_id == file_id),
        )
        module_ids = [m.id for m in modules.scalars()]

        if module_ids:
            # Delete functions linked to these modules
            await self.db_session.execute(
                delete(Function).where(Function.module_id.in_(module_ids)),
            )
            # Delete classes linked to these modules
            await self.db_session.execute(
                delete(Class).where(Class.module_id.in_(module_ids)),
            )

        await self.db_session.execute(delete(Module).where(Module.file_id == file_id))
        await self.db_session.commit()

    async def process_files(self, file_records: list[File]) -> dict[str, Any]:
        """Process multiple files."""
        logger.info("Processing %s files", len(file_records))

        # Process files sequentially to avoid database session conflicts
        # TODO(@dev): Implement proper session management for parallel processing
        results = []
        for file in file_records:
            try:
                result = await self.process_file(file)
                results.append(result)
            except Exception as e:
                logger.exception("Error processing file %s", file.path)
                results.append(e)

        # Aggregate results
        summary = {
            "total": len(file_records),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "statistics": {
                "modules": 0,
                "classes": 0,
                "functions": 0,
                "imports": 0,
            },
        }

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                summary["failed"] += 1
                summary["errors"].append(
                    {
                        "file": file_records[i].path,
                        "error": str(result),
                    },
                )
            elif isinstance(result, dict):
                if result["status"] == "success":
                    summary["success"] += 1
                    # Aggregate statistics
                    for key, value in result.get("statistics", {}).items():
                        summary["statistics"][key] += value
                elif result["status"] == "skipped":
                    summary["skipped"] += 1
                else:
                    summary["failed"] += 1
                    if "error" in result:
                        summary["errors"].append(
                            {
                                "file": file_records[i].path,
                                "error": result["error"],
                            },
                        )

        logger.info(
            "Processing complete: %s success, %s failed, %s skipped",
            summary["success"],
            summary["failed"],
            summary["skipped"],
        )

        return summary

    async def get_file_structure(self, file_record: File) -> dict[str, Any]:
        """Get the parsed structure of a file."""
        structure = {
            "file": {
                "id": file_record.id,
                "path": file_record.path,
                "language": file_record.language,
            },
            "modules": [],
            "classes": [],
            "functions": [],
            "imports": [],
        }

        # Get modules
        modules = await self.db_session.execute(
            select(Module).where(Module.file_id == file_record.id),
        )
        structure["modules"] = [
            {
                "id": m.id,
                "name": m.name,
                "docstring": m.docstring,
                "lines": f"{m.start_line}-{m.end_line}",
            }
            for m in modules.scalars()
        ]

        # Get module IDs for related queries
        module_ids = [m["id"] for m in structure["modules"]]

        # Get classes through modules
        if module_ids:
            classes = await self.db_session.execute(
                select(Class).where(Class.module_id.in_(module_ids)),
            )
        else:
            classes = None
        structure["classes"] = [
            {
                "id": c.id,
                "name": c.name,
                "docstring": c.docstring,
                "base_classes": c.base_classes,
                "is_abstract": c.is_abstract,
                "lines": f"{c.start_line}-{c.end_line}",
            }
            for c in (classes.scalars() if classes else [])
        ]

        # Get functions through modules
        if module_ids:
            functions = await self.db_session.execute(
                select(Function).where(Function.module_id.in_(module_ids)),
            )
        else:
            functions = None
        structure["functions"] = [
            {
                "id": f.id,
                "name": f.name,
                "class_id": f.class_id,
                "parameters": f.parameters,
                "return_type": f.return_type,
                "is_async": f.is_async,
                "lines": f"{f.start_line}-{f.end_line}",
            }
            for f in (functions.scalars() if functions else [])
        ]

        # Get imports
        imports = await self.db_session.execute(
            select(Import).where(Import.file_id == file_record.id),
        )
        structure["imports"] = [
            {
                "id": i.id,
                "statement": i.import_statement,
                "from": i.module_name,
                "names": i.imported_names,
                "line": i.line_number,
            }
            for i in imports.scalars()
        ]

        return structure
