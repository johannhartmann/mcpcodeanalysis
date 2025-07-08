"""Indexer service main entry point."""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.database import get_repositories, get_session, init_database
from src.indexer.chunking import CodeChunker
from src.indexer.embeddings import EmbeddingGenerator
from src.indexer.interpreter import CodeInterpreter
from src.mcp_server.config import config
from src.parser.code_extractor import CodeExtractor
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class IndexerService:
    """Service for generating embeddings and indexing code."""
    
    def __init__(self) -> None:
        self.embedding_generator = EmbeddingGenerator()
        self.code_interpreter = CodeInterpreter()
        self.code_chunker = CodeChunker()
        self.code_extractor = CodeExtractor()
        self.running = False
        self.tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start the indexer service."""
        logger.info("Starting indexer service")
        
        # Initialize database
        await init_database()
        
        self.running = True
        
        # Start indexing task
        index_task = asyncio.create_task(self.run_indexing())
        self.tasks.append(index_task)
        
        logger.info("Indexer service started")
    
    async def stop(self) -> None:
        """Stop the indexer service."""
        logger.info("Stopping indexer service")
        
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("Indexer service stopped")
    
    async def run_indexing(self) -> None:
        """Main indexing loop."""
        while self.running:
            try:
                # Process unindexed entities
                await self.process_unindexed_entities()
                
                # Wait before next iteration
                await asyncio.sleep(config.indexing.update_interval)
                
            except Exception as e:
                logger.error(f"Error in indexing loop: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def process_unindexed_entities(self) -> None:
        """Process entities that don't have embeddings yet."""
        async with get_session() as session:
            async with get_repositories(session) as repos:
                # Get files without embeddings
                # This is simplified - in practice, we'd have a more sophisticated query
                files = await session.execute(
                    "SELECT f.* FROM files f "
                    "LEFT JOIN code_embeddings e ON e.entity_type = 'file' AND e.entity_id = f.id "
                    "WHERE e.id IS NULL LIMIT 100"
                )
                
                for file in files:
                    await self.index_file(repos, file)
    
    async def index_file(self, repos: Dict[str, Any], file: Any) -> None:
        """Index a single file."""
        try:
            file_path = Path(file.path)
            
            # Get repository info
            repo = await repos["repository"].get_by_id(file.repository_id)
            if not repo:
                return
            
            # Get full file path
            full_path = Path("./repositories") / repo.owner / repo.name / file.path
            
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                return
            
            # Extract entities
            entities = self.code_extractor.extract_from_file(full_path, file.id)
            if not entities:
                return
            
            # Read file content
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # Create chunks
            chunks = self.code_chunker.chunk_by_entity(entities, content)
            chunks = self.code_chunker.merge_small_chunks(chunks)
            
            # Process each chunk
            for chunk in chunks:
                await self.process_chunk(repos, file, chunk, full_path)
            
            logger.info(f"Indexed file: {file.path}")
            
        except Exception as e:
            logger.error(f"Error indexing file {file.path}: {e}")
    
    async def process_chunk(
        self,
        repos: Dict[str, Any],
        file: Any,
        chunk: Dict[str, Any],
        file_path: Path
    ) -> None:
        """Process a single code chunk."""
        try:
            chunk_type = chunk["type"]
            chunk_content = chunk["content"]
            metadata = chunk["metadata"]
            
            # Generate interpretation
            interpretation = None
            if chunk_type == "function" or chunk_type == "method":
                interpretation = await self.code_interpreter.interpret_function(
                    chunk_content,
                    metadata.get("entity_name", "unknown"),
                    metadata.get("parameters", []),
                    metadata.get("return_type"),
                    metadata.get("docstring")
                )
            elif chunk_type == "class":
                interpretation = await self.code_interpreter.interpret_class(
                    chunk_content,
                    metadata.get("entity_name", "unknown"),
                    metadata.get("base_classes", []),
                    metadata.get("docstring"),
                    [m["name"] for m in metadata.get("methods", [])]
                )
            elif chunk_type == "module":
                interpretation = await self.code_interpreter.interpret_module(
                    file_path.stem,
                    metadata.get("docstring"),
                    metadata.get("import_names", []),
                    metadata.get("class_names", []),
                    metadata.get("function_names", [])
                )
            
            # Generate embeddings
            raw_embedding, interpreted_embedding = await self.embedding_generator.generate_code_embeddings(
                chunk_content,
                interpretation,
                f"File: {file.path}"
            )
            
            # Map chunk type to entity type and ID
            entity_type, entity_id = self._map_chunk_to_entity(chunk, file.id)
            
            # Store embeddings
            embedding_data = []
            
            # Raw embedding
            embedding_data.append({
                "entity_type": entity_type,
                "entity_id": entity_id,
                "embedding_type": "raw",
                "embedding": raw_embedding,
                "content": chunk_content[:1000],  # Store truncated content
                "metadata": metadata,
                "tokens": self.embedding_generator.count_tokens(chunk_content),
            })
            
            # Interpreted embedding
            if interpreted_embedding is not None and interpretation:
                embedding_data.append({
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "embedding_type": "interpreted",
                    "embedding": interpreted_embedding,
                    "content": interpretation[:1000],
                    "metadata": metadata,
                    "tokens": self.embedding_generator.count_tokens(interpretation),
                })
            
            await repos["embedding"].create_batch(embedding_data)
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
    
    def _map_chunk_to_entity(
        self,
        chunk: Dict[str, Any],
        file_id: int
    ) -> Tuple[str, int]:
        """Map a chunk to an entity type and ID."""
        # This is simplified - in practice, we'd look up actual entity IDs
        chunk_type = chunk["type"]
        
        if chunk_type in ("function", "method"):
            return "function", file_id  # Should be actual function ID
        elif chunk_type == "class":
            return "class", file_id  # Should be actual class ID
        elif chunk_type == "module":
            return "module", file_id  # Should be actual module ID
        else:
            return "module", file_id


async def main() -> None:
    """Main entry point for indexer service."""
    # Set up logging
    setup_logging()
    
    # Create indexer service
    indexer = IndexerService()
    
    # Handle shutdown signals
    def signal_handler(sig, frame) -> None:
        logger.info(f"Received signal {sig}")
        asyncio.create_task(indexer.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start indexer
        await indexer.start()
        
        # Keep running until stopped
        while indexer.running:
            await asyncio.sleep(1)
    
    except Exception as e:
        logger.error(f"Indexer service error: {e}")
        sys.exit(1)
    finally:
        await indexer.stop()


if __name__ == "__main__":
    asyncio.run(main())