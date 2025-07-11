"""LLM-based domain entity extraction from code."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

MIN_ENTITIES_FOR_RELATIONSHIPS = 2


class DomainEntityExtractor:
    """Extract domain entities from code using LLM analysis."""

    def __init__(self, llm: Any = None) -> None:
        """Initialize the entity extractor.
        
        Args:
            llm: Optional LLM instance (for testing)
        """
        if llm is not None:
            self.llm = llm
        else:
            # Try to create from settings
            try:
                # Handle different ways settings might store the API key
                openai_key = None
                if hasattr(settings, "OPENAI_API_KEY"):
                    openai_key = settings.OPENAI_API_KEY
                elif hasattr(settings, "openai_api_key"):
                    if hasattr(settings.openai_api_key, "get_secret_value"):
                        openai_key = settings.openai_api_key.get_secret_value()
                    else:
                        openai_key = settings.openai_api_key

                if not openai_key:
                    raise ValueError("OpenAI API key not found")

                self.llm = ChatOpenAI(
                    openai_api_key=openai_key,
                    model=settings.llm.model,
                    temperature=settings.llm.temperature,
                )
            except (AttributeError, KeyError, ValueError) as e:
                logger.warning("Failed to initialize LLM: %s", e)
                self.llm = None

    async def extract_entities(
        self,
        code_chunk: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract domain entities from a code chunk.

        Args:
            code_chunk: The code to analyze
            context: Additional context (file path, surrounding code, etc.)

        Returns:
            Dictionary containing extracted entities and metadata
        """
        prompt = self._build_entity_extraction_prompt(code_chunk, context)

        try:
            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=prompt),
            ]

            response = await self.llm.ainvoke(
                messages,
                config={"configurable": {"response_format": {"type": "json_object"}}},
            )

            result = json.loads(response.content)
            return self._process_extraction_result(result, code_chunk)

        except Exception as e:
            logger.exception("Error extracting entities")
            return {
                "entities": [],
                "error": str(e),
            }

    async def extract_relationships(
        self,
        entities: list[dict[str, Any]],
        code_chunks: list[str],
    ) -> list[dict[str, Any]]:
        """Extract relationships between domain entities.

        Args:
            entities: List of extracted domain entities
            code_chunks: Code chunks containing evidence of relationships

        Returns:
            List of domain relationships
        """
        if not entities or len(entities) < MIN_ENTITIES_FOR_RELATIONSHIPS:
            return []

        prompt = self._build_relationship_extraction_prompt(entities, code_chunks)

        try:
            messages = [
                SystemMessage(content=self._get_relationship_system_prompt()),
                HumanMessage(content=prompt),
            ]

            response = await self.llm.ainvoke(
                messages,
                config={"configurable": {"response_format": {"type": "json_object"}}},
            )

            result = json.loads(response.content)
            return self._process_relationship_result(result)

        except Exception:
            logger.exception("Error extracting relationships")
            return []

    def _get_system_prompt(self) -> str:
        """Get the system prompt for entity extraction."""
        return """You are an expert in Domain-Driven Design (DDD) and software architecture.
Your task is to analyze code and extract domain entities, not technical implementation details.

Focus on identifying:
1. Business entities and concepts (not technical classes)
2. Domain events and commands
3. Business processes and workflows
4. Actors and roles in the system
5. Business rules and invariants
6. Value objects and domain services

Distinguish between:
- Domain entities: Core business concepts with identity and lifecycle
- Value objects: Immutable concepts defined by their attributes
- Domain services: Business operations that don't belong to a single entity
- Domain events: Things that happen in the business domain
- Commands: Actions that change the system state
- Queries: Requests for information
- Policies: Business rules and decision logic
- Factories: Complex object creation logic
- Repository interfaces: Domain-level data access abstractions

DO NOT extract:
- Technical infrastructure (controllers, DAOs, DTOs)
- Framework-specific code
- Pure technical utilities
- UI components
- Database schemas (unless they clearly represent domain concepts)

Output JSON with this structure:
{
  "entities": [
    {
      "name": "Order",
      "type": "aggregate_root",
      "description": "Customer purchase order",
      "business_rules": ["Order must have at least one item", "Total cannot be negative"],
      "invariants": ["Order status progression must be valid"],
      "responsibilities": ["Track order lifecycle", "Calculate totals"],
      "ubiquitous_language": {"fulfill": "Process and ship the order", "cancel": "Stop order processing"}
    }
  ],
  "confidence": 0.85,
  "reasoning": "Identified Order as central business concept..."
}"""

    def _get_relationship_system_prompt(self) -> str:
        """Get the system prompt for relationship extraction."""
        return """You are an expert in Domain-Driven Design analyzing relationships between domain entities.

Identify semantic relationships between entities, focusing on:
1. How entities collaborate to fulfill business processes
2. Data dependencies and flow
3. Command/event relationships
4. Aggregate boundaries
5. Domain service orchestration

Relationship types:
- uses: Entity uses another for its operations
- creates: Entity creates instances of another
- modifies: Entity changes state of another
- deletes: Entity removes another
- queries: Entity reads information from another
- validates: Entity validates another
- orchestrates: Service coordinates multiple entities
- implements: Concrete implementation of abstraction
- extends: Inheritance or extension relationship
- aggregates: Entity contains/manages others
- references: Weak reference without ownership
- publishes: Entity publishes events
- subscribes_to: Entity listens to events
- depends_on: General dependency
- composed_of: Composition relationship

Output JSON with this structure:
{
  "relationships": [
    {
      "source": "Order",
      "target": "Payment",
      "type": "orchestrates",
      "description": "Order initiates and tracks payment processing",
      "strength": 0.9,
      "evidence": ["processPayment method in Order class", "PaymentStatus field"],
      "interaction_patterns": ["Order.processPayment() creates Payment", "Order listens to PaymentCompleted event"],
      "data_flow": {"order_total": "Passed to Payment for processing"}
    }
  ],
  "aggregate_boundaries": [
    {
      "root": "Order",
      "members": ["OrderItem", "ShippingAddress"],
      "reasoning": "Order maintains consistency of its items and shipping info"
    }
  ]
}"""

    def _build_entity_extraction_prompt(
        self,
        code_chunk: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Build the prompt for entity extraction."""
        prompt_parts = []

        if context:
            if context.get("file_path"):
                prompt_parts.append(f"File: {context['file_path']}")
            if context.get("module_context"):
                prompt_parts.append(f"Module context: {context['module_context']}")

        prompt_parts.append("Analyze this code and extract domain entities:")
        prompt_parts.append(f"\n```python\n{code_chunk}\n```\n")
        prompt_parts.append(
            "Remember to focus on business concepts, not technical implementation.",
        )

        return "\n".join(prompt_parts)

    def _build_relationship_extraction_prompt(
        self,
        entities: list[dict[str, Any]],
        code_chunks: list[str],
    ) -> str:
        """Build the prompt for relationship extraction."""
        prompt_parts = []

        # List entities
        prompt_parts.append("Given these domain entities:")
        prompt_parts.extend(
            f"- {entity['name']} ({entity['type']}): {entity['description']}"
            for entity in entities
        )

        # Add code evidence
        prompt_parts.append("\nAnalyze this code to find relationships between them:")
        for i, chunk in enumerate(code_chunks[:5], 1):  # Limit to 5 chunks
            prompt_parts.append(f"\nCode chunk {i}:")
            prompt_parts.append(f"```python\n{chunk[:1000]}\n```")  # Limit chunk size

        prompt_parts.append(
            "\nIdentify how these entities interact and depend on each other.",
        )

        return "\n".join(prompt_parts)

    def _process_extraction_result(
        self,
        result: dict[str, Any],
        code_chunk: str,
    ) -> dict[str, Any]:
        """Process and validate extraction results."""
        entities = result.get("entities", [])

        # Validate and enrich entities
        processed_entities = []
        for entity in entities:
            # Ensure required fields
            if not entity.get("name") or not entity.get("type"):
                continue

            # Add defaults for missing fields
            entity.setdefault("description", "")
            entity.setdefault("business_rules", [])
            entity.setdefault("invariants", [])
            entity.setdefault("responsibilities", [])
            entity.setdefault("ubiquitous_language", {})

            # Add extraction metadata
            entity["confidence_score"] = result.get("confidence", 1.0)
            entity["source_code_sample"] = code_chunk[:500]  # Store sample

            processed_entities.append(entity)

        return {
            "entities": processed_entities,
            "confidence": result.get("confidence", 1.0),
            "reasoning": result.get("reasoning", ""),
        }

    def _process_relationship_result(
        self,
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Process and validate relationship results."""
        relationships = result.get("relationships", [])

        processed_relationships = []
        for rel in relationships:
            # Ensure required fields
            if not all(key in rel for key in ["source", "target", "type"]):
                continue

            # Add defaults
            rel.setdefault("description", "")
            rel.setdefault("strength", 1.0)
            rel.setdefault("evidence", [])
            rel.setdefault("interaction_patterns", [])
            rel.setdefault("data_flow", {})

            processed_relationships.append(rel)

        return processed_relationships

    async def extract_from_module(
        self,
        module_code: str,
        module_path: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> dict[str, Any]:
        """Extract entities from an entire module with chunking.

        Args:
            module_code: Complete module code
            module_path: Path to the module
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks

        Returns:
            Aggregated extraction results
        """
        chunks = self._create_semantic_chunks(module_code, chunk_size, overlap)

        all_entities = []
        all_relationships = []

        # Extract entities from each chunk
        for i, chunk in enumerate(chunks):
            logger.info(
                "Processing chunk %d/%d from %s",
                i + 1,
                len(chunks),
                module_path,
            )

            context = {
                "file_path": module_path,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }

            result = await self.extract_entities(chunk, context)
            all_entities.extend(result.get("entities", []))

        # Deduplicate entities
        unique_entities = self._deduplicate_entities(all_entities)

        # Extract relationships if we have multiple entities
        if len(unique_entities) > 1:
            relationships = await self.extract_relationships(unique_entities, chunks)
            all_relationships.extend(relationships)

        return {
            "entities": unique_entities,
            "relationships": all_relationships,
            "module_path": module_path,
            "chunks_processed": len(chunks),
        }

    def _create_semantic_chunks(
        self,
        code: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """Create semantic chunks that preserve code structure."""
        lines = code.split("\n")
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            # Start new chunk if current is too large
            if current_size + line_size > chunk_size and current_chunk:
                # Try to break at class or function boundaries
                chunk_text = "\n".join(current_chunk)
                chunks.append(chunk_text)

                # Keep overlap
                overlap_lines = []
                overlap_size = 0
                for overlap_line in reversed(current_chunk):
                    overlap_size += len(overlap_line) + 1
                    if overlap_size >= overlap:
                        break
                    overlap_lines.insert(0, overlap_line)

                current_chunk = overlap_lines
                current_size = overlap_size

            current_chunk.append(line)
            current_size += line_size

        # Add final chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _deduplicate_entities(
        self,
        entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Deduplicate entities by name and type."""
        seen = set()
        unique = []

        for entity in entities:
            key = (entity["name"], entity["type"])
            if key not in seen:
                seen.add(key)
                unique.append(entity)
            else:
                # Merge information from duplicates
                for existing in unique:
                    if (existing["name"], existing["type"]) == key:
                        # Merge lists
                        for field in [
                            "business_rules",
                            "invariants",
                            "responsibilities",
                        ]:
                            existing[field] = list(
                                set(existing.get(field, []) + entity.get(field, [])),
                            )
                        # Merge dictionaries
                        existing["ubiquitous_language"].update(
                            entity.get("ubiquitous_language", {}),
                        )
                        break

        return unique
