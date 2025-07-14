"""Interface designer service for creating module interfaces and contracts."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.models import Class, Module
from src.database.package_models import Package
from src.llm.client import LLMClient
from src.logger import get_logger

logger = get_logger(__name__)


class InterfaceDesigner:
    """Service for designing interfaces and contracts between modules."""

    def __init__(self, session: AsyncSession, llm_client: LLMClient | None = None):
        """Initialize the interface designer.

        Args:
            session: Database session
            llm_client: Optional LLM client for generation
        """
        self.session = session
        self.llm_client = llm_client

    async def design_module_interface(
        self, package_id: int, target_architecture: str = "modular_monolith"
    ) -> dict[str, Any]:
        """Design a clean interface for a module/package.

        Args:
            package_id: Package to design interface for
            target_architecture: Target architecture style

        Returns:
            Interface design including:
            - public_api: Public API definition
            - data_contracts: Data transfer objects
            - events: Domain events
            - dependencies: Required interfaces
            - implementation_notes: Guidelines
        """
        logger.info("Designing interface for package %d", package_id)

        # Get package with dependencies
        package = await self.session.get(
            Package,
            package_id,
            options=[
                selectinload(Package.modules),
                selectinload(Package.dependencies),
                selectinload(Package.dependents),
            ],
        )

        if not package:
            msg = f"Package {package_id} not found"
            raise ValueError(msg)

        # Analyze current interface
        current_interface = await self._analyze_current_interface(package)

        # Identify domain concepts
        domain_concepts = await self._identify_domain_concepts(package)

        # Design public API
        public_api = await self._design_public_api(
            package, current_interface, domain_concepts, target_architecture
        )

        # Design data contracts
        data_contracts = await self._design_data_contracts(
            package, domain_concepts, public_api
        )

        # Design events if event-driven
        events = []
        if target_architecture in ["microservices", "event_driven"]:
            events = await self._design_domain_events(package, domain_concepts)

        # Identify required dependencies
        dependencies = await self._identify_interface_dependencies(package)

        # Generate implementation notes
        implementation_notes = self._generate_implementation_notes(
            package, public_api, target_architecture
        )

        return {
            "package_id": package_id,
            "package_path": package.path,
            "interface_version": "1.0.0",
            "target_architecture": target_architecture,
            "public_api": public_api,
            "data_contracts": data_contracts,
            "events": events,
            "dependencies": dependencies,
            "implementation_notes": implementation_notes,
            "breaking_changes": self._identify_breaking_changes(
                current_interface, public_api
            ),
        }

    async def _analyze_current_interface(self, package: Package) -> dict[str, Any]:
        """Analyze the current public interface of a package.

        Args:
            package: Package to analyze

        Returns:
            Current interface analysis
        """
        public_classes = []
        public_functions = []
        exports = []

        # Get all modules in package
        for pm in package.modules:
            module = await self.session.get(
                Module,
                pm.file_id,
                options=[
                    selectinload(Module.classes).selectinload(Class.methods),
                    selectinload(Module.functions),
                ],
            )

            if not module:
                continue

            # Analyze public classes
            for cls in module.classes:
                if not cls.name.startswith("_"):  # Public class
                    class_info = {
                        "name": cls.name,
                        "module": module.name,
                        "methods": [],
                        "docstring": cls.docstring,
                    }

                    # Get public methods
                    for method in cls.methods:
                        if not method.name.startswith("_") or method.name in [
                            "__init__",
                            "__str__",
                            "__repr__",
                        ]:
                            class_info["methods"].append(
                                {
                                    "name": method.name,
                                    "parameters": method.parameters,
                                    "return_type": method.return_type,
                                    "docstring": method.docstring,
                                }
                            )

                    public_classes.append(class_info)

            # Analyze public functions
            public_functions.extend(
                {
                    "name": func.name,
                    "module": module.name,
                    "parameters": func.parameters,
                    "return_type": func.return_type,
                    "docstring": func.docstring,
                }
                for func in module.functions
                if not func.name.startswith("_")  # Public function
            )

            # Check __all__ exports
            if pm.exports:
                exports.extend(pm.exports)

        return {
            "public_classes": public_classes,
            "public_functions": public_functions,
            "exports": exports,
            "total_public_elements": len(public_classes) + len(public_functions),
        }

    async def _identify_domain_concepts(self, package: Package) -> list[dict[str, Any]]:
        """Identify domain concepts in the package.

        Args:
            package: Package to analyze

        Returns:
            List of domain concepts
        """
        # Query domain entities related to this package
        # This is simplified - in practice would use the source_entities field
        concepts = []

        # Analyze class names and docstrings for domain concepts
        for pm in package.modules:
            module = await self.session.get(
                Module,
                pm.file_id,
                options=[selectinload(Module.classes)],
            )

            if not module:
                continue

            for cls in module.classes:
                if cls.docstring and not cls.name.startswith("_"):
                    # Extract potential domain concept
                    concept = {
                        "name": cls.name,
                        "type": self._classify_domain_concept(cls.name, cls.docstring),
                        "description": (
                            cls.docstring.split("\n")[0] if cls.docstring else ""
                        ),
                        "source_class": cls.name,
                    }
                    concepts.append(concept)

        return concepts

    def _classify_domain_concept(self, name: str, docstring: str) -> str:
        """Classify a domain concept based on name and docstring.

        Args:
            name: Class name
            docstring: Class docstring

        Returns:
            Domain concept type
        """
        name_lower = name.lower()
        doc_lower = (docstring or "").lower()

        if "service" in name_lower:
            return "domain_service"
        if "repository" in name_lower:
            return "repository_interface"
        if "factory" in name_lower:
            return "factory"
        if "event" in name_lower or "event" in doc_lower:
            return "domain_event"
        if "command" in name_lower:
            return "command"
        if "query" in name_lower:
            return "query"
        if any(term in doc_lower for term in ["entity", "aggregate", "domain object"]):
            return "entity"
        if "value" in doc_lower and "object" in doc_lower:
            return "value_object"
        return "entity"  # Default

    async def _design_public_api(
        self,
        _package: Package,
        current_interface: dict[str, Any],
        domain_concepts: list[dict[str, Any]],
        target_architecture: str,
    ) -> dict[str, Any]:
        """Design a clean public API for the package.

        Args:
            package: Package to design for
            current_interface: Current interface analysis
            domain_concepts: Identified domain concepts
            target_architecture: Target architecture

        Returns:
            Public API design
        """
        api = {
            "services": [],
            "repositories": [],
            "factories": [],
            "commands": [],
            "queries": [],
        }

        # Group current interface by domain concept
        for cls in current_interface["public_classes"]:
            concept_type = self._classify_domain_concept(
                cls["name"], cls.get("docstring", "")
            )

            if concept_type == "domain_service":
                # Design service interface
                service = {
                    "name": f"I{cls['name']}",  # Interface name
                    "implementation": cls["name"],
                    "methods": [],
                }

                for method in cls["methods"]:
                    if (
                        not method["name"].startswith("_")
                        and method["name"] != "__init__"
                    ):
                        service["methods"].append(
                            {
                                "name": method["name"],
                                "parameters": self._clean_parameters(
                                    method["parameters"]
                                ),
                                "return_type": method["return_type"] or "Any",
                                "description": method.get("docstring", ""),
                            }
                        )

                api["services"].append(service)

            elif concept_type == "repository_interface":
                # Design repository interface
                repo = {
                    "name": f"I{cls['name']}",
                    "implementation": cls["name"],
                    "entity": self._extract_entity_from_repo(cls["name"]),
                    "methods": self._design_repository_methods(cls),
                }
                api["repositories"].append(repo)

        # Add CQRS patterns if appropriate
        if target_architecture in ["microservices", "cqrs"]:
            api["commands"] = self._design_commands(domain_concepts)
            api["queries"] = self._design_queries(domain_concepts)

        # Add facade if interface is complex
        if len(api["services"]) > 3:
            api["facade"] = self._design_facade(api["services"])

        return api

    async def _design_data_contracts(
        self,
        _package: Package,
        domain_concepts: list[dict[str, Any]],
        public_api: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Design data transfer objects and contracts.

        Args:
            package: Package to design for
            domain_concepts: Domain concepts
            public_api: Public API design

        Returns:
            List of data contracts
        """
        contracts = []

        # Create DTOs for each service method
        for service in public_api.get("services", []):
            for method in service["methods"]:
                # Request DTO
                if len(method["parameters"]) > 2:
                    request_dto = {
                        "name": f"{method['name'].title()}Request",
                        "type": "request_dto",
                        "fields": [
                            {
                                "name": param["name"],
                                "type": param["type"],
                                "required": not param.get("default"),
                                "description": "",
                            }
                            for param in method["parameters"]
                            if param["name"] != "self"
                        ],
                    }
                    contracts.append(request_dto)

                # Response DTO
                if method["return_type"] and method["return_type"] not in [
                    "None",
                    "bool",
                    "str",
                    "int",
                    "float",
                ]:
                    response_dto = {
                        "name": f"{method['name'].title()}Response",
                        "type": "response_dto",
                        "fields": self._infer_response_fields(method),
                    }
                    contracts.append(response_dto)

        # Create entity DTOs
        for concept in domain_concepts:
            if concept["type"] in ["entity", "aggregate_root"]:
                entity_dto = {
                    "name": f"{concept['name']}DTO",
                    "type": "entity_dto",
                    "source_entity": concept["name"],
                    "fields": self._infer_entity_fields(concept),
                }
                contracts.append(entity_dto)

        return contracts

    async def _design_domain_events(
        self, _package: Package, domain_concepts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Design domain events for event-driven architecture.

        Args:
            package: Package to design for
            domain_concepts: Domain concepts

        Returns:
            List of domain events
        """
        events = []

        # Create events for state changes in entities
        for concept in domain_concepts:
            if concept["type"] in ["entity", "aggregate_root"]:
                # Common events
                base_events = ["Created", "Updated", "Deleted"]

                for event_type in base_events:
                    event = {
                        "name": f"{concept['name']}{event_type}",
                        "type": "domain_event",
                        "aggregate": concept["name"],
                        "description": f"Raised when {concept['name']} is {event_type.lower()}",
                        "payload": {
                            "aggregate_id": "str",
                            "timestamp": "datetime",
                            "user_id": "str | None",
                        },
                    }

                    if event_type == "Created":
                        event["payload"]["data"] = f"{concept['name']}DTO"
                    elif event_type == "Updated":
                        event["payload"]["changes"] = "dict[str, Any]"
                        event["payload"]["previous_values"] = "dict[str, Any]"

                    events.append(event)

        return events

    async def _identify_interface_dependencies(
        self, package: Package
    ) -> list[dict[str, Any]]:
        """Identify required interface dependencies.

        Args:
            package: Package to analyze

        Returns:
            List of required interfaces
        """
        dependencies = []

        # Analyze package dependencies
        for dep in package.dependencies:
            target_package = await self.session.get(Package, dep.target_package_id)

            if target_package:
                # Determine interface type needed
                dep_interface = {
                    "package": target_package.path,
                    "interfaces": [],
                    "purpose": self._infer_dependency_purpose(dep.import_details),
                }

                # Analyze imports to determine specific interfaces
                for import_detail in dep.import_details or []:
                    if "Repository" in import_detail:
                        dep_interface["interfaces"].append(
                            {
                                "type": "repository",
                                "name": import_detail,
                            }
                        )
                    elif "Service" in import_detail:
                        dep_interface["interfaces"].append(
                            {
                                "type": "service",
                                "name": import_detail,
                            }
                        )

                dependencies.append(dep_interface)

        return dependencies

    def _generate_implementation_notes(
        self,
        _package: Package,
        _public_api: dict[str, Any],
        target_architecture: str,
    ) -> dict[str, Any]:
        """Generate implementation guidelines.

        Args:
            package: Package being designed
            public_api: Public API design
            target_architecture: Target architecture

        Returns:
            Implementation notes
        """
        notes = {
            "principles": [],
            "patterns": [],
            "antipatterns": [],
            "testing": {},
            "documentation": {},
        }

        # Architecture-specific principles
        if target_architecture == "modular_monolith":
            notes["principles"] = [
                "Use dependency injection for all service dependencies",
                "Keep module boundaries clear - no direct database access across modules",
                "Use interfaces for all inter-module communication",
                "Implement module-specific configuration",
            ]
            notes["patterns"] = [
                "Repository pattern for data access",
                "Service layer for business logic",
                "DTO pattern for data transfer",
                "Factory pattern for complex object creation",
            ]

        elif target_architecture == "microservices":
            notes["principles"] = [
                "Design for failure - implement circuit breakers",
                "Keep services autonomous",
                "Use async communication where possible",
                "Implement idempotency for all operations",
            ]
            notes["patterns"] = [
                "API Gateway pattern",
                "Event sourcing for state changes",
                "Saga pattern for distributed transactions",
                "CQRS for read/write separation",
            ]

        # Common antipatterns
        notes["antipatterns"] = [
            "Avoid shared database access",
            "Don't expose internal data models",
            "Prevent circular dependencies",
            "No synchronous cascading calls",
        ]

        # Testing guidelines
        notes["testing"] = {
            "unit_tests": "Test each service method in isolation",
            "integration_tests": "Test module interactions through interfaces",
            "contract_tests": "Verify interface contracts are maintained",
            "performance_tests": "Benchmark critical operations",
        }

        # Documentation requirements
        notes["documentation"] = {
            "api_docs": "Document all public interfaces with examples",
            "architecture_docs": "Maintain module interaction diagrams",
            "runbook": "Create operational runbook for the module",
        }

        return notes

    def _identify_breaking_changes(
        self, current_interface: dict[str, Any], new_api: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Identify breaking changes between current and new interface.

        Args:
            current_interface: Current interface
            new_api: New API design

        Returns:
            List of breaking changes
        """
        changes = []

        # Check removed methods
        current_methods = set()
        for cls in current_interface["public_classes"]:
            for method in cls["methods"]:
                current_methods.add(f"{cls['name']}.{method['name']}")

        new_methods = set()
        for service in new_api.get("services", []):
            impl_name = service["implementation"]
            for method in service["methods"]:
                new_methods.add(f"{impl_name}.{method['name']}")

        # Removed methods
        removed = current_methods - new_methods
        changes.extend(
            {
                "type": "removed_method",
                "method": method,
                "severity": "breaking",
                "migration": "Deprecate first, then remove in next major version",
            }
            for method in removed
        )

        return changes

    def _clean_parameters(
        self, parameters: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Clean and standardize parameter definitions.

        Args:
            parameters: Raw parameter list

        Returns:
            Cleaned parameters
        """
        cleaned = []
        for param in parameters:
            if param.get("name") == "self":
                continue

            cleaned.append(
                {
                    "name": param.get("name", ""),
                    "type": param.get("type", "Any"),
                    "default": param.get("default"),
                    "required": not bool(param.get("default")),
                }
            )

        return cleaned

    def _extract_entity_from_repo(self, repo_name: str) -> str:
        """Extract entity name from repository name.

        Args:
            repo_name: Repository class name

        Returns:
            Entity name
        """
        # Remove common suffixes
        for suffix in ["Repository", "Repo", "Store", "DAO"]:
            if repo_name.endswith(suffix):
                return repo_name[: -len(suffix)]

        return repo_name

    def _design_repository_methods(
        self, repo_class: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Design standard repository methods.

        Args:
            repo_class: Repository class info

        Returns:
            Repository methods
        """
        entity = self._extract_entity_from_repo(repo_class["name"])

        # Standard repository methods
        methods = [
            {
                "name": "find_by_id",
                "parameters": [{"name": "id", "type": "str", "required": True}],
                "return_type": f"{entity} | None",
                "description": f"Find {entity} by ID",
            },
            {
                "name": "find_all",
                "parameters": [
                    {"name": "limit", "type": "int", "default": 100},
                    {"name": "offset", "type": "int", "default": 0},
                ],
                "return_type": f"list[{entity}]",
                "description": f"Find all {entity} entities with pagination",
            },
            {
                "name": "save",
                "parameters": [{"name": "entity", "type": entity, "required": True}],
                "return_type": entity,
                "description": f"Save {entity} entity",
            },
            {
                "name": "delete",
                "parameters": [{"name": "id", "type": "str", "required": True}],
                "return_type": "bool",
                "description": f"Delete {entity} by ID",
            },
        ]

        # Add existing methods that follow patterns
        methods.extend(
            {
                "name": method["name"],
                "parameters": self._clean_parameters(method["parameters"]),
                "return_type": method.get("return_type", f"list[{entity}]"),
                "description": method.get("docstring", ""),
            }
            for method in repo_class.get("methods", [])
            if method["name"].startswith("find_by_") and method["name"] != "find_by_id"
        )

        return methods

    def _design_commands(
        self, domain_concepts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Design command objects for CQRS pattern.

        Args:
            domain_concepts: Domain concepts

        Returns:
            List of commands
        """
        commands = []

        for concept in domain_concepts:
            if concept["type"] in ["entity", "aggregate_root"]:
                # Standard commands
                for action in ["Create", "Update", "Delete"]:
                    command = {
                        "name": f"{action}{concept['name']}Command",
                        "type": "command",
                        "aggregate": concept["name"],
                        "fields": self._design_command_fields(concept, action),
                        "handler": f"{action}{concept['name']}CommandHandler",
                    }
                    commands.append(command)

        return commands

    def _design_queries(
        self, domain_concepts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Design query objects for CQRS pattern.

        Args:
            domain_concepts: Domain concepts

        Returns:
            List of queries
        """
        queries = []

        for concept in domain_concepts:
            if concept["type"] in ["entity", "aggregate_root"]:
                # Standard queries
                queries.extend(
                    [
                        {
                            "name": f"Get{concept['name']}ByIdQuery",
                            "type": "query",
                            "parameters": [{"name": "id", "type": "str"}],
                            "return_type": f"{concept['name']}DTO",
                            "handler": f"Get{concept['name']}ByIdQueryHandler",
                        },
                        {
                            "name": f"List{concept['name']}Query",
                            "type": "query",
                            "parameters": [
                                {"name": "filter", "type": "dict[str, Any] | None"},
                                {"name": "limit", "type": "int"},
                                {"name": "offset", "type": "int"},
                            ],
                            "return_type": f"list[{concept['name']}DTO]",
                            "handler": f"List{concept['name']}QueryHandler",
                        },
                    ]
                )

        return queries

    def _design_facade(self, services: list[dict[str, Any]]) -> dict[str, Any]:
        """Design a facade interface for complex modules.

        Args:
            services: List of services

        Returns:
            Facade design
        """
        facade = {
            "name": "ModuleFacade",
            "description": "Simplified interface for module operations",
            "services": [s["name"] for s in services],
            "methods": [],
        }

        # Aggregate common operations
        for service in services:
            # Pick most important methods
            important_methods = [
                m
                for m in service["methods"]
                if (not m["name"].startswith("_") and "create" in m["name"].lower())
                or "get" in m["name"].lower()
            ][
                :2
            ]  # Limit to 2 per service

            for method in important_methods:
                facade["methods"].append(
                    {
                        "name": method["name"],
                        "delegates_to": f"{service['name']}.{method['name']}",
                        "parameters": method["parameters"],
                        "return_type": method["return_type"],
                    }
                )

        return facade

    def _infer_response_fields(self, _method: dict[str, Any]) -> list[dict[str, Any]]:
        """Infer response fields from method signature.

        Args:
            method: Method information

        Returns:
            List of fields
        """
        # This is a simplified implementation
        # In practice, would analyze the return type annotation
        return [
            {"name": "success", "type": "bool", "required": True},
            {"name": "data", "type": "Any", "required": False},
            {"name": "error", "type": "str | None", "required": False},
        ]

    def _infer_entity_fields(self, _concept: dict[str, Any]) -> list[dict[str, Any]]:
        """Infer entity fields from domain concept.

        Args:
            concept: Domain concept

        Returns:
            List of fields
        """
        # Basic fields common to all entities
        return [
            {"name": "id", "type": "str", "required": True},
            {"name": "created_at", "type": "datetime", "required": True},
            {"name": "updated_at", "type": "datetime", "required": True},
        ]

        # Add concept-specific fields based on name
        # This is simplified - in practice would analyze the actual class

    def _infer_dependency_purpose(self, import_details: list[str] | None) -> str:
        """Infer the purpose of a dependency from imports.

        Args:
            import_details: List of imported items

        Returns:
            Purpose description
        """
        if not import_details:
            return "general_dependency"

        purposes = []
        for detail in import_details:
            if "Repository" in detail:
                purposes.append("data_access")
            elif "Service" in detail:
                purposes.append("business_logic")
            elif "Client" in detail:
                purposes.append("external_integration")
            elif "Util" in detail or "Helper" in detail:
                purposes.append("utilities")

        return ", ".join(set(purposes)) if purposes else "general_dependency"

    async def generate_interface_documentation(
        self, interface_design: dict[str, Any]
    ) -> str:
        """Generate markdown documentation for an interface design.

        Args:
            interface_design: Interface design from design_module_interface

        Returns:
            Markdown documentation
        """
        doc_lines = []

        # Header
        doc_lines.append(
            f"# {interface_design['package_path']} Interface Specification"
        )
        doc_lines.append(f"\nVersion: {interface_design['interface_version']}")
        doc_lines.append(f"Architecture: {interface_design['target_architecture']}")
        doc_lines.append("")

        # Table of Contents
        doc_lines.append("## Table of Contents")
        doc_lines.append("1. [Public API](#public-api)")
        doc_lines.append("2. [Data Contracts](#data-contracts)")
        if interface_design.get("events"):
            doc_lines.append("3. [Domain Events](#domain-events)")
        doc_lines.append("4. [Dependencies](#dependencies)")
        doc_lines.append("5. [Implementation Notes](#implementation-notes)")
        doc_lines.append("")

        # Public API
        doc_lines.append("## Public API")
        doc_lines.append("")

        api = interface_design["public_api"]

        # Services
        if api.get("services"):
            doc_lines.append("### Services")
            for service in api["services"]:
                doc_lines.append(f"\n#### {service['name']}")
                doc_lines.append(f"Implementation: `{service['implementation']}`")
                doc_lines.append("\nMethods:")
                for method in service["methods"]:
                    params = ", ".join(
                        f"{p['name']}: {p['type']}" for p in method["parameters"]
                    )
                    doc_lines.append(
                        f"- `{method['name']}({params}) -> {method['return_type']}`"
                    )
                    if method.get("description"):
                        doc_lines.append(f"  - {method['description']}")

        # Repositories
        if api.get("repositories"):
            doc_lines.append("\n### Repositories")
            for repo in api["repositories"]:
                doc_lines.append(f"\n#### {repo['name']}")
                doc_lines.append(f"Entity: `{repo['entity']}`")
                doc_lines.append("\nMethods:")
                for method in repo["methods"]:
                    params = ", ".join(
                        f"{p['name']}: {p['type']}" for p in method["parameters"]
                    )
                    doc_lines.append(
                        f"- `{method['name']}({params}) -> {method['return_type']}`"
                    )

        # Data Contracts
        doc_lines.append("\n## Data Contracts")
        doc_lines.append("")

        for contract in interface_design["data_contracts"]:
            doc_lines.append(f"### {contract['name']}")
            doc_lines.append(f"Type: {contract['type']}")
            doc_lines.append("\nFields:")
            for field in contract["fields"]:
                required = "required" if field.get("required", True) else "optional"
                doc_lines.append(f"- `{field['name']}: {field['type']}` ({required})")

        # Events
        if interface_design.get("events"):
            doc_lines.append("\n## Domain Events")
            doc_lines.append("")

            for event in interface_design["events"]:
                doc_lines.append(f"### {event['name']}")
                doc_lines.append(f"Aggregate: {event['aggregate']}")
                doc_lines.append(f"Description: {event['description']}")
                doc_lines.append("\nPayload:")
                for field, field_type in event["payload"].items():
                    doc_lines.append(f"- `{field}: {field_type}`")

        # Dependencies
        doc_lines.append("\n## Dependencies")
        doc_lines.append("")

        for dep in interface_design["dependencies"]:
            doc_lines.append(f"### {dep['package']}")
            doc_lines.append(f"Purpose: {dep['purpose']}")
            if dep["interfaces"]:
                doc_lines.append("Required interfaces:")
                doc_lines.extend(
                    f"- {iface['type']}: `{iface['name']}`"
                    for iface in dep["interfaces"]
                )

        # Implementation Notes
        doc_lines.append("\n## Implementation Notes")
        doc_lines.append("")

        notes = interface_design["implementation_notes"]

        doc_lines.append("### Principles")
        doc_lines.extend(f"- {principle}" for principle in notes["principles"])

        doc_lines.append("\n### Patterns")
        doc_lines.extend(f"- {pattern}" for pattern in notes["patterns"])

        doc_lines.append("\n### Anti-patterns to Avoid")
        doc_lines.extend(f"- {antipattern}" for antipattern in notes["antipatterns"])

        return "\n".join(doc_lines)
