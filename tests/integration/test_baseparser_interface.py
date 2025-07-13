"""Test BaseParser interface with TypeScript and JavaScript parsers."""

import tempfile
from pathlib import Path

import pytest

from src.parser.base_parser import ElementType, ParseResult
from src.parser.plugin_registry import LanguagePluginRegistry


class TestBaseParserInterface:
    """Test that TypeScript and JavaScript parsers properly implement BaseParser interface."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_typescript_baseparser_interface(self, temp_dir):
        """Test TypeScript parser implements BaseParser interface correctly."""
        plugin = LanguagePluginRegistry.get_plugin("typescript")

        try:
            parser = plugin.create_parser()

            # Create a complex TypeScript file
            ts_file = temp_dir / "complex.ts"
            ts_content = """
import { EventEmitter } from 'events';
import axios, { AxiosResponse } from 'axios';

interface ApiResponse<T> {
    data: T;
    status: number;
    message?: string;
}

interface User {
    id: number;
    name: string;
    email: string;
    roles: Role[];
}

interface Role {
    id: number;
    name: string;
    permissions: string[];
}

abstract class BaseService {
    protected apiUrl: string;

    constructor(apiUrl: string) {
        this.apiUrl = apiUrl;
    }

    abstract getData(): Promise<any>;
}

class UserService extends BaseService implements EventEmitter {
    private users: Map<number, User> = new Map();

    constructor(apiUrl: string) {
        super(apiUrl);
    }

    async fetchUser(id: number): Promise<User | null> {
        try {
            const response: AxiosResponse<ApiResponse<User>> = await axios.get(`${this.apiUrl}/users/${id}`);
            const user = response.data.data;
            this.users.set(user.id, user);
            return user;
        } catch (error) {
            console.error('Failed to fetch user:', error);
            return null;
        }
    }

    async createUser(userData: Omit<User, 'id'>): Promise<User> {
        const response = await axios.post<ApiResponse<User>>(`${this.apiUrl}/users`, userData);
        const newUser = response.data.data;
        this.users.set(newUser.id, newUser);
        return newUser;
    }

    getUserById(id: number): User | undefined {
        return this.users.get(id);
    }

    async getData(): Promise<User[]> {
        return Array.from(this.users.values());
    }
}

// Generic utility function
function processApiResponse<T>(response: ApiResponse<T>): T | null {
    if (response.status >= 200 && response.status < 300) {
        return response.data;
    }
    return null;
}

// Arrow function with type constraints
const filterUsersByRole = <T extends User>(users: T[], roleName: string): T[] => {
    return users.filter(user =>
        user.roles.some(role => role.name === roleName)
    );
};

export { UserService, User, Role, ApiResponse };
export default processApiResponse;
"""
            ts_file.write_text(ts_content)

            # Test BaseParser interface
            result = parser.parse_file(ts_file)

            # Verify ParseResult structure
            assert isinstance(result, ParseResult)
            assert result.file_path == ts_file
            assert result.language == "typescript"
            assert result.success is True
            assert len(result.errors) == 0

            # Verify root element
            assert result.root_element.type == ElementType.MODULE
            assert result.root_element.name == "complex"

            # Verify imports are extracted
            assert len(result.imports) > 0
            import_strings = [imp for imp in result.imports]
            assert any("EventEmitter" in imp for imp in import_strings)
            assert any("axios" in imp for imp in import_strings)

            # Verify dependencies are extracted
            assert len(result.dependencies) > 0
            assert "events" in result.dependencies
            assert "axios" in result.dependencies

            # Verify elements are extracted
            classes = result.find_elements(ElementType.CLASS)
            assert len(classes) >= 2  # BaseService, UserService

            class_names = [cls.name for cls in classes]
            assert "BaseService" in class_names
            assert "UserService" in class_names

            # Verify functions are extracted
            functions = result.find_elements(ElementType.FUNCTION)
            assert len(functions) >= 2  # processApiResponse, filterUsersByRole

            function_names = [func.name for func in functions]
            assert "processApiResponse" in function_names
            assert "filterUsersByRole" in function_names

            # Verify methods are extracted
            methods = result.find_elements(ElementType.METHOD)
            assert (
                len(methods) >= 4
            )  # constructor, fetchUser, createUser, getUserById, getData

            # Verify references are extracted
            assert isinstance(result.references, list)
            # Should have references for inheritance, imports, type usage, etc.

            # Verify metadata is populated
            user_service = next(
                (cls for cls in classes if cls.name == "UserService"), None
            )
            assert user_service is not None
            assert "base_classes" in user_service.metadata
            assert "BaseService" in user_service.metadata["base_classes"]

        except ImportError:
            pytest.skip("tree-sitter-typescript not available")

    def test_javascript_baseparser_interface(self, temp_dir):
        """Test JavaScript parser implements BaseParser interface correctly."""
        plugin = LanguagePluginRegistry.get_plugin("javascript")

        try:
            parser = plugin.create_parser()

            # Create a complex JavaScript file
            js_file = temp_dir / "complex.js"
            js_content = """
const EventEmitter = require('events');
const axios = require('axios');

class BaseService {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
    }

    async getData() {
        throw new Error('Not implemented');
    }
}

class UserService extends BaseService {
    constructor(apiUrl) {
        super(apiUrl);
        this.users = new Map();
        this.eventEmitter = new EventEmitter();
    }

    async fetchUser(id) {
        try {
            const response = await axios.get(`${this.apiUrl}/users/${id}`);
            const user = response.data.data;
            this.users.set(user.id, user);
            this.eventEmitter.emit('userFetched', user);
            return user;
        } catch (error) {
            console.error('Failed to fetch user:', error);
            this.eventEmitter.emit('error', error);
            return null;
        }
    }

    async createUser(userData) {
        const response = await axios.post(`${this.apiUrl}/users`, userData);
        const newUser = response.data.data;
        this.users.set(newUser.id, newUser);
        this.eventEmitter.emit('userCreated', newUser);
        return newUser;
    }

    getUserById(id) {
        return this.users.get(id);
    }

    async getData() {
        return Array.from(this.users.values());
    }

    static createInstance(apiUrl) {
        return new UserService(apiUrl);
    }
}

// Regular function
function processApiResponse(response) {
    if (response.status >= 200 && response.status < 300) {
        return response.data;
    }
    return null;
}

// Arrow function
const filterUsersByRole = (users, roleName) => {
    return users.filter(user =>
        user.roles && user.roles.some(role => role.name === roleName)
    );
};

// Async arrow function
const fetchAllUsers = async (apiUrl) => {
    const service = new UserService(apiUrl);
    return await service.getData();
};

module.exports = { UserService, processApiResponse, filterUsersByRole, fetchAllUsers };
"""
            js_file.write_text(js_content)

            # Test BaseParser interface
            result = parser.parse_file(js_file)

            # Verify ParseResult structure
            assert isinstance(result, ParseResult)
            assert result.file_path == js_file
            assert result.language == "javascript"
            assert result.success is True
            assert len(result.errors) == 0

            # Verify root element
            assert result.root_element.type == ElementType.MODULE
            assert result.root_element.name == "complex"

            # Verify imports are extracted
            assert len(result.imports) > 0
            import_strings = [imp for imp in result.imports]
            assert any("events" in imp for imp in import_strings)
            assert any("axios" in imp for imp in import_strings)

            # Verify elements are extracted
            classes = result.find_elements(ElementType.CLASS)
            assert len(classes) >= 2  # BaseService, UserService

            class_names = [cls.name for cls in classes]
            assert "BaseService" in class_names
            assert "UserService" in class_names

            # Verify functions are extracted
            functions = result.find_elements(ElementType.FUNCTION)
            assert (
                len(functions) >= 3
            )  # processApiResponse, filterUsersByRole, fetchAllUsers

            function_names = [func.name for func in functions]
            assert "processApiResponse" in function_names
            assert "filterUsersByRole" in function_names
            assert "fetchAllUsers" in function_names

            # Verify methods are extracted
            methods = result.find_elements(ElementType.METHOD)
            assert (
                len(methods) >= 5
            )  # constructor, fetchUser, createUser, getUserById, getData

            # Verify references are extracted
            assert isinstance(result.references, list)

            # Verify metadata is populated
            user_service = next(
                (cls for cls in classes if cls.name == "UserService"), None
            )
            assert user_service is not None
            assert "base_classes" in user_service.metadata
            assert "BaseService" in user_service.metadata["base_classes"]

        except ImportError:
            pytest.skip("tree-sitter-javascript not available")

    def test_typescript_entity_extraction_integration(self, temp_dir):
        """Test that TypeScript parser integrates with database entity extraction."""
        plugin = LanguagePluginRegistry.get_plugin("typescript")

        try:
            parser = plugin.create_parser()

            # Create TypeScript file with specific patterns
            ts_file = temp_dir / "entities.ts"
            ts_content = """
export interface ProductData {
    id: string;
    name: string;
    price: number;
}

export class ProductService {
    private products: ProductData[] = [];

    constructor() {
        this.products = [];
    }

    addProduct(product: ProductData): void {
        this.products.push(product);
    }

    getProduct(id: string): ProductData | undefined {
        return this.products.find(p => p.id === id);
    }
}

export function createProduct(name: string, price: number): ProductData {
    return {
        id: Math.random().toString(36),
        name,
        price
    };
}
"""
            ts_file.write_text(ts_content)

            # Test entity extraction for database
            entities = parser.extract_entities(ts_file, file_id=1)

            # Verify entity structure
            assert isinstance(entities, dict)
            assert "modules" in entities
            assert "classes" in entities
            assert "functions" in entities
            assert "imports" in entities

            # Verify module entities
            modules = entities["modules"]
            assert len(modules) == 1
            assert modules[0]["name"] == "entities"
            assert modules[0]["file_id"] == 1

            # Verify class entities
            classes = entities["classes"]
            assert len(classes) == 1
            assert classes[0]["name"] == "ProductService"
            assert classes[0]["file_id"] == 1
            assert classes[0]["methods"] >= 2  # constructor, addProduct, getProduct

            # Verify function entities
            functions = entities["functions"]
            # Should include both standalone functions and methods
            function_names = [f["name"] for f in functions]
            assert "createProduct" in function_names
            assert "ProductService.addProduct" in function_names
            assert "ProductService.getProduct" in function_names

        except ImportError:
            pytest.skip("tree-sitter-typescript not available")

    def test_javascript_entity_extraction_integration(self, temp_dir):
        """Test that JavaScript parser integrates with database entity extraction."""
        plugin = LanguagePluginRegistry.get_plugin("javascript")

        try:
            parser = plugin.create_parser()

            # Create JavaScript file with specific patterns
            js_file = temp_dir / "entities.js"
            js_content = """
class ProductService {
    constructor() {
        this.products = [];
    }

    addProduct(product) {
        this.products.push(product);
    }

    getProduct(id) {
        return this.products.find(p => p.id === id);
    }

    static createInstance() {
        return new ProductService();
    }
}

function createProduct(name, price) {
    return {
        id: Math.random().toString(36),
        name,
        price
    };
}

const validateProduct = (product) => {
    return product && product.name && product.price > 0;
};

module.exports = { ProductService, createProduct, validateProduct };
"""
            js_file.write_text(js_content)

            # Test entity extraction for database
            entities = parser.extract_entities(js_file, file_id=2)

            # Verify entity structure
            assert isinstance(entities, dict)
            assert "modules" in entities
            assert "classes" in entities
            assert "functions" in entities
            assert "imports" in entities

            # Verify module entities
            modules = entities["modules"]
            assert len(modules) == 1
            assert modules[0]["name"] == "entities"
            assert modules[0]["file_id"] == 2

            # Verify class entities
            classes = entities["classes"]
            assert len(classes) == 1
            assert classes[0]["name"] == "ProductService"
            assert classes[0]["file_id"] == 2
            assert classes[0]["methods"] >= 3  # constructor, addProduct, getProduct

            # Verify function entities
            functions = entities["functions"]
            function_names = [f["name"] for f in functions]
            assert "createProduct" in function_names
            assert "validateProduct" in function_names
            assert "ProductService.addProduct" in function_names
            assert "ProductService.getProduct" in function_names

        except ImportError:
            pytest.skip("tree-sitter-javascript not available")
