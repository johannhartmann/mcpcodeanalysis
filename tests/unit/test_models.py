"""Tests for database models."""

import contextlib
from collections.abc import Generator
from datetime import UTC, datetime

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import (
    Class,
    CodeEmbedding,
    Commit,
    File,
    Function,
    Import,
    Module,
    Repository,
    SearchHistory,
)


@pytest.fixture
def db_session(sync_engine: Engine) -> Generator[Session, None, None]:
    """Create a database session for testing."""
    connection = sync_engine.connect()
    transaction = connection.begin()

    session_maker = sessionmaker(bind=connection)
    session = session_maker()

    yield session

    session.close()
    with contextlib.suppress(Exception):
        # Transaction might already be rolled back if test failed
        transaction.rollback()
    connection.close()


class TestRepository:
    """Test Repository model."""

    def test_create_repository(self, db_session: Session) -> None:
        """Test creating a repository."""
        repo = Repository(
            github_url="https://github.com/test/repo",
            owner="test",
            name="repo",
            default_branch="main",
        )
        db_session.add(repo)
        db_session.commit()

        assert repo.id is not None
        assert repo.is_active is True
        assert repo.last_synced is not None
        assert repo.created_at is not None

    def test_unique_github_url(self, db_session: Session) -> None:
        """Test unique constraint on github_url."""
        repo1 = Repository(
            github_url="https://github.com/test/repo",
            owner="test",
            name="repo",
        )
        db_session.add(repo1)
        db_session.commit()

        # Try to create duplicate
        repo2 = Repository(
            github_url="https://github.com/test/repo",
            owner="test",
            name="repo",
        )
        db_session.add(repo2)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_repository_relationships(self, db_session: Session) -> None:
        """Test repository relationships."""
        repo = Repository(
            github_url="https://github.com/test/repo",
            owner="test",
            name="repo",
        )
        db_session.add(repo)
        db_session.commit()

        # Add file
        file = File(
            repository_id=repo.id,
            path="test.py",
            language="python",
        )
        db_session.add(file)

        # Add commit
        commit = Commit(
            repository_id=repo.id,
            sha="abc123",
            message="Test commit",
            author="Test Author",
            timestamp=datetime.now(tz=UTC),
        )
        db_session.add(commit)
        db_session.commit()

        # Test relationships
        # Query the relationships directly for SQLite compatibility
        from sqlalchemy import select

        files = (
            db_session.execute(select(File).where(File.repository_id == repo.id))
            .scalars()
            .all()
        )
        commits = (
            db_session.execute(select(Commit).where(Commit.repository_id == repo.id))
            .scalars()
            .all()
        )

        assert len(files) == 1
        assert files[0].path == "test.py"
        assert len(commits) == 1
        assert commits[0].sha == "abc123"


class TestFile:
    """Test File model."""

    def test_create_file(self, db_session: Session) -> None:
        """Test creating a file."""
        repo = Repository(
            github_url="https://github.com/test/repo",
            owner="test",
            name="repo",
        )
        db_session.add(repo)
        db_session.commit()

        file = File(
            repository_id=repo.id,
            path="src/main.py",
            content_hash="hash123",
            git_hash="git456",
            size=1024,
            language="python",
        )
        db_session.add(file)
        db_session.commit()

        assert file.id is not None
        assert file.branch == "main"
        assert file.is_deleted is False

    def test_file_unique_constraint(self, db_session: Session) -> None:
        """Test unique constraint on repository_id, path, branch."""
        repo = Repository(
            github_url="https://github.com/test/repo",
            owner="test",
            name="repo",
        )
        db_session.add(repo)
        db_session.commit()

        file1 = File(
            repository_id=repo.id,
            path="src/main.py",
            branch="main",
        )
        db_session.add(file1)
        db_session.commit()

        # Same file on different branch should work
        file2 = File(
            repository_id=repo.id,
            path="src/main.py",
            branch="develop",
        )
        db_session.add(file2)
        db_session.commit()

        # Same file on same branch should fail
        file3 = File(
            repository_id=repo.id,
            path="src/main.py",
            branch="main",
        )
        db_session.add(file3)

        with pytest.raises(IntegrityError):
            db_session.commit()


class TestModule:
    """Test Module model."""

    def test_create_module(self, db_session: Session) -> None:
        """Test creating a module."""
        repo = Repository(
            github_url="https://github.com/test/repo",
            owner="test",
            name="repo",
        )
        db_session.add(repo)
        db_session.commit()

        file = File(
            repository_id=repo.id,
            path="src/utils.py",
            language="python",
        )
        db_session.add(file)
        db_session.commit()

        module = Module(
            file_id=file.id,
            name="utils",
            docstring="Utility functions",
            start_line=1,
            end_line=100,
        )
        db_session.add(module)
        db_session.commit()

        assert module.id is not None
        assert module.file.path == "src/utils.py"


class TestClass:
    """Test Class model."""

    def test_create_class(self, db_session: Session) -> None:
        """Test creating a class."""
        repo = Repository(
            github_url="https://github.com/test/repo",
            owner="test",
            name="repo",
        )
        db_session.add(repo)
        db_session.commit()

        file = File(repository_id=repo.id, path="models.py")
        db_session.add(file)
        db_session.commit()

        module = Module(file_id=file.id, name="models")
        db_session.add(module)
        db_session.commit()

        cls = Class(
            module_id=module.id,
            name="UserModel",
            docstring="User data model",
            base_classes=["BaseModel", "ABC"],
            decorators=["dataclass"],
            start_line=10,
            end_line=50,
            is_abstract=True,
        )
        db_session.add(cls)
        db_session.commit()

        assert cls.id is not None
        assert cls.base_classes == ["BaseModel", "ABC"]
        assert cls.is_abstract is True


class TestFunction:
    """Test Function model."""

    def test_create_function(self, db_session: Session) -> None:
        """Test creating a module-level function."""
        repo = Repository(
            github_url="https://github.com/test/repo",
            owner="test",
            name="repo",
        )
        db_session.add(repo)
        db_session.commit()

        file = File(repository_id=repo.id, path="utils.py")
        db_session.add(file)
        db_session.commit()

        module = Module(file_id=file.id, name="utils")
        db_session.add(module)
        db_session.commit()

        func = Function(
            module_id=module.id,
            name="process_data",
            docstring="Process input data",
            parameters=[
                {"name": "data", "type": "List[str]"},
                {"name": "validate", "type": "bool", "default": "True"},
            ],
            return_type="Dict[str, Any]",
            decorators=["lru_cache"],
            start_line=20,
            end_line=35,
            is_async=True,
            complexity=5,
        )
        db_session.add(func)
        db_session.commit()

        assert func.id is not None
        assert func.class_id is None
        assert func.is_async is True
        assert len(func.parameters) == 2

    def test_create_method(self, db_session: Session) -> None:
        """Test creating a class method."""
        repo = Repository(
            github_url="https://github.com/test/repo",
            owner="test",
            name="repo",
        )
        db_session.add(repo)
        db_session.commit()

        file = File(repository_id=repo.id, path="models.py")
        db_session.add(file)
        db_session.commit()

        module = Module(file_id=file.id, name="models")
        db_session.add(module)
        db_session.commit()

        cls = Class(module_id=module.id, name="DataProcessor")
        db_session.add(cls)
        db_session.commit()

        method = Function(
            module_id=module.id,
            class_id=cls.id,
            name="__init__",
            parameters=[
                {"name": "self"},
                {"name": "config", "type": "Config"},
            ],
            start_line=25,
            end_line=30,
        )
        db_session.add(method)
        db_session.commit()

        assert method.parent_class.name == "DataProcessor"
        # Query the relationship directly for SQLite compatibility
        from sqlalchemy import select

        methods = (
            db_session.execute(select(Function).where(Function.class_id == cls.id))
            .scalars()
            .all()
        )
        assert len(methods) == 1


class TestImport:
    """Test Import model."""

    def test_create_import(self, db_session: Session) -> None:
        """Test creating an import."""
        repo = Repository(
            github_url="https://github.com/test/repo",
            owner="test",
            name="repo",
        )
        db_session.add(repo)
        db_session.commit()

        file = File(repository_id=repo.id, path="main.py")
        db_session.add(file)
        db_session.commit()

        import_stmt = Import(
            file_id=file.id,
            import_statement="from typing import List, Dict",
            module_name="typing",
            imported_names=["List", "Dict"],
            is_relative=False,
            line_number=2,
        )
        db_session.add(import_stmt)
        db_session.commit()

        assert import_stmt.id is not None
        assert import_stmt.imported_names == ["List", "Dict"]


class TestCommit:
    """Test Commit model."""

    def test_create_commit(self, db_session: Session) -> None:
        """Test creating a commit."""
        repo = Repository(
            github_url="https://github.com/test/repo",
            owner="test",
            name="repo",
        )
        db_session.add(repo)
        db_session.commit()

        commit = Commit(
            repository_id=repo.id,
            sha="abc123def456",
            message="Add new feature",
            author="John Doe",
            author_email="john@example.com",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            files_changed=["src/main.py", "tests/test_main.py"],
            additions=50,
            deletions=10,
        )
        db_session.add(commit)
        db_session.commit()

        assert commit.id is not None
        assert commit.processed is False
        assert len(commit.files_changed) == 2


class TestCodeEmbedding:
    """Test CodeEmbedding model."""

    def test_create_embedding(self, db_session: Session) -> None:
        """Test creating a code embedding."""
        repo = Repository(
            github_url="https://github.com/test/repo",
            owner="test",
            name="repo",
        )
        db_session.add(repo)
        db_session.commit()

        file = File(repository_id=repo.id, path="main.py")
        db_session.add(file)
        db_session.commit()

        module = Module(file_id=file.id, name="main")
        db_session.add(module)
        db_session.commit()

        func = Function(module_id=module.id, name="main")
        db_session.add(func)
        db_session.commit()

        # Create embedding with mock vector (normally from OpenAI)
        embedding = CodeEmbedding(
            entity_type="function",
            entity_id=func.id,
            file_id=file.id,
            embedding_type="raw",
            embedding=[0.1] * 1536,  # Mock 1536-dimensional vector
            content="def main():\n    pass",
            tokens=10,
            metadata={"language": "python"},
        )
        db_session.add(embedding)
        db_session.commit()

        assert embedding.id is not None
        assert len(embedding.embedding) == 1536


class TestSearchHistory:
    """Test SearchHistory model."""

    def test_create_search_history(self, db_session: Session) -> None:
        """Test creating a search history entry."""
        search = SearchHistory(
            query="find all authentication functions",
            query_type="search_code",
            results_count=5,
            response_time_ms=125.5,
            user_id="user123",
            session_id="session456",
        )
        db_session.add(search)
        db_session.commit()

        assert search.id is not None
        assert search.response_time_ms == 125.5
