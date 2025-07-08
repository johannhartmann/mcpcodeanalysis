"""Tests for logging utilities."""

import json
import logging
import tempfile
from pathlib import Path

import pytest

from src.mcp_server.config import LoggingConfig
from src.utils.logger import JSONFormatter, StructuredLogger, get_logger, setup_logging


class TestJSONFormatter:
    """Test JSON log formatter."""
    
    def test_format_basic_record(self):
        """Test formatting a basic log record."""
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data["level"] == "INFO"
        assert data["logger"] == "test.module"
        assert data["message"] == "Test message"
        assert data["line"] == 42
        assert "timestamp" in data
    
    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test.module",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data["level"] == "ERROR"
        assert "exception" in data
        assert "ValueError: Test error" in data["exception"]
    
    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        # Add extra fields
        record.user_id = "user123"
        record.request_id = "req456"
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data["user_id"] == "user123"
        assert data["request_id"] == "req456"


class TestStructuredLogger:
    """Test structured logger."""
    
    def test_basic_logging(self, caplog):
        """Test basic logging methods."""
        base_logger = logging.getLogger("test")
        logger = StructuredLogger(base_logger)
        
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
        
        assert len(caplog.records) == 5
        assert caplog.records[0].levelname == "DEBUG"
        assert caplog.records[1].levelname == "INFO"
        assert caplog.records[2].levelname == "WARNING"
        assert caplog.records[3].levelname == "ERROR"
        assert caplog.records[4].levelname == "CRITICAL"
    
    def test_with_context(self, caplog):
        """Test logging with context."""
        base_logger = logging.getLogger("test")
        logger = StructuredLogger(base_logger)
        
        # Create logger with context
        context_logger = logger.with_context(
            user_id="user123",
            request_id="req456",
        )
        
        with caplog.at_level(logging.INFO):
            context_logger.info("Test message", action="test_action")
        
        record = caplog.records[0]
        assert record.user_id == "user123"
        assert record.request_id == "req456"
        assert record.action == "test_action"
    
    def test_nested_context(self, caplog):
        """Test nested context."""
        base_logger = logging.getLogger("test")
        logger = StructuredLogger(base_logger)
        
        # Create nested contexts
        logger1 = logger.with_context(level1="value1")
        logger2 = logger1.with_context(level2="value2")
        
        with caplog.at_level(logging.INFO):
            logger2.info("Test message")
        
        record = caplog.records[0]
        assert record.level1 == "value1"
        assert record.level2 == "value2"
    
    def test_exception_logging(self, caplog):
        """Test exception logging."""
        base_logger = logging.getLogger("test")
        logger = StructuredLogger(base_logger)
        
        try:
            raise ValueError("Test error")
        except ValueError:
            with caplog.at_level(logging.ERROR):
                logger.exception("Error occurred", error_code="E001")
        
        record = caplog.records[0]
        assert record.levelname == "ERROR"
        assert record.error_code == "E001"
        assert record.exc_info is not None


class TestLoggingSetup:
    """Test logging setup functionality."""
    
    def test_setup_console_logging(self):
        """Test console logging setup."""
        config = LoggingConfig(
            level="DEBUG",
            format="text",
            console_enabled=True,
            console_colorized=False,
            file_enabled=False,
        )
        
        setup_logging(config)
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        
        # Should have console handler
        handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(handlers) >= 1
    
    def test_setup_file_logging(self, tmp_path):
        """Test file logging setup."""
        log_file = tmp_path / "test.log"
        
        config = LoggingConfig(
            level="INFO",
            format="json",
            console_enabled=False,
            file_enabled=True,
            file_path=log_file,
            file_rotation="daily",
            file_retention_days=7,
        )
        
        setup_logging(config)
        
        # Log a message
        logger = logging.getLogger("test")
        logger.info("Test message")
        
        # Check log file exists and contains JSON
        assert log_file.exists()
        content = log_file.read_text()
        data = json.loads(content.strip())
        assert data["message"] == "Test message"
    
    def test_setup_json_formatter(self):
        """Test JSON formatter setup."""
        config = LoggingConfig(
            level="INFO",
            format="json",
            console_enabled=True,
            console_colorized=False,
            file_enabled=False,
        )
        
        setup_logging(config)
        
        root_logger = logging.getLogger()
        console_handlers = [
            h for h in root_logger.handlers 
            if isinstance(h, logging.StreamHandler)
        ]
        
        assert len(console_handlers) > 0
        handler = console_handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)
    
    def test_specific_logger_levels(self):
        """Test setting specific logger levels."""
        config = LoggingConfig(level="DEBUG")
        setup_logging(config)
        
        # These loggers should have WARNING level
        assert logging.getLogger("sqlalchemy.engine").level == logging.WARNING
        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.WARNING


class TestGetLogger:
    """Test get_logger function."""
    
    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test.module")
        
        assert isinstance(logger, StructuredLogger)
        assert logger.logger.name == "test.module"
    
    def test_get_logger_different_names(self):
        """Test getting loggers with different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        
        assert logger1.logger.name == "module1"
        assert logger2.logger.name == "module2"
        assert logger1.logger is not logger2.logger