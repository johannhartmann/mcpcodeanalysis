"""File watcher for monitoring code changes."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.mcp_server.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CodeFileHandler(FileSystemEventHandler):
    """Handle file system events for code files."""
    
    def __init__(
        self,
        callback: Callable[[str, str], None],
        extensions: Set[str] = None,
        exclude_patterns: Set[str] = None
    ) -> None:
        self.callback = callback
        self.extensions = extensions or {".py"}
        self.exclude_patterns = exclude_patterns or set(config.scanner.exclude_patterns)
        self.pending_changes: Dict[str, datetime] = {}
        self.debounce_seconds = 1.0
    
    def should_process(self, path: str) -> bool:
        """Check if a file should be processed."""
        path_obj = Path(path)
        
        # Check extension
        if path_obj.suffix not in self.extensions:
            return False
        
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if self._match_pattern(path_obj.name, pattern):
                return False
            
            # Check parent directories
            for parent in path_obj.parents:
                if self._match_pattern(parent.name, pattern):
                    return False
        
        return True
    
    def _match_pattern(self, name: str, pattern: str) -> bool:
        """Check if a name matches an exclusion pattern."""
        if pattern.startswith("*"):
            return name.endswith(pattern[1:])
        elif pattern.endswith("*"):
            return name.startswith(pattern[:-1])
        else:
            return name == pattern
    
    def on_modified(self, event) -> None:
        """Handle file modification."""
        if not event.is_directory and self.should_process(event.src_path):
            self.pending_changes[event.src_path] = datetime.now()
            self._schedule_callback(event.src_path, "modified")
    
    def on_created(self, event) -> None:
        """Handle file creation."""
        if not event.is_directory and self.should_process(event.src_path):
            self.pending_changes[event.src_path] = datetime.now()
            self._schedule_callback(event.src_path, "created")
    
    def on_deleted(self, event) -> None:
        """Handle file deletion."""
        if not event.is_directory and self.should_process(event.src_path):
            self.callback(event.src_path, "deleted")
            self.pending_changes.pop(event.src_path, None)
    
    def on_moved(self, event) -> None:
        """Handle file move/rename."""
        if not event.is_directory:
            if self.should_process(event.src_path):
                self.callback(event.src_path, "deleted")
                self.pending_changes.pop(event.src_path, None)
            
            if self.should_process(event.dest_path):
                self.pending_changes[event.dest_path] = datetime.now()
                self._schedule_callback(event.dest_path, "created")
    
    def _schedule_callback(self, path: str, event_type: str) -> None:
        """Schedule callback with debouncing."""
        async def delayed_callback():
            await asyncio.sleep(self.debounce_seconds)
            
            # Check if this is still the latest change
            if path in self.pending_changes:
                change_time = self.pending_changes[path]
                if (datetime.now() - change_time).total_seconds() >= self.debounce_seconds:
                    self.callback(path, event_type)
                    self.pending_changes.pop(path, None)
        
        asyncio.create_task(delayed_callback())


class FileWatcher:
    """Watch directories for code file changes."""
    
    def __init__(self) -> None:
        self.observer = Observer()
        self.watches: Dict[str, Any] = {}
    
    def add_watch(
        self,
        path: Path,
        callback: Callable[[str, str], None],
        recursive: bool = True,
        extensions: Optional[Set[str]] = None
    ) -> str:
        """Add a directory to watch."""
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        handler = CodeFileHandler(callback, extensions)
        watch = self.observer.schedule(handler, str(path), recursive=recursive)
        
        watch_id = str(path)
        self.watches[watch_id] = {
            "watch": watch,
            "handler": handler,
            "path": path,
        }
        
        logger.info(f"Added watch for: {path}")
        return watch_id
    
    def remove_watch(self, watch_id: str) -> None:
        """Remove a directory watch."""
        if watch_id in self.watches:
            watch_info = self.watches[watch_id]
            self.observer.unschedule(watch_info["watch"])
            del self.watches[watch_id]
            logger.info(f"Removed watch for: {watch_id}")
    
    def start(self) -> None:
        """Start the file watcher."""
        self.observer.start()
        logger.info("File watcher started")
    
    def stop(self) -> None:
        """Stop the file watcher."""
        self.observer.stop()
        self.observer.join()
        logger.info("File watcher stopped")
    
    def get_active_watches(self) -> List[Dict[str, Any]]:
        """Get list of active watches."""
        return [
            {
                "path": str(info["path"]),
                "recursive": True,  # Always true for now
            }
            for info in self.watches.values()
        ]