from pathlib import Path

import pytest
from _pytest.nodes import Item


def pytest_collection_modifyitems(
    session: object, config: object, items: list[Item]
) -> None:
    # use session/config to satisfy linters and pluggy hook signature
    _ = (session, config)

    """Mark all collected tests in this directory as 'unit'.

    This conftest lives in tests/unit/ so it will only be loaded for tests
    collected under that directory. We add a 'unit' marker to every collected
    test item here so tests can be selected with -m unit without editing
    each test file.
    """
    root = Path(__file__).resolve().parent
    for item in items:
        try:
            item_path = Path(str(item.fspath)).resolve()
        except (TypeError, ValueError, AttributeError) as e:
            # If we cannot resolve path, skip marking this item
            # Log at debug level so we can diagnose path resolution issues if needed
            import logging

            logging.getLogger(__name__).debug(
                "Failed to resolve path for %s: %s", item, e
            )
            continue

        # Only mark items that live under this conftest's directory
        try:
            item_path.relative_to(root)
            item.add_marker(pytest.mark.unit)
        except ValueError:
            # item not under this directory
            continue
