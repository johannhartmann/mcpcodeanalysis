# Vulture whitelist - these are false positives
# This file is used to tell vulture about code that appears unused but is actually used

# SQLAlchemy models need these imports to register with metadata
import src.database.package_models  # noqa: F401

# These are used by SQLAlchemy as column types
dim = None  # Used in pgvector columns
item_type = None  # Used in ENUM columns

# Used in argparse/click
reload = None  # CLI argument

# Used by context managers
exc_type = None
exc_val = None
exc_tb = None
frame = None

# Used by structlog processors
Processor = None
