-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable additional useful extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search
CREATE EXTENSION IF NOT EXISTS btree_gin;  -- For composite indexes
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;  -- For query performance monitoring

-- Create custom types
CREATE TYPE code_entity_type AS ENUM ('module', 'class', 'function', 'method', 'variable');
CREATE TYPE embedding_type AS ENUM ('raw', 'interpreted');

-- Note: shared_buffers must be set in postgresql.conf, not via ALTER DATABASE
-- These parameters can be set at the database level:
ALTER DATABASE code_analysis SET effective_cache_size = '1GB';
ALTER DATABASE code_analysis SET maintenance_work_mem = '64MB';
ALTER DATABASE code_analysis SET work_mem = '16MB';

-- Create indexes for better performance (will be created after tables are created by Alembic)
-- This file ensures the database is properly initialized with required extensions
