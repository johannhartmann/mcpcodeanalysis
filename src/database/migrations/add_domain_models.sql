-- Migration: Add domain-driven design models for semantic analysis
-- This migration adds tables for storing LLM-extracted domain entities,
-- relationships, bounded contexts, and hierarchical summaries.

-- Create enum types
CREATE TYPE domain_entity_type AS ENUM (
    'aggregate_root',
    'entity',
    'value_object',
    'domain_service',
    'domain_event',
    'command',
    'query',
    'policy',
    'factory',
    'repository_interface'
);

CREATE TYPE domain_relationship_type AS ENUM (
    'uses',
    'creates',
    'modifies',
    'deletes',
    'queries',
    'validates',
    'orchestrates',
    'implements',
    'extends',
    'aggregates',
    'references',
    'publishes',
    'subscribes_to',
    'depends_on',
    'composed_of'
);

CREATE TYPE bounded_context_type AS ENUM (
    'core',
    'supporting',
    'generic',
    'external'
);

CREATE TYPE context_relationship_type AS ENUM (
    'shared_kernel',
    'customer_supplier',
    'conformist',
    'anti_corruption_layer',
    'open_host_service',
    'published_language',
    'partnership',
    'big_ball_of_mud'
);

CREATE TYPE summary_level AS ENUM (
    'function',
    'class',
    'module',
    'package',
    'context'
);

-- Domain entities table
CREATE TABLE domain_entities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    entity_type domain_entity_type NOT NULL,
    description TEXT,
    business_rules JSONB DEFAULT '[]'::jsonb,
    invariants JSONB DEFAULT '[]'::jsonb,
    responsibilities JSONB DEFAULT '[]'::jsonb,
    ubiquitous_language JSONB DEFAULT '{}'::jsonb,
    source_entities INTEGER[] DEFAULT '{}',
    confidence_score FLOAT DEFAULT 1.0,
    concept_embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    extraction_metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for domain_entities
CREATE INDEX idx_domain_entity_name ON domain_entities(name);
CREATE INDEX idx_domain_entity_type ON domain_entities(entity_type);
CREATE INDEX idx_domain_entity_source ON domain_entities USING gin(source_entities);
CREATE INDEX idx_domain_entity_embedding ON domain_entities USING ivfflat (concept_embedding vector_cosine_ops);

-- Domain relationships table
CREATE TABLE domain_relationships (
    id SERIAL PRIMARY KEY,
    source_entity_id INTEGER NOT NULL REFERENCES domain_entities(id) ON DELETE CASCADE,
    target_entity_id INTEGER NOT NULL REFERENCES domain_entities(id) ON DELETE CASCADE,
    relationship_type domain_relationship_type NOT NULL,
    description TEXT,
    strength FLOAT DEFAULT 1.0,
    confidence_score FLOAT DEFAULT 1.0,
    evidence JSONB DEFAULT '[]'::jsonb,
    interaction_patterns JSONB DEFAULT '[]'::jsonb,
    data_flow JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uq_domain_relationship UNIQUE(source_entity_id, target_entity_id, relationship_type)
);

-- Indexes for domain_relationships
CREATE INDEX idx_domain_rel_source ON domain_relationships(source_entity_id);
CREATE INDEX idx_domain_rel_target ON domain_relationships(target_entity_id);
CREATE INDEX idx_domain_rel_type ON domain_relationships(relationship_type);

-- Bounded contexts table
CREATE TABLE bounded_contexts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    ubiquitous_language JSONB DEFAULT '{}'::jsonb,
    core_concepts JSONB DEFAULT '[]'::jsonb,
    published_language JSONB DEFAULT '{}'::jsonb,
    anti_corruption_layer JSONB DEFAULT '{}'::jsonb,
    context_type bounded_context_type DEFAULT 'supporting',
    summary TEXT,
    responsibilities JSONB DEFAULT '[]'::jsonb,
    cohesion_score FLOAT,
    coupling_score FLOAT,
    modularity_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for bounded_contexts
CREATE INDEX idx_bounded_context_name ON bounded_contexts(name);
CREATE INDEX idx_bounded_context_type ON bounded_contexts(context_type);

-- Bounded context memberships (many-to-many)
CREATE TABLE bounded_context_memberships (
    id SERIAL PRIMARY KEY,
    domain_entity_id INTEGER NOT NULL REFERENCES domain_entities(id) ON DELETE CASCADE,
    bounded_context_id INTEGER NOT NULL REFERENCES bounded_contexts(id) ON DELETE CASCADE,
    role VARCHAR(100),
    importance_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uq_context_membership UNIQUE(domain_entity_id, bounded_context_id)
);

-- Indexes for bounded_context_memberships
CREATE INDEX idx_membership_entity ON bounded_context_memberships(domain_entity_id);
CREATE INDEX idx_membership_context ON bounded_context_memberships(bounded_context_id);

-- Context relationships table
CREATE TABLE context_relationships (
    id SERIAL PRIMARY KEY,
    source_context_id INTEGER NOT NULL REFERENCES bounded_contexts(id) ON DELETE CASCADE,
    target_context_id INTEGER NOT NULL REFERENCES bounded_contexts(id) ON DELETE CASCADE,
    relationship_type context_relationship_type NOT NULL,
    description TEXT,
    interface_description JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uq_context_relationship UNIQUE(source_context_id, target_context_id, relationship_type)
);

-- Indexes for context_relationships
CREATE INDEX idx_context_rel_source ON context_relationships(source_context_id);
CREATE INDEX idx_context_rel_target ON context_relationships(target_context_id);

-- Domain summaries table
CREATE TABLE domain_summaries (
    id SERIAL PRIMARY KEY,
    level summary_level NOT NULL,
    entity_type VARCHAR(50),
    entity_id INTEGER,
    business_summary TEXT,
    technical_summary TEXT,
    domain_concepts JSONB DEFAULT '[]'::jsonb,
    parent_summary_id INTEGER REFERENCES domain_summaries(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for domain_summaries
CREATE INDEX idx_summary_level ON domain_summaries(level);
CREATE INDEX idx_summary_entity ON domain_summaries(entity_type, entity_id);
CREATE INDEX idx_summary_parent ON domain_summaries(parent_summary_id);

-- Add update timestamp triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_domain_entities_updated_at BEFORE UPDATE ON domain_entities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_domain_relationships_updated_at BEFORE UPDATE ON domain_relationships
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_bounded_contexts_updated_at BEFORE UPDATE ON bounded_contexts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_context_relationships_updated_at BEFORE UPDATE ON context_relationships
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_domain_summaries_updated_at BEFORE UPDATE ON domain_summaries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
