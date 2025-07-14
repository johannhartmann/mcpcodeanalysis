-- Migration to add missing columns to migration_patterns table
-- These columns are referenced in the code but missing from the schema

ALTER TABLE migration_patterns
ADD COLUMN IF NOT EXISTS implementation_steps JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS best_practices JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS anti_patterns JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS risks JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS example_code TEXT;

-- Update existing rows with default values if needed
UPDATE migration_patterns
SET
    implementation_steps = COALESCE(implementation_steps, '[]'::jsonb),
    best_practices = COALESCE(best_practices, '[]'::jsonb),
    anti_patterns = COALESCE(anti_patterns, '[]'::jsonb),
    risks = COALESCE(risks, '[]'::jsonb)
WHERE
    implementation_steps IS NULL
    OR best_practices IS NULL
    OR anti_patterns IS NULL
    OR risks IS NULL;
