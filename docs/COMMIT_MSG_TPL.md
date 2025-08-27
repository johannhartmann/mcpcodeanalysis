Commit message template for LM-assisted drafting

Use this file as the source of truth when instructing language models to draft or reword commit messages for this repository.

Template
--------

<type>(<scope>): <short, imperative subject (<=60 chars)>

- <One-line summary or top-level change>
- <Concrete list of changes, one per bullet>
- <Notes about breaking changes, compatibility, or migration steps (if any)>
- Testing: <what was tested / tests added / how to run them>
- Reviewer: <focus points / what to review>

Optional sections (only when relevant)
- Benefits: <why this change helps>
- Risk/Drawbacks: <known risks>
- References: <issue/PR numbers or links>

Preferred types
- feat: new feature
- fix: bug fix
- refactor: code restructuring without behavior change
- perf: performance improvement
- style: formatting / import ordering / lint
- chore: repo maintenance / tooling
- test: tests added/changed
- docs: documentation

Guidelines for LLM prompts
- Always start the commit subject with one of the preferred types and include a scope in parentheses.
- Keep the subject imperative, present tense, and <=60 characters.
- Provide a concise bulleted body: each bullet should be a single change or note.
- Include explicit "Testing" and "Reviewer" lines when applicable.
- Only include optional sections when they convey important, non-obvious information.
- If a change is part of a larger ticket, include a References line with the issue/PR number.

Example
-------
fix(indexer): handle numeric-suffixed entity names

- Group raw entities by type using raw_by_type and normalize display names
- Collapse numeric-suffixed variants (Order0, Order1 -> Order)
- Add DomainIndexer import and remove duplicate import
- Small readability and formatting improvements

Testing: unit tests for indexer adjusted; ran full test suite locally
Reviewer: focus on raw_by_type grouping and import cleanup

Why this file
---------------
LLMs and human contributors should use this template to produce consistent, reviewer-friendly commit messages. Keeping a stable, repo-local template helps automation and review processes.
