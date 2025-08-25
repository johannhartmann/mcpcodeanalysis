# Plugin Initialization Strategy

This document outlines strategies for handling plugin initialization and ParserFactory interaction. Two options are considered: a legacy-only deterministic approach (Option 1) and a hybrid deterministic initialization approach (Option 2).

## Background

Current test suite exposes order-dependent behavior between ParserFactory and LanguagePluginRegistry. Some unit tests expect ParserFactory to behave using legacy-registered parsers only (no plugin registry initialization). Integration tests expect ParserFactory to reflect plugin-registered languages (i.e., plugin registry initialized and active).

We need a clear strategy to make behavior deterministic and reduce test flakiness while preserving integration semantics.

---

## Option 1 — Legacy-only ParserFactory (Recommended for strict unit stability)

Goal

- Make ParserFactory deterministic and independent of the global plugin registry.

Approach

- ParserFactory should exclusively use legacy registration mappings: _parsers and _language_parsers.
- It must not consult or trigger LanguagePluginRegistry initialization in any of its APIs.
- Integration code that needs plugin-backed behaviour should use LanguagePluginRegistry directly (or a new adapter) rather than relying on ParserFactory.

Implementation steps

- Remove any calls to LanguagePluginRegistry from ParserFactory.
- Populate legacy mappings at module import for supported languages (Python by default). Keep imports guarded with ImportError.
- Update docs to instruct integration tests and runtime code to use LanguagePluginRegistry for plugin-based discovery.
- Update tests that currently rely on ParserFactory to detect plugin-registered languages to use LanguagePluginRegistry instead.

Pros

- Deterministic behavior; unit tests become order-independent and stable.
- Clear separation of concerns: ParserFactory for legacy mapping, LanguagePluginRegistry for plugin discovery.
- Lower risk of global side-effects during imports.

Cons

- Requires updating integration tests (small scope) that currently call ParserFactory for plugin-backed checks.
- Slight API duplication exists if callers expect a single discovery mechanism.

When to choose

- When strict test stability is required and we prefer the least surprising behavior in unit tests.

---

## Option 2 — Hybrid with Deterministic Plugin Initialization (Recommended for preserving integration semantics)

Goal

- Keep ParserFactory able to reflect plugin-registered languages while ensuring deterministic, predictable initialization of LanguagePluginRegistry.

Approach

- Provide a deterministic initialization policy for LanguagePluginRegistry so test order doesn't matter:
  - The registry must not auto-initialize on import.
  - It should expose an explicit initialization API and a configuration hook to control which default plugins to register (e.g., minimal, full, or custom).
  - ParserFactory may consult the registry only if it has been explicitly initialized (i.e., no implicit initialization triggered by ParserFactory).

Implementation steps

1. Modify LanguagePluginRegistry:
   - Keep _initialized = False by default.
   - Remove implicit initialization from APIs like is_supported or get_plugin_by_extension. Instead, APIs must either:
     - Raise a clear exception if used before explicit initialization, or
     - Accept a parameter `auto_init: bool = True` to control initialization behavior.
   - Add an explicit `initialize(default_plugins: list[str] | None)` method which registers a known stable default set (e.g., ["python"] or a flag for full auto-discovery).
   - Provide `clear_plugins()` for tests to reset state.

2. Modify ParserFactory:
   - Prefer legacy mappings first.
   - Consult LanguagePluginRegistry only if `LanguagePluginRegistry._initialized` is True (i.e., registry was explicitly initialized by the test or runtime).
   - Do not call any plugin registry API that would implicitly initialize the registry.

3. Adjust tests:
   - Unit tests that expect legacy-only behavior do nothing (registry stays uninitialized).
   - Integration tests explicitly call `LanguagePluginRegistry.initialize()` in their setup to enable plugin-backed behavior.

Pros

- Preserves integration semantics (ParserFactory can reflect plugin-based languages when desired).
- Keeps backward compatibility while making initialization explicit and deterministic.
- Tests that require plugins will explicitly request plugin initialization, making intent explicit.

Cons

- More code changes and a small behavioral migration for callers who implicitly relied on auto-init APIs.
- Requires updating tests and possibly some runtime initialization code.

When to choose

- When both unit test stability and integration semantics are important and you prefer explicit control over plugin initialization.

---

## Recommendation

Option 2 is preferred for long-term maintainability: it retains the plugin system's power while making initialization explicit and predictable. Option 1 is simpler and provides the fastest path to deterministic unit tests, but requires updating integration tests to use the LanguagePluginRegistry directly.

If you prefer Option 2, the next steps are:

- Implement `LanguagePluginRegistry.initialize(default_plugins: list[str] | None)` and make auto-init opt-in only (or explicit).
- Update ParserFactory to consult the registry only when initialized.
- Update integration tests to call initialize() in their setup.

If you prefer Option 1, the next steps are:

- Remove plugin registry consultation completely from ParserFactory.
- Update integration tests that expect plugin-based detection via ParserFactory to instead call LanguagePluginRegistry explicitly.

---

## Notes

- All changes must be accompanied by focused unit tests and small commits with clear messages.
- Keep exception handling narrow (ImportError), prefer explicit logging for missing plugins, and avoid global side-effects in module imports.
