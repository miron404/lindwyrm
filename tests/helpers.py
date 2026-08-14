"""Shared test fixtures.

Tests build a real Config rather than a stand-in. Four separate fake config
dataclasses used to drift from the real one every time a field was added,
and the production code had grown defensive getattr() calls to tolerate
them -- which also swallowed genuine AttributeErrors.
"""

from pathlib import Path

from lindwyrm.config import Config, Policy

# Paths that certainly don't exist, so a test never picks up the AGENTS.md or
# project files of whoever is running it.
NOWHERE = Path("/nonexistent-lindwyrm-project")
NO_USER_DIR = Path("/nonexistent-lindwyrm-userdir")


def make_config(**overrides) -> Config:
    """A Config with harmless defaults, overridable per test."""
    settings = {
        "api_key": "test-key",
        "project_root": NOWHERE,
        "user_context_dir": NO_USER_DIR,
        "policy": Policy(),
        # Small enough that a handful of short fake messages can cross it.
        "context_limit": 100_000,
        "compact_keep_last": 2,
        "compact_keep_tokens": 1,
        "offload": False,
        "save_sessions": False,
    }
    settings.update(overrides)
    return Config(**settings)
