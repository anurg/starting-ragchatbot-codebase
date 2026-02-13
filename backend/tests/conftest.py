import sys
import os
from dataclasses import dataclass
from unittest.mock import MagicMock
from typing import Optional

import pytest

# Add backend directory to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@dataclass
class BuggyConfig:
    """Config that mirrors the current buggy defaults"""
    ANTHROPIC_API_KEY: str = "test-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 0          # Bug 1: should be > 0
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@dataclass
class FixedConfig:
    """Config with bugs corrected"""
    ANTHROPIC_API_KEY: str = "test-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5          # Fixed
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def buggy_config():
    return BuggyConfig()


@pytest.fixture
def fixed_config():
    return FixedConfig()


# ---------------------------------------------------------------------------
# Sample search results
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_search_results():
    """Non-empty search results with two documents"""
    return SearchResults(
        documents=[
            "MCP allows LLMs to interact with external tools.",
            "Agents use planning and tool use to accomplish tasks.",
        ],
        metadata=[
            {"course_title": "MCP Course", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "AI Agents", "lesson_number": 3, "chunk_index": 2},
        ],
        distances=[0.25, 0.42],
    )


@pytest.fixture
def empty_search_results():
    """Empty search results (no error)"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """Search results with an error"""
    return SearchResults.empty("Search error: n_results must be > 0")


# ---------------------------------------------------------------------------
# Mock VectorStore
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store(sample_search_results):
    """A VectorStore mock that returns sample results by default"""
    store = MagicMock()
    store.search.return_value = sample_search_results
    store.get_course_outline.return_value = {
        "title": "MCP Course",
        "course_link": "https://example.com/mcp",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Introduction to MCP"},
            {"lesson_number": 2, "lesson_title": "Building MCP Servers"},
        ],
    }
    return store
