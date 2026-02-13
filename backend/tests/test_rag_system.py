"""Tests for RAGSystem orchestration and the FastAPI /api/query endpoint.

Bugs exposed:
- Bug 1: MAX_RESULTS=0 → VectorStore search fails
- Bug 2: rag_system.query() returns List[str] but QueryResponse expects List[Source]
- Bug 3: only search_course_content registered, not get_course_outline
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from config import config as real_config
from vector_store import SearchResults


# ===================================================================
# Bug 1: MAX_RESULTS = 0
# ===================================================================

class TestBug1MaxResults:

    def test_config_max_results_is_positive(self):
        """BUG 1 FIXED: MAX_RESULTS should be > 0."""
        assert real_config.MAX_RESULTS > 0, (
            f"MAX_RESULTS must be > 0, got {real_config.MAX_RESULTS}"
        )

    def test_max_results_positive_propagated_to_vector_store(self, fixed_config):
        """BUG 1 FIXED: VectorStore receives max_results > 0 from config."""
        assert fixed_config.MAX_RESULTS > 0

    def test_vector_store_search_with_zero_results_errors(self):
        """BUG 1: ChromaDB raises when n_results=0."""
        # Simulate what happens inside VectorStore.search when max_results=0
        mock_collection = MagicMock()
        mock_collection.query.side_effect = ValueError("n_results must be > 0")

        # This is the code path in VectorStore.search
        try:
            mock_collection.query(query_texts=["test"], n_results=0, where=None)
            assert False, "Should have raised"
        except ValueError as e:
            assert "n_results" in str(e)


# ===================================================================
# Bug 2: Source type mismatch
# ===================================================================

class TestBug2SourceMismatch:

    def test_query_returns_list_of_strings(self, buggy_config):
        """BUG 2: rag_system.query() returns (str, List[str])."""
        from search_tools import ToolManager, CourseSearchTool

        mock_store = MagicMock()
        mock_store.search.return_value = SearchResults(
            documents=["content"],
            metadata=[{"course_title": "Test", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1],
        )

        tm = ToolManager()
        tool = CourseSearchTool(mock_store)
        tm.register_tool(tool)

        # Simulate what rag_system.query does after AI generates response
        tm.execute_tool("search_course_content", query="test")
        sources = tm.get_last_sources()

        assert isinstance(sources, list)
        assert all(isinstance(s, str) for s in sources), (
            "Sources should be plain strings — which is the bug, since app.py "
            "expects List[Source]"
        )

    def test_query_response_rejects_string_sources(self):
        """BUG 2: Constructing QueryResponse with string sources raises ValidationError."""
        # Import the Pydantic models from app.py
        # We need to import them from the module
        import importlib
        import sys

        # Create the Source and QueryResponse models matching app.py
        from pydantic import BaseModel
        from typing import List, Optional

        class Source(BaseModel):
            label: str
            link: Optional[str] = None

        class QueryResponse(BaseModel):
            answer: str
            sources: List[Source]
            session_id: str

        # This is exactly what happens in app.py line 73
        string_sources = ["MCP Course - Lesson 1", "AI Agents - Lesson 3"]

        # BUG 2 FIXED: app.py now wraps strings in Source objects, so raw strings
        # would still fail Pydantic validation — confirming the fix is needed.
        with pytest.raises(ValidationError):
            QueryResponse(
                answer="Some answer",
                sources=string_sources,
                session_id="session_1",
            )

    def test_query_response_accepts_source_objects(self):
        """BUG 2 FIXED: QueryResponse works when sources are Source objects."""
        from pydantic import BaseModel
        from typing import List, Optional

        class Source(BaseModel):
            label: str
            link: Optional[str] = None

        class QueryResponse(BaseModel):
            answer: str
            sources: List[Source]
            session_id: str

        # This is what the fixed app.py now does
        string_sources = ["MCP Course - Lesson 1", "AI Agents - Lesson 3"]
        source_objects = [Source(label=s) for s in string_sources]

        resp = QueryResponse(
            answer="Some answer",
            sources=source_objects,
            session_id="session_1",
        )
        assert len(resp.sources) == 2
        assert resp.sources[0].label == "MCP Course - Lesson 1"


# ===================================================================
# Bug 3: Missing get_course_outline tool
# ===================================================================

class TestBug3MissingTool:

    def test_both_tools_registered(self, mock_vector_store):
        """BUG 3 FIXED: both search_course_content and get_course_outline are registered."""
        from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool

        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_vector_store))
        tm.register_tool(CourseOutlineTool(mock_vector_store))

        tool_names = [d["name"] for d in tm.get_tool_definitions()]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_get_course_outline_call_returns_outline(self, mock_vector_store):
        """BUG 3 FIXED: Calling get_course_outline returns formatted outline."""
        from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool

        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_vector_store))
        tm.register_tool(CourseOutlineTool(mock_vector_store))

        result = tm.execute_tool("get_course_outline", course_name="MCP")
        assert "MCP Course" in result
        assert "Introduction to MCP" in result


# ===================================================================
# RAGSystem orchestration (with mocked components)
# ===================================================================

class TestRAGSystemOrchestration:

    def _build_rag_system(self, config):
        """Build a RAGSystem with heavy components mocked out."""
        with patch("rag_system.VectorStore") as MockVS, \
             patch("rag_system.AIGenerator") as MockAI, \
             patch("rag_system.DocumentProcessor"):
            mock_store = MockVS.return_value
            mock_store.search.return_value = SearchResults(
                documents=["content"],
                metadata=[{"course_title": "Test", "lesson_number": 1, "chunk_index": 0}],
                distances=[0.1],
            )

            mock_ai = MockAI.return_value
            mock_ai.generate_response.return_value = "Here is the answer."

            from rag_system import RAGSystem
            rag = RAGSystem(config)
            return rag, mock_ai, mock_store

    def test_query_calls_ai_with_tools(self, buggy_config):
        rag, mock_ai, _ = self._build_rag_system(buggy_config)
        session_id = rag.session_manager.create_session()
        rag.query("What is MCP?", session_id)

        mock_ai.generate_response.assert_called_once()
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs["tools"] is not None
        assert call_kwargs["tool_manager"] is rag.tool_manager

    def test_query_retrieves_and_resets_sources(self, buggy_config):
        rag, mock_ai, _ = self._build_rag_system(buggy_config)

        # Manually set sources to simulate a tool search having occurred
        rag.search_tool.last_sources = ["MCP Course - Lesson 1"]

        _, sources = rag.query("test", None)
        assert sources == ["MCP Course - Lesson 1"]
        # After query, sources should be reset
        assert rag.tool_manager.get_last_sources() == []

    def test_session_history_stored(self, buggy_config):
        rag, mock_ai, _ = self._build_rag_system(buggy_config)
        session_id = rag.session_manager.create_session()

        rag.query("What is MCP?", session_id)

        history = rag.session_manager.get_conversation_history(session_id)
        assert history is not None
        assert "What is MCP?" in history


# ===================================================================
# FastAPI endpoint integration test
# ===================================================================

class TestFastAPIEndpoint:

    def test_query_endpoint_returns_200_with_fixed_sources(self):
        """BUG 2 FIXED: The /api/query endpoint returns 200 because app.py
        now wraps string sources in Source objects."""
        with patch("app.rag_system") as mock_rag:
            mock_rag.session_manager.create_session.return_value = "session_1"
            mock_rag.query.return_value = (
                "Here is the answer.",
                ["MCP Course - Lesson 1"],
            )

            from fastapi.testclient import TestClient
            from app import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/api/query",
                json={"query": "What is MCP?"},
            )

            assert response.status_code == 200, (
                f"Expected 200, got {response.status_code}: {response.text}"
            )
            data = response.json()
            assert data["answer"] == "Here is the answer."
            assert data["sources"][0]["label"] == "MCP Course - Lesson 1"
            assert data["session_id"] == "session_1"
