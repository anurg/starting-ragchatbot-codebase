"""Tests for CourseSearchTool and ToolManager.

Bugs exposed:
- Bug 2: get_last_sources() returns plain strings, not Source objects
- Bug 3: get_course_outline tool is never registered
"""

from unittest.mock import MagicMock

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# ===================================================================
# CourseSearchTool.execute
# ===================================================================

class TestCourseSearchToolExecute:

    def test_execute_returns_formatted_output(self, mock_vector_store, sample_search_results):
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="MCP tools")

        assert "[MCP Course - Lesson 1]" in result
        assert "[AI Agents - Lesson 3]" in result
        assert "MCP allows LLMs" in result

    def test_execute_populates_last_sources(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="MCP tools")

        assert len(tool.last_sources) == 2
        assert tool.last_sources[0] == "MCP Course - Lesson 1"
        assert tool.last_sources[1] == "AI Agents - Lesson 3"

    def test_execute_forwards_course_filter(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="tools", course_name="MCP")

        mock_vector_store.search.assert_called_once_with(
            query="tools", course_name="MCP", lesson_number=None
        )

    def test_execute_forwards_lesson_filter(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="tools", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="tools", course_name=None, lesson_number=2
        )

    def test_execute_empty_results_returns_friendly_message(self, mock_vector_store, empty_search_results):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_empty_results_includes_filter_info(self, mock_vector_store, empty_search_results):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="x", course_name="MCP", lesson_number=5)

        assert "MCP" in result
        assert "lesson 5" in result

    def test_execute_error_returns_error_string(self, mock_vector_store, error_search_results):
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="anything")

        assert "Search error" in result

    def test_format_results_without_lesson_number(self, mock_vector_store):
        """When lesson_number is None the header omits it."""
        no_lesson = SearchResults(
            documents=["Some content"],
            metadata=[{"course_title": "Intro Course", "chunk_index": 0}],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = no_lesson
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")

        assert "[Intro Course]" in result
        assert "Lesson" not in result


# ===================================================================
# ToolManager
# ===================================================================

class TestToolManager:

    def test_register_and_get_definitions(self, mock_vector_store):
        tm = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tm.register_tool(tool)

        defs = tm.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "search_course_content"

    def test_execute_registered_tool(self, mock_vector_store):
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_vector_store))

        result = tm.execute_tool("search_course_content", query="MCP")
        assert "MCP allows LLMs" in result

    def test_execute_nonexistent_tool_returns_not_found(self):
        """Calling a completely unknown tool returns an error string."""
        tm = ToolManager()
        mock_store = MagicMock()
        tm.register_tool(CourseSearchTool(mock_store))

        result = tm.execute_tool("totally_fake_tool", foo="bar")
        assert "not found" in result.lower()

    def test_get_course_outline_tool_registered_and_works(self, mock_vector_store):
        """BUG 3 FIXED: get_course_outline is now a registered tool."""
        from search_tools import CourseOutlineTool

        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_vector_store))
        tm.register_tool(CourseOutlineTool(mock_vector_store))

        result = tm.execute_tool("get_course_outline", course_name="MCP")
        assert "MCP Course" in result
        assert "Introduction to MCP" in result

    def test_get_last_sources_returns_plain_strings(self, mock_vector_store):
        """BUG 2 evidence: sources are plain strings, not Source(label, link) objects.
        QueryResponse in app.py expects List[Source]."""
        tm = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tm.register_tool(tool)

        tm.execute_tool("search_course_content", query="MCP")
        sources = tm.get_last_sources()

        assert len(sources) > 0
        # Each source is a plain str, NOT a dict or Source model
        for s in sources:
            assert isinstance(s, str), f"Expected str, got {type(s)}"

    def test_reset_sources_clears_state(self, mock_vector_store):
        tm = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tm.register_tool(tool)

        tm.execute_tool("search_course_content", query="MCP")
        assert len(tm.get_last_sources()) > 0

        tm.reset_sources()
        assert tm.get_last_sources() == []
