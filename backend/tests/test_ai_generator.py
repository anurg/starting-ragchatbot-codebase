"""Tests for AIGenerator with fully mocked Anthropic client.

Bugs exposed:
- Bug 3: system prompt references get_course_outline, but that tool is never registered
"""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic responses
# ---------------------------------------------------------------------------

def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(tool_name: str, tool_input: dict, tool_id: str = "toolu_123"):
    return SimpleNamespace(type="tool_use", name=tool_name, input=tool_input, id=tool_id)


def _make_response(content_blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=content_blocks, stop_reason=stop_reason)


FAKE_TOOLS = [{"name": "search_course_content"}]


# ---------------------------------------------------------------------------
# Tests — Direct responses (no tool use)
# ---------------------------------------------------------------------------

class TestAIGeneratorDirectResponse:

    @patch("ai_generator.anthropic.Anthropic")
    def test_direct_text_response(self, MockAnthropic):
        """When Claude returns text with no tool use, generate_response returns it."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _make_response(
            [_text_block("Python is a programming language.")]
        )

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response(query="What is Python?")

        assert result == "Python is a programming language."
        mock_client.messages.create.assert_called_once()

    @patch("ai_generator.anthropic.Anthropic")
    def test_tools_omitted_when_none(self, MockAnthropic):
        """When tools=None, the API call should NOT include 'tools' key."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _make_response(
            [_text_block("Hello")]
        )

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response(query="hi", tools=None)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_conversation_history_appended(self, MockAnthropic):
        """Conversation history is appended to the system prompt."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _make_response(
            [_text_block("Sure")]
        )

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response(query="Follow up", conversation_history="User: hi\nAssistant: hello")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_kwargs["system"]
        assert "User: hi" in call_kwargs["system"]


# ---------------------------------------------------------------------------
# Tests — Single tool call
# ---------------------------------------------------------------------------

class TestAIGeneratorSingleToolCall:

    @patch("ai_generator.anthropic.Anthropic")
    def test_single_tool_call_returns_final_text(self, MockAnthropic):
        """When Claude requests one tool, execute it and return the synthesized answer."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = [
            _make_response([_tool_use_block("search_course_content", {"query": "MCP"})], stop_reason="tool_use"),
            _make_response([_text_block("MCP enables tool integration.")]),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "MCP allows LLMs to call external tools."

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response(
            query="What is MCP?",
            tools=FAKE_TOOLS,
            tool_manager=tool_manager,
        )

        assert result == "MCP enables tool integration."
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="MCP")
        assert mock_client.messages.create.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_single_tool_call_message_chain(self, MockAnthropic):
        """After one tool call, the follow-up API call has user→assistant(tool_use)→user(tool_result)."""
        mock_client = MockAnthropic.return_value

        tool_block = _tool_use_block("search_course_content", {"query": "agents"}, tool_id="t1")
        mock_client.messages.create.side_effect = [
            _make_response([tool_block], stop_reason="tool_use"),
            _make_response([_text_block("Agents plan and act.")]),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Agent content here"

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response(query="Tell me about agents", tools=FAKE_TOOLS, tool_manager=tool_manager)

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "t1"
        assert messages[2]["content"][0]["content"] == "Agent content here"

    @patch("ai_generator.anthropic.Anthropic")
    def test_single_tool_call_keeps_tools_in_follow_up(self, MockAnthropic):
        """After round 0 tool call, the follow-up API call still includes tools."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = [
            _make_response([_tool_use_block("search_course_content", {"query": "x"})], stop_reason="tool_use"),
            _make_response([_text_block("Answer.")]),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response(query="q", tools=FAKE_TOOLS, tool_manager=tool_manager)

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_call_get_course_outline_gets_not_found(self, MockAnthropic):
        """BUG 3 evidence: when Claude calls get_course_outline the tool_manager
        returns 'Tool not found' because it was never registered."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = [
            _make_response([_tool_use_block("get_course_outline", {"course_name": "MCP"})], stop_reason="tool_use"),
            _make_response([_text_block("I couldn't find that information.")]),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Tool 'get_course_outline' not found"

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response(
            query="What lessons are in the MCP course?",
            tools=FAKE_TOOLS,
            tool_manager=tool_manager,
        )

        tool_manager.execute_tool.assert_called_once_with("get_course_outline", course_name="MCP")


# ---------------------------------------------------------------------------
# Tests — Two sequential tool calls
# ---------------------------------------------------------------------------

class TestAIGeneratorSequentialToolCalls:

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_sequential_tool_calls(self, MockAnthropic):
        """Claude calls get_course_outline, then search_course_content, then answers."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = [
            _make_response(
                [_tool_use_block("get_course_outline", {"course_name": "MCP"}, tool_id="t1")],
                stop_reason="tool_use",
            ),
            _make_response(
                [_tool_use_block("search_course_content", {"query": "tool integration"}, tool_id="t2")],
                stop_reason="tool_use",
            ),
            _make_response([_text_block("The course on tool integration is AI Agents.")]),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = [
            "Lesson 4: Tool Integration",
            "AI Agents course covers tool integration.",
        ]

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response(
            query="Which course covers the same topic as lesson 4 of MCP?",
            tools=FAKE_TOOLS,
            tool_manager=tool_manager,
        )

        assert result == "The course on tool integration is AI Agents."
        assert mock_client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="MCP")
        tool_manager.execute_tool.assert_any_call("search_course_content", query="tool integration")

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_tool_calls_message_chain(self, MockAnthropic):
        """After two tool rounds, the third API call has the full 5-message chain."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = [
            _make_response(
                [_tool_use_block("get_course_outline", {"course_name": "X"}, tool_id="t1")],
                stop_reason="tool_use",
            ),
            _make_response(
                [_tool_use_block("search_course_content", {"query": "topic"}, tool_id="t2")],
                stop_reason="tool_use",
            ),
            _make_response([_text_block("Final answer.")]),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["outline result", "search result"]

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response(query="complex query", tools=FAKE_TOOLS, tool_manager=tool_manager)

        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        messages = third_call_kwargs["messages"]

        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "t1"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"
        assert messages[4]["content"][0]["type"] == "tool_result"
        assert messages[4]["content"][0]["tool_use_id"] == "t2"

    @patch("ai_generator.anthropic.Anthropic")
    def test_tools_removed_after_max_rounds(self, MockAnthropic):
        """After MAX_TOOL_ROUNDS tool calls, the final API call has no tools."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = [
            _make_response([_tool_use_block("search_course_content", {"query": "a"}, tool_id="t1")], stop_reason="tool_use"),
            _make_response([_tool_use_block("search_course_content", {"query": "b"}, tool_id="t2")], stop_reason="tool_use"),
            _make_response([_text_block("Done.")]),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["result a", "result b"]

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response(query="q", tools=FAKE_TOOLS, tool_manager=tool_manager)

        # First follow-up (round 0→1) should have tools
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs

        # Second follow-up (round 1→2) should NOT have tools
        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_no_fourth_api_call_after_max_rounds(self, MockAnthropic):
        """Even if the response after max rounds has tool_use, no further API call is made."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = [
            _make_response([_tool_use_block("search_course_content", {"query": "a"}, tool_id="t1")], stop_reason="tool_use"),
            _make_response([_tool_use_block("search_course_content", {"query": "b"}, tool_id="t2")], stop_reason="tool_use"),
            # Third response is text since tools were removed — Claude can't request tools
            _make_response([_text_block("Best effort answer.")]),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["r1", "r2"]

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response(query="q", tools=FAKE_TOOLS, tool_manager=tool_manager)

        assert result == "Best effort answer."
        assert mock_client.messages.create.call_count == 3


# ---------------------------------------------------------------------------
# Tests — Error handling
# ---------------------------------------------------------------------------

class TestAIGeneratorErrorHandling:

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_exception_sent_back_to_claude(self, MockAnthropic):
        """If execute_tool raises, the error is passed to Claude as a tool_result."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = [
            _make_response([_tool_use_block("search_course_content", {"query": "x"}, tool_id="t1")], stop_reason="tool_use"),
            _make_response([_text_block("Sorry, I encountered an error.")]),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError("connection failed")

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response(query="q", tools=FAKE_TOOLS, tool_manager=tool_manager)

        assert result == "Sorry, I encountered an error."

        # Verify the error was sent back as a tool_result
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        tool_result = second_call_kwargs["messages"][2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert "connection failed" in tool_result["content"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_no_tool_manager_returns_text_even_with_tool_use_stop(self, MockAnthropic):
        """If tool_manager is None but stop_reason is tool_use, return any text present."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _make_response(
            [_text_block("Partial response"), _tool_use_block("search_course_content", {"query": "x"})],
            stop_reason="tool_use",
        )

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response(query="q", tools=FAKE_TOOLS, tool_manager=None)

        assert result == "Partial response"
        mock_client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — System prompt
# ---------------------------------------------------------------------------

class TestAIGeneratorSystemPrompt:

    def test_system_prompt_references_get_course_outline(self):
        """BUG 3 documentation: the static system prompt tells Claude about
        get_course_outline, but rag_system.py never registers that tool."""
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT

    def test_system_prompt_allows_sequential_tool_calls(self):
        """System prompt should allow up to 2 sequential tool calls."""
        assert "One tool call per query maximum" not in AIGenerator.SYSTEM_PROMPT
        assert "2 sequential tool calls" in AIGenerator.SYSTEM_PROMPT
