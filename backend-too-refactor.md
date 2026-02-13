1 Refactor @backend/ai_generator.py to support sequential tool calling where Claude can
make up to 2 tool calls in separate API rounds.
2
3
4 Current behavior:
5 - Claude makes 1 tool call → tools are removed from API params → final response
6 - If Claude wants another tool call after seeing results, it can't (gets empty
response)
7
8
9 Desired behavior:
10 - Each tool call should be a separate API request where Claude can reason about
previous results
11 - Support for complex queries requiring multiple searches for comparisons, multi-part
questions, or when information from different courses/lessons is needed
12

14 Example flow:
15 1. User: "Search for a course that discusses the same topic as lesson 4 of course X"
16 2. Claude: get course outline for course X → gets title of lesson 4
17 3. Claude: uses the title to search for a course that discusses the same topic → returns course information
18 4. Claude: provides complete answer

21 Requirements:
22 - Maximum 2 sequential rounds per user query
23 - Terminate when: (a) 2 rounds completed, (b) Claude's response has no tool_use
blocks, or (c) tool call fails
24 - Preserve conversation context between rounds
25 - Handle tool execution errors gracefully
26
27
28 Notes:
29 - update the system prompt in @backend/ai_generator.py
30 - update the test @backend/tests/test_ai_generator.py
31 - Write tests that verify the external behavior (API calls made, tools executed,
results returned) rather than internal state details.

Use two parallel subagents to brainstorm possible plans. Do not implement any code.