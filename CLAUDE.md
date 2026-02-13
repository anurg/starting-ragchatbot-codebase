# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. FastAPI backend with vanilla JS frontend, using ChromaDB for vector search and Claude API with tool calling for response generation.

## Commands

```bash
# Install dependencies (uses uv, not pip)
uv sync

# Run the server (from project root)
cd backend && uv run uvicorn app:app --reload --port 8000

# Or use the startup script
./run.sh
```

- Web UI: http://localhost:8000
- Swagger docs: http://localhost:8000/docs

## Environment

Requires `.env` file with `ANTHROPIC_API_KEY`. Python 3.13+ with `uv` package manager.

## Architecture

```
Frontend (vanilla JS) → FastAPI → RAGSystem → AIGenerator (Claude API w/ tool use)
                                                    ↓
                                            CourseSearchTool
                                                    ↓
                                          VectorStore (ChromaDB)
```

**Request flow for `/api/query`:** Query arrives → RAGSystem looks up/creates session → AIGenerator calls Claude with tool definitions → Claude decides whether to invoke CourseSearchTool → tool performs semantic search on VectorStore → results fed back to Claude for synthesis → response returned with tracked sources.

### Key backend modules

- **`rag_system.py`** — Orchestrator that wires all components together
- **`ai_generator.py`** — Claude API integration with tool-calling loop (temperature=0, max_tokens=800)
- **`search_tools.py`** — ToolManager registry and CourseSearchTool implementation (semantic search with optional course/lesson filtering)
- **`vector_store.py`** — ChromaDB wrapper with two collections: `course_catalog` (metadata) and `course_content` (chunked text)
- **`document_processor.py`** — Parses `docs/*.txt` files into chunks (800 chars, 100 overlap, sentence-boundary aware)
- **`config.py`** — All tunable constants (model names, chunk sizes, search limits)
- **`session_manager.py`** — In-memory conversation history (max 2 exchanges per session)

### Document format

Course text files in `docs/` follow a specific format: header lines (`Course Title:`, `Course Link:`, `Course Instructor:`) followed by lessons marked with `Lesson N: [title]` and `Lesson Link: [url]`.

### API response shape

```json
{"answer": "markdown string", "sources": [{"label": "string", "link": "url|null"}], "session_id": "string"}
```

### Frontend

Vanilla JS with markdown rendering (marked.js). State managed via global `currentSessionId`. Suggested questions are hardcoded. Responsive layout with CSS grid (sidebar collapses on mobile).
