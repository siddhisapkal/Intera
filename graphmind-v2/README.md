# GraphMind v2

**AI chat with persistent memory powered by Neo4j knowledge graphs.**

## Features

- 🧠 **Graph Memory** — every conversation builds a personal knowledge graph per user
- 🌐 **DuckDuckGo Search** — real-time web research integrated into responses
- 🏢 **Company Analyzer** — deep-dive interview prep for any company
- 📄 **Resume Analyzer** — upload a PDF resume; skills, companies, education auto-saved to your graph
- 🔐 **User Isolation** — each user has a fully isolated memory space
- 💬 **Persistent Chat** — conversations saved and resumable
- 🤖 **Multi-LLM** — supports Gemini + Groq with automatic fallback

---

## Quick Start (Docker)

### 1. Clone and configure

```bash
git clone <your-repo>
cd graphmind-v2
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY and/or GROQ_API_KEY
```

### 2. Start everything

```bash
docker compose up -d
```

### 3. Open the UI

```
http://localhost:8000
```

That's it. Neo4j and Redis start automatically.

---

## Local Development (without Docker)

### Prerequisites

- Python 3.11+
- Neo4j 5.x running locally
- Redis (optional — falls back to in-memory)

### Install

```bash
pip install -r backend/requirements.txt
```

### Configure

```bash
cp .env.example .env
# Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GEMINI_API_KEY
```

### Run

```bash
uvicorn backend.main:app --reload --port 8000
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register new account |
| POST | `/auth/login` | Login |
| POST | `/auth/logout` | Logout |
| GET | `/auth/me` | Current user |
| POST | `/chat` | Send message (with memory) |
| GET | `/chat/conversations` | List conversations |
| GET | `/chat/history/{id}` | Conversation history |
| POST | `/planner/company` | Company analyzer |
| POST | `/resume/analyze` | Upload & analyze resume PDF |
| GET | `/graph/memory/{user_id}` | User's graph memory |
| GET | `/graph/view/{user_id}` | Graph visualization data |
| GET | `/memory/ephemeral/{user_id}` | Working memory |
| GET | `/profile/summary/{user_id}` | Strength/weakness profile |
| DELETE | `/memory/reset/{user_id}` | Reset all user memory |
| GET | `/health` | Service health |

---

## Architecture

```
GraphMind v2
├── backend/
│   ├── main.py              # FastAPI app, all routes
│   ├── resume_analyzer.py   # PDF resume parsing → graph signals
│   ├── gemini_chat.py       # LLM integration (Gemini / Groq)
│   ├── web_research.py      # DuckDuckGo search
│   ├── graph/
│   │   └── service.py       # Neo4j graph memory service
│   ├── vector_store.py      # FAISS semantic search
│   ├── auth_store.py        # User auth & sessions
│   └── ...
├── frontend/
│   └── index.html           # Full SPA (no framework, pure JS)
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEO4J_URI` | ✅ | Neo4j connection URI |
| `NEO4J_USER` | ✅ | Neo4j username |
| `NEO4J_PASSWORD` | ✅ | Neo4j password |
| `GEMINI_API_KEY` | ✅* | Google Gemini API key |
| `GROQ_API_KEY` | ✅* | Groq API key |
| `REDIS_URL` | ❌ | Redis URL (optional) |
| `FAISS_INDEX_DIR` | ❌ | FAISS storage path |

*At least one of Gemini or Groq is required.
