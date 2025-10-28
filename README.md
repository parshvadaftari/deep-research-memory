# Deep Research Memory

An AI-powered research assistant that retrieves context from long-term memory and past conversations to provide grounded, well-structured answers with proper citations.

## ✨ Key Features

- **Semantic Memory Search** – Find relevant research context instantly
- **Real-time Streaming** – WebSocket-powered responses as they generate
- **Smart Citations** – Every response includes source references
- **Persistent Memory** – Your research stays available across sessions
- **Modern Interface** – Responsive UI with dark/light mode support

## 🛠️ Tech Stack

**Frontend**
- Next.js 15 + TypeScript + React 19
- Tailwind CSS + Radix UI

**Backend**
- FastAPI (Python 3.11+)
- ChromaDB (vectors) + SQLite (conversations)
- OpenAI GPT-4.1-mini via LangChain
- Mem0AI for memory management

**Architecture**
- `simple_agent/` – Basic research agent
- `sequential_agent/` – LangGraph sequential flow
- `multiagent/` – LangGraph multi-agent system

> **Why GPT-4.1-mini?** 15x more cost-effective than GPT-4o, 2x faster streaming, and optimized for research tasks.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.11+
- OpenAI API key

### Setup

```bash
# Clone repository
git clone git@github.com:parshvadaftari/deep-research-memory.git
cd deep-research-memory
```

Create `.env` in backend directory:
```plaintext
OPENAI_API_KEY=your_openai_api_key_here
```

### Frontend

```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
```

Visit http://localhost:3000

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

API available at http://localhost:8000

## 🧪 Testing

```bash
cd backend
pip install -r requirements-test.txt
python -m pytest -v
```

## 📡 API Endpoints

**HTTP**
- `GET /` – Health check
- `POST /api/v1/search` – Search with streaming

**WebSocket**
- `WS /ws` – Real-time search connection
