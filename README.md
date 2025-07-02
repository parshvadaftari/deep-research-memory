# Deep Research Memory

An AI-powered research assistant that retrieves context from long-term memory (Mem0) and past conversations to provide grounded, well-structured answers with proper citations.

## Features

- **Intelligent Memory Search**: Search through stored research memories with semantic understanding
- **Real-time Streaming**: Get responses streamed in real-time with WebSocket support
- **Citation Tracking**: View source citations and references for all responses
- **Contextual Responses**: AI-powered responses that understand the context of your research
- **Memory Persistence**: Store and retrieve research memories across sessions
- **Modern UI**: Clean, responsive interface built with Next.js and Tailwind CSS

## Technology Stack

- **Frontend**: Next.js 15 with TypeScript and React 19
- **Backend**: FastAPI (Python 3.11+)
- **Database**: ChromaDB (vector database) + SQLite (Conversations)
- **AI Models**: OpenAI GPT-4.1-mini via LangChain
- **Memory System**: Mem0AI for advanced memory management
- **UI Components**: Radix UI + Tailwind CSS
- **Real-time**: WebSocket for streaming responses

## Modular Backend Structure

The backend is organized into modular subfolders for each agent type:

- `simple_agent/` – Simple research agent logic and endpoints
- `sequential_agent/` – LangGraph sequential agent logic and endpoints
- `multiagent/` – LangGraph multiagent logic and endpoints


### Why GPT-4.1-mini?

We chose GPT-4.1-mini for its optimal balance of performance, cost, and speed:

- **Cost**: 15x more affordable than GPT-4o
- **Speed**: 2x faster for real-time streaming
- **Research Focus**: Excellent at memory retrieval and citation tasks

Perfect for long research sessions while maintaining high-quality responses.

## Getting Started

### Prerequisites

- Git
- Node.js 18+ (for frontend)
- Python 3.11+ (for backend)
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone git@github.com:parshvadaftari/deep-research-memory.git

cd deep-research-memory
```

### Environment Setup

#### Backend Variables (.env)
```plaintext
OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Application

#### Frontend (Next.js)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install --legacy-peer-deps

# Start development server
npm run dev
```

The frontend will be available at http://localhost:3000

#### Backend (FastAPI)

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or using the built-in run script:
```bash
cd backend
python main.py
```

## Testing

The project includes a comprehensive test suite covering both **retrieval** and **generation** components of the deep research memory system.

### Prerequisites

Install test dependencies:
```bash
cd backend
pip install -r requirements-test.txt
```

### Running Tests

#### Basic Test Execution
```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_retrieval.py
```

## API Endpoints

### HTTP Endpoints
- `GET /` - Root endpoint
- `POST /api/v1/search` - Search endpoint (streaming response)

### WebSocket Endpoints
- `WS /ws` - WebSocket endpoint for real-time search

### Key Features

- **Real-time Streaming**: WebSocket-based streaming for immediate feedback
- **Memory Management**: Persistent storage of research memories
- **Citation System**: Track and display source references
- **Responsive Design**: Works on desktop and mobile devices
- **Dark/Light Mode**: Theme switching support

## License

This project is open source and available under the MIT License.

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Mem0AI Documentation](https://docs.mem0.ai/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Radix UI Documentation](https://www.radix-ui.com/)
