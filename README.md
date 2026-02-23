# Production-Grade GPT Chatbot

A FastAPI-based GPT chatbot with:

- Tool calling (weather, KB search, grade calculator, live web lookup)
- Streaming responses (`/chat`)
- Conversation memory (short-term rolling window)
- Long-term memory (extra credit, persisted facts)
- Safety enforcement
- Rate limiting
- Structured logging + metrics
- Structured JSON output endpoint (`/chat_json`) (extra credit)
- 20-prompt evaluation harness

---

## Project Structure

    ChatBotBuild/
    ├── app.py
    ├── kb.md
    ├── eval_prompts.json
    ├── run_eval.py
    ├── summarize_eval.py
    ├── requirements.txt
    ├── pytest.ini
    ├── tests/
    └── results/
        ├── transcripts/
        ├── eval_runs/
        ├── metrics.jsonl
        └── ltm_facts.jsonl    (created at runtime if LTM_ENABLED=1)

---

## Installation & Setup

### 1) Clone Repository

    git clone https://github.com/FarmerSamuel-FS/ChatBotBuild.git
    cd ChatBotBuild

### 2) Create Virtual Environment

    python -m venv venv
    source venv/bin/activate

### 3) Install Dependencies

    pip install -r requirements.txt

### 4) Create a .env File

Create a `.env` file in the project root:

    touch .env

Add the following (example values):

    OPENAI_API_KEY=your_openai_key_here
    OPENAI_MODEL=gpt-4o-mini
    TEMP=0.4
    EVAL_MODE=0
    MEMORY_WINDOW=12
    RATE_LIMIT_RPM=60
    LOG_DIR=results

---

## Run the Server

    uvicorn app:app --reload

Open:

    http://127.0.0.1:8000

---

## API Endpoints

### POST /chat (streaming text)

Request body:

    {
      "conversation_id": "demo",
      "user_message": "What are office hours?"
    }

Response: streamed plain text (token-by-token).

### POST /chat_json (structured JSON)

Same request body as `/chat`.

Response shape:

    {
      "conversation_id": "demo",
      "answer": "…",
      "tool_calls": ["kb_search"],
      "latency_ms": 1234,
      "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
      "cost_usd": 0.00003,
      "ltm_facts_used": ["Name: Sam"]
    }

---

## Run Evaluation Suite

Make sure the server is running first.

### Run All Prompts

    python run_eval.py

This generates:

    results/transcripts/<conversation_id>.txt
    results/eval_runs/<conversation_id>.json
    results/metrics.jsonl

### Summarize Results

    python summarize_eval.py results/eval_runs/*.json

---

## Tools Implemented

### get_weather

- Uses Open-Meteo API (no key)
- Returns temperature and wind speed

### kb_search

Searches `kb.md` for:

- Office hours
- Grading percentages

### calculate_grade

Weighted formula:

    Projects: 60%
    Exams: 30%
    Participation: 10%

### web_lookup

- Current US President → USA.gov
- Other live facts → DuckDuckGo Instant Answer API

---

## Safety Enforcement

Automatic refusal / safe handling for:

- Explosives / bomb instructions
- Self-harm content (returns crisis resources)
- Secret/API key storage
- Guess-without-tools prompts

Secrets are redacted in logs.

---

## Memory

### Short-term memory

Conversation history stored per `conversation_id` (rolling `MEMORY_WINDOW` messages).

### Long-term memory (extra credit)

If enabled (`LTM_ENABLED=1`), the server stores simple “facts” (example: name, major)
in:

    results/ltm_facts.jsonl

Example:

    User: Remember that my name is Sam.
    User: What is my name?
    Bot: Your name is Sam.

---

## Metrics Logging

Each request appends a JSON line into:

    results/metrics.jsonl

Fields include:

- latency_ms
- tool_calls
- token usage
- cost_usd (if configured)
- ltm_used (facts retrieved)

---

## Tech Stack

- Python 3.10+
- FastAPI
- OpenAI API (Python SDK)
- HTTPX
- Uvicorn
- python-dotenv
- orjson
- pytest

---

## Assignment Coverage Checklist

- Tool usage (function calling)
- Conversation state (memory)
- Streaming responses (`/chat`)
- Safety enforcement
- Rate limiting
- Metrics logging (latency, tokens, cost estimate)
- Evaluation harness (20 prompts)
- long-term memory + structured JSON output (`/chat_json`)

---
