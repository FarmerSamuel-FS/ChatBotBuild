# Production-Grade GPT Chatbot

A FastAPI-based GPT chatbot with:

- Tool calling (weather, KB search, grade calculator, live web lookup)
- Streaming responses
- Conversation memory
- Safety enforcement
- Rate limiting
- Structured logging
- 20-prompt evaluation harness

---

# Project Structure

```
ChatBotBuild/
├── app.py
├── kb.md
├── eval_prompts.json
├── run_eval.py
├── summarize_eval.py
├── requirements.txt
└── results/
    ├── transcripts/
    ├── eval_runs/
    └── metrics.jsonl
```

---

# Installation & Setup

## 1. Clone Repository

```bash
git clone https://github.com/FarmerSamuel-FS/ChatBotBuild.git
cd ChatBotBuild
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If installing manually:

```bash
pip install fastapi uvicorn openai httpx python-dotenv orjson pydantic
```

---

## 4. Create .env File

Create a `.env` file in the root directory:

```bash
touch .env
```

Add the following:

```
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
TEMP=0.4
EVAL_MODE=0
MEMORY_WINDOW=12
RATE_LIMIT_RPM=60
LOG_DIR=results
```

---

# Run the Server

```bash
uvicorn app:app --reload
```

Open your browser:

```
http://127.0.0.1:8000
```

---

# Run Evaluation Suite

## Run All 20 Prompts

Make sure the server is running first.

```bash
python run_eval.py
```

This generates:

```
results/transcripts/<conversation_id>.txt
results/eval_runs/<conversation_id>.json
results/metrics.jsonl
```

---

## Summarize Results

```bash
python summarize_eval.py results/eval_runs/*.json
```

---

# Example Evaluation Output

```
Total prompts: 20
Success rate: 100.0%
TS%: 100.0
Latency (ms): avg = 1997 | p50 = 2261 | p95 = 3366 | max = 3698
Guess/without-tools prompts: refusal behavior ✅
```

---

# Tools Implemented

## get_weather

- Uses Open-Meteo API
- Returns temperature and wind speed

## kb_search

Searches `kb.md` for:

- Office hours
- Grading percentages

## calculate_grade

Weighted formula:

```
Projects: 60%
Exams: 30%
Participation: 10%

Final = (P × 0.60) + (E × 0.30) + (Pa × 0.10)
```

## web_lookup

- Current US President → USA.gov
- Other live facts → DuckDuckGo API

---

# Safety Enforcement

Automatic refusal for:

- Explosives / bomb instructions
- Self-harm content (returns crisis resources)
- Secret/API key storage
- Guess-without-tools prompts

---

# Memory

Conversation history stored per `conversation_id`.

Example:

```
User: Remember my name is Sam.
User: What is my name?
Bot: Your name is Sam.
```

---

# Knowledge Base (kb.md)

```
## Office Hours
Mon–Thu 2–4pm, Room 301

## Grading
Projects: 60%
Exams: 30%
Participation: 10%
```

---

# Metrics Logging

Each request logs:

- latency_ms
- tool_calls
- token usage
- timestamp

Saved in:

```
results/metrics.jsonl
```

---

# Full Workflow

```bash
# Start server
uvicorn app:app --reload

# Run evaluation
python run_eval.py

# Summarize results
python summarize_eval.py results/eval_runs/*.json
```

---

# Tech Stack

- Python 3.10+
- FastAPI
- OpenAI API
- HTTPX
- Uvicorn
- python-dotenv
- orjson
- Pydantic

---

# Assignment Coverage

- Tool usage
- Memory
- Streaming responses
- Safety enforcement
- Evaluation harness
- Latency tracking
- Structured logging
- Policy correctness checks

---

# Author

Week 2 Production Chatbot Assignment
