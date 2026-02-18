# app.py
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import orjson
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Production-Grade GPT Chatbot")


# Basic setup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "kb.md")
TEMP = float(os.getenv("TEMP", "0.4"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EVAL_MODE = os.getenv("EVAL_MODE", "0").strip().lower() in ("1", "true", "yes", "on")
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "12"))
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))
LOG_DIR = os.getenv("LOG_DIR", "results")

SYSTEM_PROMPT = (
    "You are a task-oriented assistant.\n"
    "Use tools when they help, and don't guess if a tool can answer.\n"
    "Tools:\n"
    "- Weather questions -> get_weather\n"
    "- Office hours / grading policy / rubric / percentages -> kb_search\n"
    "- If user provides scores -> calculate_grade\n"
    "- Live facts / current roles -> web_lookup\n"
    "If the user asks you to guess or says 'without tools', do NOT guess.\n"
    "If the user asks for a calculation, finish it and include the final value.\n"
    "If a request is unsafe, refuse briefly.\n"
)


# Memory + rate limiting
CONV: Dict[str, List[Dict[str, Any]]] = {}
RATE: Dict[str, List[float]] = {}


# Safety checks
UNSAFE_PATTERNS = [
    re.compile(r"\bhow to (make|build)\b.*\b(bomb|explosive)\b", re.I),
]
SELF_HARM_PATTERNS = [
    re.compile(r"\b(suicide|kill myself|self-harm|end my life)\b", re.I),
]
SECRET_PATTERNS = [re.compile(r"sk-[A-Za-z0-9_\-]{20,}")]

def is_unsafe(text: str) -> bool:
    return any(p.search(text) for p in UNSAFE_PATTERNS)

def is_self_harm(text: str) -> bool:
    return any(p.search(text) for p in SELF_HARM_PATTERNS)

def redact_secrets(text: str) -> str:
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED_SECRET]", text)
    return text

def rate_limit(ip: str) -> None:
    now = time.time()
    RATE.setdefault(ip, [])
    RATE[ip] = [t for t in RATE[ip] if now - t < 60]
    if len(RATE[ip]) >= RATE_LIMIT_RPM:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (RPM).")
    RATE[ip].append(now)


# Tools

async def get_weather(city: str) -> Dict[str, Any]:
    """
    Gets real weather from Open-Meteo (free, no API key).
    """
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    forecast_url = "https://api.open-meteo.com/v1/forecast"

    async with httpx.AsyncClient(timeout=15) as client:
        geo = await client.get(geo_url, params={"name": city, "count": 1})
        gj = geo.json()
        if not gj.get("results"):
            return {"error": f"City not found: {city}"}

        lat = gj["results"][0]["latitude"]
        lon = gj["results"][0]["longitude"]

        wx = await client.get(
            forecast_url,
            params={"latitude": lat, "longitude": lon, "current": "temperature_2m,wind_speed_10m"},
        )
        wj = wx.json()
        cur = wj.get("current", {})
        return {
            "city": city,
            "temperature_c": cur.get("temperature_2m"),
            "wind_speed_kph": cur.get("wind_speed_10m"),
            "source": "open-meteo.com",
        }

def kb_search(query: str) -> Dict[str, Any]:
    """
    Searches kb.md using '##' section headings.
    """
    if not os.path.exists(KB_PATH):
        return {"results": {"error": f"kb.md not found at {KB_PATH}"}}

    q = (query or "").lower().strip()

    # Normalize common phrases
    if re.search(r"\b(grades?|grading|percent|percentage|rubric|points|weight|weights)\b", q):
        q = "grading"
    elif re.search(r"\b(office hours|office)\b|\bhours\b", q):
        q = "office hours"

    text = open(KB_PATH, "r", encoding="utf-8").read()
    hits: Dict[str, str] = {}

    parts = text.split("##")
    for part in parts[1:]:
        lines = [ln.strip() for ln in part.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        title = lines[0]
        body = "\n".join(lines[1:]).strip()
        blob = (title + "\n" + body).lower()
        if q in blob:
            hits[title] = body

    return {"results": hits or {"note": "no match"}, "kb_path": KB_PATH}

def calculate_grade(project: float, exams: float, participation: float) -> Dict[str, Any]:
    """
    Weighted grade:
      Projects 60%, Exams 30%, Participation 10%
    Inputs are percentages (0-100).
    """
    # Small helper so weird inputs don't break the math
    def clamp(x: float) -> float:
        return max(0.0, min(100.0, float(x)))

    p = clamp(project)
    e = clamp(exams)
    pa = clamp(participation)

    final_pct = p * 0.60 + e * 0.30 + pa * 0.10
    return {
        "final_percentage": round(final_pct, 2),
        "weights": {"projects": 0.60, "exams": 0.30, "participation": 0.10},
        "inputs": {"project": p, "exams": e, "participation": pa},
    }


# Web lookup tool
_USA_GOV_PRES_URL = "https://www.usa.gov/presidents"

async def _lookup_us_president_via_usagov() -> Dict[str, Any]:
    """
    Simple parse from USA.gov page for 'current president'.
    Not perfect, but good enough for this class project.
    """
    async with httpx.AsyncClient(timeout=15, headers={"User-Agent": "Mozilla/5.0"}) as client:
        r = await client.get(_USA_GOV_PRES_URL)
        html = r.text

    m = re.search(
        r"current president of the United States is\s+([A-Z][A-Za-z .'\-]+)\.",
        html,
        re.I,
    )
    name = m.group(1).strip() if m else ""

    m2 = re.search(r"sworn into office on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", html, re.I)
    sworn = m2.group(1).strip() if m2 else ""

    if not name:
        return {
            "query": "current president of the United States",
            "answer": "",
            "sources": [{"title": "USA.gov Presidents", "url": _USA_GOV_PRES_URL}],
            "error": "Could not parse president name from USA.gov.",
        }

    answer = f"The current president of the United States is {name}."
    if sworn:
        answer += f" Sworn into office on {sworn}."

    return {
        "query": "current president of the United States",
        "answer": answer,
        "sources": [{"title": "USA.gov Presidents", "url": _USA_GOV_PRES_URL}],
    }

async def web_lookup(query: str) -> Dict[str, Any]:
    """
    Live facts lookup:
    - If asking about current US president -> USA.gov
    - Otherwise -> DuckDuckGo Instant Answer JSON (free)
    """
    q = (query or "").strip()

    if re.search(r"\b(current|who is)\b.*\bpresident\b.*\b(united states|usa|u\.s\.)\b", q, re.I):
        return await _lookup_us_president_via_usagov()

    ddg_url = "https://api.duckduckgo.com/"
    params = {
        "q": q,
        "format": "json",
        "no_html": "1",
        "no_redirect": "1",
        "skip_disambig": "1",
    }

    async with httpx.AsyncClient(timeout=15, headers={"User-Agent": "Mozilla/5.0"}) as client:
        r = await client.get(ddg_url, params=params)

    try:
        data = r.json()
    except Exception:
        return {"query": q, "answer": "", "sources": [], "error": "Non-JSON response from DuckDuckGo."}

    sources: List[Dict[str, str]] = []
    if data.get("AbstractURL"):
        sources.append({"title": data.get("Heading") or "Abstract source", "url": data["AbstractURL"]})

    for item in data.get("Results", [])[:3]:
        if item.get("FirstURL") and item.get("Text"):
            sources.append({"title": item["Text"], "url": item["FirstURL"]})

    def flatten_related(rt: Any) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if isinstance(rt, list):
            for x in rt:
                out.extend(flatten_related(x))
        elif isinstance(rt, dict):
            if rt.get("FirstURL") and rt.get("Text"):
                out.append({"title": rt["Text"], "url": rt["FirstURL"]})
            if "Topics" in rt:
                out.extend(flatten_related(rt["Topics"]))
        return out

    related = flatten_related(data.get("RelatedTopics", []))
    sources.extend(related[:3])

    answer = (data.get("Answer") or data.get("AbstractText") or "").strip()

    return {
        "query": q,
        "answer": answer,
        "sources": sources[:5],
        "note": "If answer is empty, use the sources or say you can't verify.",
    }

TOOL_FNS = {
    "get_weather": get_weather,
    "kb_search": kb_search,
    "calculate_grade": calculate_grade,
    "web_lookup": web_lookup,
}

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city using Open-Meteo (free).",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_search",
            "description": "Search kb.md for office hours / grading policy / percentages.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_grade",
            "description": "Compute weighted grade from project/exam/participation percentages (0-100).",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "number"},
                    "exams": {"type": "number"},
                    "participation": {"type": "number"},
                },
                "required": ["project", "exams", "participation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_lookup",
            "description": "Look up live facts (special-case US president via USA.gov). Returns answer + sources.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


# OpenAI client

_client: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# Helpers

def convo_messages(cid: str) -> List[Dict[str, Any]]:
    hist = CONV.get(cid, [])[-MEMORY_WINDOW:]
    return [{"role": "system", "content": SYSTEM_PROMPT}] + hist

def append_msg(cid: str, role: str, content: str) -> None:
    CONV.setdefault(cid, [])
    CONV[cid].append({"role": role, "content": content})

def log_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "ab") as f:
        f.write(orjson.dumps(record) + b"\n")


# Simple tool router
WEATHER_RE = re.compile(r"\b(weather|temperature|forecast)\b", re.I)
KB_RE = re.compile(r"\b(office hours|office|grading|grades?|percent|percentage|rubric|points|weight|weights)\b", re.I)
SCORE_WORDS_RE = re.compile(r"\b(project|projects|exam|exams|participation)\b", re.I)
CURRENT_FACT_RE = re.compile(r"\b(current|today|latest|who is|president|prime minister|secretary of state)\b", re.I)
QUESTIONISH_RE = re.compile(r"\?\s*$|\b(what|when|where|who|how|can you|could you|tell me)\b", re.I)
PROFILE_SET_RE = re.compile(
    r"\b(remember|my name is|i am|i'm|call me|my major is|i major in|my major)\b",
    re.I,
)
def choose_forced_tool(user_text: str) -> Optional[str]:
    t = (user_text or "").strip()
    tl = t.lower()

    # 1) If user is setting personal memory, do NOT force tools.

    if PROFILE_SET_RE.search(t):
        return None

    # 2) If user says "without tools" / "guess", do NOT force tools 
    if re.search(r"\b(without tools|no tools|guess)\b", tl):
        return None

    # 3) Weather questions: usually safe to force
    if WEATHER_RE.search(tl):
        return "get_weather"

    # 4) KB questions

    if KB_RE.search(tl) and (QUESTIONISH_RE.search(t) or re.search(r"\b(what|when|where|hours|percent|percentage|rubric|grading)\b", tl)):
        return "kb_search"

    # 5) Scores -> calculate
    if SCORE_WORDS_RE.search(tl) and re.search(r"\d", tl):
        return "calculate_grade"

    # 6) Current facts -> web lookup
    if CURRENT_FACT_RE.search(tl):
        return "web_lookup"

    return None

# Web UI
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>GPT Chatbot</title>
  <style>
    :root{
      --bg: #0b0f17;
      --panel: #0f1623;
      --border: rgba(255,255,255,.08);
      --text: rgba(255,255,255,.92);
      --muted: rgba(255,255,255,.65);
      --user: #1f6feb;
      --bot: #1b2436;
      --shadow: 0 10px 30px rgba(0,0,0,.35);
      --radius: 18px;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: radial-gradient(1200px 600px at 20% 0%, rgba(31,111,235,.18), transparent 55%),
                  radial-gradient(900px 500px at 90% 10%, rgba(120,70,255,.14), transparent 60%),
                  var(--bg);
      color:var(--text);
      height:100vh;
      display:flex;
      align-items:center;
      justify-content:center;
      padding:20px;
    }
    .app{
      width:min(980px, 100%);
      height:min(84vh, 820px);
      background: rgba(15,22,35,.75);
      backdrop-filter: blur(10px);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
      display:flex;
      flex-direction:column;
      overflow:hidden;
    }
    .topbar{
      padding:14px 16px;
      border-bottom:1px solid var(--border);
      display:flex;
      align-items:center;
      gap:10px;
    }
    .dot{
      width:10px;height:10px;border-radius:50%;
      background: rgba(255,255,255,.14);
    }
    .title{
      display:flex;flex-direction:column;gap:2px;
      margin-left:6px;
    }
    .title strong{font-size:14px;letter-spacing:.2px}
    .title span{font-size:12px;color:var(--muted)}
    .chat{
      flex:1;
      padding:18px;
      overflow:auto;
      scroll-behavior:smooth;
    }
    .row{
      display:flex;
      margin:10px 0;
      gap:10px;
    }
    .row.user{justify-content:flex-end}
    .bubble{
      max-width:min(720px, 86%);
      padding:12px 14px;
      border-radius: var(--radius);
      border:1px solid var(--border);
      line-height:1.35;
      font-size:14px;
      white-space:pre-wrap;
      word-wrap:break-word;
    }
    .bubble.user{
      background: linear-gradient(180deg, rgba(31,111,235,.95), rgba(31,111,235,.78));
      border-color: rgba(31,111,235,.35);
    }
    .bubble.bot{
      background: rgba(27,36,54,.9);
    }
    .composer{
      padding:14px;
      border-top:1px solid var(--border);
      background: rgba(10,14,22,.35);
      display:flex;
      gap:10px;
      align-items:flex-end;
    }
    textarea{
      flex:1;
      resize:none;
      min-height:44px;
      max-height:140px;
      padding:12px 12px;
      border-radius: 14px;
      border:1px solid var(--border);
      outline:none;
      background: rgba(8,10,16,.6);
      color: var(--text);
      font-size:14px;
      line-height:1.35;
    }
    textarea:focus{
      border-color: rgba(31,111,235,.55);
      box-shadow: 0 0 0 4px rgba(31,111,235,.12);
    }
    button{
      border:0;
      cursor:pointer;
      padding:12px 14px;
      border-radius: 14px;
      font-weight:600;
      color:white;
      background: linear-gradient(180deg, rgba(31,111,235,1), rgba(31,111,235,.78));
      box-shadow: 0 8px 18px rgba(31,111,235,.25);
      min-width:86px;
    }
    button:disabled{
      cursor:not-allowed;
      opacity:.55;
      box-shadow:none;
    }
    .typing{
      display:inline-flex;
      gap:6px;
      align-items:center;
    }
    .typing i{
      width:6px;height:6px;border-radius:50%;
      background: rgba(255,255,255,.55);
      display:inline-block;
      animation: blink 1.2s infinite;
    }
    .typing i:nth-child(2){animation-delay:.15s}
    .typing i:nth-child(3){animation-delay:.3s}
    @keyframes blink{
      0%, 80%, 100%{opacity:.2; transform: translateY(0)}
      40%{opacity:1; transform: translateY(-2px)}
    }
    .hint{
      padding:0 14px 14px;
      font-size:12px;
      color: var(--muted);
    }
    code.k{
      padding:.12rem .35rem;
      border:1px solid var(--border);
      border-radius:8px;
      background: rgba(255,255,255,.06);
      color: rgba(255,255,255,.9);
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="topbar">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
      <div class="title">
        <strong>GPT Chatbot</strong>
        <span>Streaming • tools • memory</span>
      </div>
    </div>

    <div id="chat" class="chat"></div>

    <div class="composer">
      <textarea id="msg" placeholder="Type a message... (Enter to send, Shift+Enter for newline)"></textarea>
      <button id="sendBtn" onclick="send()">Send</button>
    </div>
    <div class="hint">
      Tip: press <code class="k">Enter</code> to send, <code class="k">Shift</code>+<code class="k">Enter</code> for newline.
    </div>
  </div>

<script>
  const cid = "demo";
  const chatEl = document.getElementById("chat");
  const msgEl = document.getElementById("msg");
  const sendBtn = document.getElementById("sendBtn");

  function addBubble(role, text) {
    const row = document.createElement("div");
    row.className = "row " + (role === "user" ? "user" : "bot");

    const bubble = document.createElement("div");
    bubble.className = "bubble " + (role === "user" ? "user" : "bot");
    bubble.textContent = text;

    row.appendChild(bubble);
    chatEl.appendChild(row);
    chatEl.scrollTop = chatEl.scrollHeight;

    return bubble;
  }

  function addTyping() {
    const row = document.createElement("div");
    row.className = "row bot";
    const bubble = document.createElement("div");
    bubble.className = "bubble bot";

    const t = document.createElement("span");
    t.className = "typing";
    t.innerHTML = "<i></i><i></i><i></i>";
    bubble.appendChild(t);

    row.appendChild(bubble);
    chatEl.appendChild(row);
    chatEl.scrollTop = chatEl.scrollHeight;

    return { row, bubble };
  }

  async function send() {
    const msg = msgEl.value.trim();
    if (!msg) return;

    msgEl.value = "";
    msgEl.focus();
    sendBtn.disabled = true;

    addBubble("user", msg);
    const typing = addTyping();

    const botBubble = typing.bubble;
    botBubble.textContent = "";

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ conversation_id: cid, user_message: msg })
      });

      if (!res.ok) {
        const errText = await res.text();
        botBubble.textContent = "Server error: " + errText;
        return;
      }

      const reader = res.body.getReader();
      const dec = new TextDecoder();

      let full = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        full += dec.decode(value);
        botBubble.textContent = full;
        chatEl.scrollTop = chatEl.scrollHeight;
      }
    } catch (e) {
      botBubble.textContent = "Network error: " + e;
    } finally {
      sendBtn.disabled = false;
    }
  }

  msgEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  addBubble("bot", "Hey! Ask me about weather, office hours, grading policy, or a live fact (like current president).");
</script>
</body>
</html>
"""

# API schema

class ChatIn(BaseModel):
    conversation_id: str
    user_message: str


# Tool call round
async def run_tool_round(
    messages: List[Dict[str, Any]],
    forced_tool: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """
    1) Ask model for tool_calls (forced or auto)
    2) Run tool calls server-side
    3) Return messages containing assistant(tool_calls) -> tool outputs
    """
    client = get_client()

    tool_choice: Any = "auto"
    if forced_tool:
        tool_choice = {"type": "function", "function": {"name": forced_tool}}

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SPECS,
        tool_choice=tool_choice,
        temperature=TEMP,
    )

    usage: Dict[str, Any] = {}
    if getattr(resp, "usage", None):
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }

    msg = resp.choices[0].message
    tools_used: List[str] = []

    assistant_message: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        assistant_message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            }
            for tc in msg.tool_calls
        ]
    messages.append(assistant_message)

    if not getattr(msg, "tool_calls", None):
        return messages, tools_used, usage

    for tc in msg.tool_calls:
        name = tc.function.name
        raw_args = tc.function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}

        fn = TOOL_FNS.get(name)
        if fn is None:
            result = {"error": f"unknown tool: {name}"}
        else:
            out = fn(**args)
            if hasattr(out, "__await__"):
                result = await out
            else:
                result = out

        tools_used.append(name)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    return messages, tools_used, usage


def stream_answer(messages: List[Dict[str, Any]]):
    """
    Stream final answer as text only (no new tools while streaming).
    """
    client = get_client()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SPECS,
        tool_choice="none",
        temperature=TEMP,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content

# Main endpoint
@app.post("/chat")
async def chat(inp: ChatIn, request: Request):
    ip = request.client.host if request.client else "unknown"
    rate_limit(ip)

    raw_text = inp.user_message or ""
    user_text = redact_secrets(raw_text)

    if is_self_harm(raw_text):
        msg = (
            "I'm really sorry you're feeling this way. You deserve support right now.\n\n"
            "If you're in the U.S., you can call or text 988 (Suicide & Crisis Lifeline). "
            "If you're outside the U.S., tell me your country and I can point you to a local number.\n\n"
            "If you're in immediate danger, please call your local emergency number right now."
        )
        return StreamingResponse(iter([msg]), media_type="text/plain; charset=utf-8")

    if is_unsafe(raw_text):
        msg = "I'm sorry, but I can't assist with that."
        return StreamingResponse(iter([msg]), media_type="text/plain; charset=utf-8")

    if SECRET_PATTERNS[0].search(raw_text):
        msg = "I can’t store API keys or secrets. Please remove it from the message (and rotate it if it was real)."
        return StreamingResponse(iter([msg]), media_type="text/plain; charset=utf-8")

# Enforce: if user asks to guess or says "without tools", do NOT guess.
    if re.search(r"\b(without tools|no tools|guess)\b", raw_text, re.I):
        msg = (
            "I can’t guess that without using tools or the knowledge base.\n"
            "If you want, ask normally (e.g., “What are our office hours?”) and I’ll look it up."
        )
        return StreamingResponse(iter([msg]), media_type="text/plain; charset=utf-8")

    append_msg(inp.conversation_id, "user", user_text)

    async def gen():
        start = time.time()
        tools_used_all: List[str] = []
        usage_all: Dict[str, Any] = {}

        messages = convo_messages(inp.conversation_id)
        forced = choose_forced_tool(user_text)

        try:
            messages, used, usage = await run_tool_round(messages, forced)
            tools_used_all.extend(used)
            if usage:
                usage_all = usage
        except Exception as ex:
            yield f"\n\n[error={type(ex).__name__}: {ex}]"
            return

        extra = ""
        lt = user_text.lower()
        if ("grading" in lt or "percent" in lt) and "average" in lt:
            extra = "If the user asked for the average of the three grading percentages, compute (60+30+10)/3 = 33.33% and include it."

        messages.append(
            {
                "role": "user",
                "content": "Please answer the last user message. If tool results are above, use them." + ((" " + extra) if extra else ""),
            }
        )

        final_parts: List[str] = []
        try:
            for piece in stream_answer(messages):
                final_parts.append(piece)
                yield piece
        except Exception as ex:
            yield f"\n\n[error={type(ex).__name__}: {ex}]"
            return

        final_text = "".join(final_parts).strip()
        if final_text:
            append_msg(inp.conversation_id, "assistant", final_text)

        latency_ms = int((time.time() - start) * 1000)
        log_jsonl(
            f"{LOG_DIR}/metrics.jsonl",
            {
                "ts": time.time(),
                "conversation_id": inp.conversation_id,
                "latency_ms": latency_ms,
                "tool_calls": tools_used_all,
                "usage": usage_all,
            },
        )
        yield f"\n\n[latency_ms={latency_ms}]"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
