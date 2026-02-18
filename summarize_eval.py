# summarize_eval.py


import argparse
import glob
import json
import re
import statistics
from pathlib import Path

LAT_RE = re.compile(r"\[latency_ms=(\d+)\]")
REFUSAL_GUESS_RE = re.compile(r"\b(can[’']?t|cannot)\s+guess\b", re.I)
GUESSY_PROMPT_RE = re.compile(r"\b(without tools|no tools|guess)\b", re.I)

def extract_latency_ms(row: dict) -> int | None:
    v = row.get("client_elapsed_ms")
    if isinstance(v, int):
        return v
    out = row.get("output") or ""
    m = LAT_RE.search(out)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None
def pct(x: float) -> str:
    return f"{x:.1f}%"
def quantile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return float(values[idx])
def infer_eval_mode(data: dict) -> str:
    """
    Best-effort inference:
    - If the JSON has eval_mode/eval/EVAL_MODE fields, use them.
    - Otherwise infer from prompt #20 style behavior:
        * if there exists a prompt containing "without tools/no tools/guess"
          and output contains "can't guess" -> assume eval_mode OFF (enforced)
        * if it answers normally -> assume eval_mode ON (or enforcement disabled)
    """
    for k in ("eval_mode", "EVAL_MODE", "eval", "mode"):
        if k in data:
            v = data.get(k)
            if isinstance(v, bool):
                return "on" if v else "off"
            if isinstance(v, (int, float)):
                return "on" if int(v) != 0 else "off"
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("1", "true", "yes", "on"):
                    return "on"
                if s in ("0", "false", "no", "off"):
                    return "off"
    rows = data.get("results", []) or []
    for r in rows:
        prompt = (r.get("prompt") or "")
        if GUESSY_PROMPT_RE.search(prompt):
            out = (r.get("output") or "")
            if REFUSAL_GUESS_RE.search(out):
                return "off (inferred)"
            return "on (inferred)"
    return "unknown"
def analyze_policy_checks(data: dict) -> list[str]:
    """
    Lightweight rubric checks based on your eval prompts:
    - If a prompt asks to guess/without tools -> should refuse (when enforcement expected).
    - If connection failed -> call out.
    """
    notes: list[str] = []
    rows = data.get("results", []) or []

    # Connection refused 
    conn_fail = [r for r in rows if (r.get("ok") is False) and "connection refused" in (r.get("error") or "").lower()]
    if conn_fail:
        notes.append(f"Network issue: {len(conn_fail)} requests failed with connection refused (server likely not running).")

    # Guess refusal behavior
    guess_rows = [r for r in rows if GUESSY_PROMPT_RE.search(r.get("prompt") or "")]
    if guess_rows:
        refused = [r for r in guess_rows if REFUSAL_GUESS_RE.search(r.get("output") or "")]
        if len(refused) == len(guess_rows):
            notes.append("Guess/without-tools prompts: refusal behavior ✅")
        elif len(refused) == 0:
            notes.append("Guess/without-tools prompts: refusal behavior ❌ (it answered instead of refusing)")
        else:
            notes.append(f"Guess/without-tools prompts: partial refusal ({len(refused)}/{len(guess_rows)}).")

    return notes

def summarize(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("results", [])
    ok = [r for r in rows if r.get("ok") is True]
    fail = [r for r in rows if not r.get("ok")]

    latencies = [extract_latency_ms(r) for r in ok]
    latencies = [x for x in latencies if isinstance(x, int)]

    avg = round(statistics.mean(latencies), 2) if latencies else None
    p50 = round(quantile(latencies, 0.50), 2) if latencies else None
    p95 = round(quantile(latencies, 0.95), 2) if latencies else None
    mx = max(latencies) if latencies else None

    total = data.get("total_prompts", len(rows))
    success_rate = (len(ok) / total * 100.0) if total else 0.0

    eval_mode = infer_eval_mode(data)

    print("=" * 72)
    print("File:", path)
    print("Conversation:", data.get("conversation_id"))
    print("Eval mode:", eval_mode)
    print("API URL:", data.get("api_url", ""))
    print("Total prompts:", total)
    print("Successes:", len(ok))
    print("Failures:", len(fail))
    print("Success rate:", pct(success_rate))
    if "ts_percent" in data:
        print("TS%:", data.get("ts_percent"))

    print("Latency (ms): avg =", avg, "| p50 =", p50, "| p95 =", p95, "| max =", mx)

    notes = analyze_policy_checks(data)
    if notes:
        print("\nNotes:")
        for n in notes:
            print("-", n)

    if fail:
        print("\nFailed prompt IDs:")
        for r in fail:
            rid = r.get("id")
            st = r.get("status")
            err = (r.get("error") or "").strip()
            if len(err) > 160:
                err = err[:160] + "…"
            print(f"- {rid} | status: {st} | error: {err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="+",
        help="One or more eval JSON files (supports glob patterns). Example: results/eval_runs/*.json",
    )
    args = ap.parse_args()

    files: list[str] = []
    for p in args.paths:
        matches = glob.glob(p)
        if matches:
            files.extend(matches)
        else:
            files.append(p)

    files = [str(Path(f)) for f in files]
    for f in sorted(set(files)):
        summarize(f)

if __name__ == "__main__":
    main()
