import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

API_URL_DEFAULT = os.getenv("EVAL_API_URL", "http://127.0.0.1:8000/chat")
PROMPTS_FILE_DEFAULT = os.getenv("EVAL_PROMPTS_FILE", "eval_prompts.json")

LAT_RE = re.compile(r"\[latency_ms=(\d+)\]")


def read_streaming_text(resp: httpx.Response) -> str:
    # endpoint returns text/plain streamed chunks
    out_parts: List[str] = []
    for chunk in resp.iter_text():
        if chunk:
            out_parts.append(chunk)
    return "".join(out_parts)


def extract_server_latency_ms(text: str) -> Optional[int]:
    m = LAT_RE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=API_URL_DEFAULT)
    ap.add_argument("--prompts-file", default=PROMPTS_FILE_DEFAULT)
    ap.add_argument(
        "--eval-mode",
        default=None,
        choices=[None, "on", "off"],
        help="Optional: sends X-Eval-Mode header (useful if server reads it).",
    )
    args = ap.parse_args()

    api_url = args.api_url
    prompts_file = args.prompts_file

    with open(prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cid = data["conversation_id"]
    prompts = data["prompts"]

    os.makedirs("results/transcripts", exist_ok=True)
    os.makedirs("results/eval_runs", exist_ok=True)

    transcript_path = f"results/transcripts/{cid}.txt"
    results_path = f"results/eval_runs/{cid}.json"

    results: List[Dict[str, Any]] = []

    headers: Dict[str, str] = {}
    if args.eval_mode == "on":
        headers["X-Eval-Mode"] = "1"
    elif args.eval_mode == "off":
        headers["X-Eval-Mode"] = "0"

    with open(transcript_path, "w", encoding="utf-8") as tlog:
        tlog.write(f"=== Evaluation run: {cid} ===\n")
        tlog.write(f"API: {api_url}\n")
        if headers:
            tlog.write(f"Headers: {headers}\n")
        tlog.write("\n")

        with httpx.Client(timeout=None) as client:
            for idx, p in enumerate(prompts, start=1):
                pid = p.get("id", idx)
                text = p.get("text", "")

                print(f"Running prompt {idx}/{len(prompts)} (id={pid})...")
                print(f"  Sending: {text!r}")

                payload = {"conversation_id": cid, "user_message": text}

                tlog.write(f"\n\nPROMPT {pid}: {text}\n")
                tlog.write("BOT:\n")

                start = time.time()

                try:
                    resp = client.post(api_url, json=payload, headers=headers)
                    elapsed_ms = int((time.time() - start) * 1000)

                    if resp.status_code != 200:
                        err_text = (resp.text or "").strip()
                        tlog.write(f"[ERROR status={resp.status_code}] {err_text}\n")
                        tlog.write(f"\n[client_elapsed_ms={elapsed_ms}]\n")

                        results.append(
                            {
                                "id": pid,
                                "prompt": text,
                                "ok": False,
                                "status": resp.status_code,
                                "error": err_text,
                                "output": "",
                                "client_elapsed_ms": elapsed_ms,
                                "server_latency_ms": None,
                            }
                        )
                        continue

                    bot_text = read_streaming_text(resp).strip()
                    server_latency_ms = extract_server_latency_ms(bot_text)

                    tlog.write(bot_text + "\n")
                    tlog.write(f"\n[client_elapsed_ms={elapsed_ms}]\n")

                    results.append(
                        {
                            "id": pid,
                            "prompt": text,
                            "ok": True,
                            "status": 200,
                            "error": "",
                            "output": bot_text,
                            "client_elapsed_ms": elapsed_ms,
                            "server_latency_ms": server_latency_ms,
                        }
                    )

                except Exception as ex:
                    elapsed_ms = int((time.time() - start) * 1000)
                    tlog.write(f"[EXCEPTION] {type(ex).__name__}: {ex}\n")
                    tlog.write(f"\n[client_elapsed_ms={elapsed_ms}]\n")

                    results.append(
                        {
                            "id": pid,
                            "prompt": text,
                            "ok": False,
                            "status": None,
                            "error": f"{type(ex).__name__}: {ex}",
                            "output": "",
                            "client_elapsed_ms": elapsed_ms,
                            "server_latency_ms": None,
                        }
                    )

    summary = {
        "conversation_id": cid,
        "api_url": api_url,
        "prompts_file": prompts_file,
        "run_ts": time.time(),
        "total_prompts": len(prompts),
        "successes": sum(1 for r in results if r["ok"]),
        "failures": sum(1 for r in results if not r["ok"]),
        "ts_percent": round(100.0 * sum(1 for r in results if r["ok"]) / max(1, len(prompts)), 2),
        "results": results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Transcript saved to: {transcript_path}")
    print(f"Structured results saved to: {results_path}")
    print("Server-side metrics are still saved in results/metrics.jsonl")


if __name__ == "__main__":
    main()
