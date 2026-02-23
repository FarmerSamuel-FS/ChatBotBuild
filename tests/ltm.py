# ltm.py
import os
import time
from typing import Dict, List

import chromadb
from chromadb.config import Settings

DEFAULT_DIR = os.getenv("LTM_DIR", "results/ltm_db")
COLLECTION_NAME = os.getenv("LTM_COLLECTION", "chatbot_ltm")

# lazy singleton
_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    os.makedirs(DEFAULT_DIR, exist_ok=True)

    _client = chromadb.PersistentClient(
        path=DEFAULT_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def add_fact(conversation_id: str, fact: str) -> None:
    """
    Store a fact for a conversation_id.
    """
    fact = (fact or "").strip()
    if not fact:
        return

    col = _get_collection()
    doc_id = f"{conversation_id}-{int(time.time() * 1000)}"
    col.add(
        ids=[doc_id],
        documents=[fact],
        metadatas=[{"conversation_id": conversation_id, "ts": time.time()}],
    )


def search_facts(conversation_id: str, query: str, k: int = 3) -> List[str]:
    """
    Retrieve up to k relevant facts for a conversation_id.
    """
    query = (query or "").strip()
    if not query:
        return []

    col = _get_collection()
    res = col.query(
        query_texts=[query],
        n_results=max(1, int(k)),
        where={"conversation_id": conversation_id},
    )

    docs = (res.get("documents") or [[]])[0]
    # basic cleanup + de-dupe preserving order
    out: List[str] = []
    seen = set()
    for d in docs:
        d = (d or "").strip()
        if d and d not in seen:
            out.append(d)
            seen.add(d)
    return out
