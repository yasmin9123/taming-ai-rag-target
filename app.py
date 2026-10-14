from pathlib import Path
import json
import math
import re
from collections import Counter

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Taming AI RAG Target API",
    description="Retrieval API for the Taming AI PACTrans 2026 keynote.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = ROOT / "chunks" / "rag_chunks.jsonl"


def load_chunks():
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


CHUNKS = load_chunks()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def similarity(query, document):
    """
    Lightweight cosine similarity using token frequencies.

    This keeps the public API easy to deploy without requiring a
    persistent ChromaDB instance. The canonical retrieval chunks remain
    the JSONL objects stored in this repository.
    """
    q = Counter(tokenize(query))
    d = Counter(tokenize(document))

    if not q or not d:
        return 0.0

    dot_product = sum(q[token] * d.get(token, 0) for token in q)

    q_length = math.sqrt(sum(value * value for value in q.values()))
    d_length = math.sqrt(sum(value * value for value in d.values()))

    if q_length == 0 or d_length == 0:
        return 0.0

    return dot_product / (q_length * d_length)


@app.get("/")
def root():
    return {
        "name": "Taming AI RAG Target API",
        "rag_target_id": "TAI-PAC-2026",
        "status": "online",
        "search_endpoint": "/search?q=your+question",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_loaded": len(CHUNKS),
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=2, description="Natural-language search query"),
    k: int = Query(5, ge=1, le=10, description="Number of results"),
):
    scored = []

    for chunk in CHUNKS:
        searchable_text = chunk.get(
            "embedding_text",
            f"{chunk.get('title', '')} {chunk.get('content', '')}",
        )

        score = similarity(q, searchable_text)

        scored.append(
            {
                "score": round(score, 6),
                "chunk_id": chunk.get("chunk_id"),
                "object_id": chunk.get("parent_object_id"),
                "slide": chunk.get("slide"),
                "title": chunk.get("title"),
                "object_type": chunk.get("object_type"),
                "topics": chunk.get("topics", []),
                "content": chunk.get("content"),
                "source": chunk.get("source"),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)

    return {
        "query": q,
        "rag_target_id": "TAI-PAC-2026",
        "results": scored[:k],
    }
