from pathlib import Path
import json

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Taming AI RAG Target API",
    description="Semantic retrieval API for the Taming AI PACTrans 2026 keynote.",
    version="2.0.0",
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

MODEL_NAME = "all-MiniLM-L6-v2"

MODEL = None
DOCUMENT_EMBEDDINGS = None


def load_chunks():
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


CHUNKS = load_chunks()

DOCUMENT_TEXTS = [
    chunk.get(
        "embedding_text",
        f"{chunk.get('title', '')} {chunk.get('content', '')}"
    )
    for chunk in CHUNKS
]


def ensure_model_loaded():
    global MODEL, DOCUMENT_EMBEDDINGS

    if MODEL is None:
        from sentence_transformers import SentenceTransformer

        MODEL = SentenceTransformer(MODEL_NAME)

        DOCUMENT_EMBEDDINGS = MODEL.encode(
            DOCUMENT_TEXTS,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )


@app.get("/")
def root():
    return {
        "name": "Taming AI RAG Target API",
        "rag_target_id": "TAI-PAC-2026",
        "status": "online",
        "retrieval": "semantic-vector",
        "embedding_model": MODEL_NAME,
        "chunks": len(CHUNKS),
        "search_endpoint": "/search?q=your+question",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "retrieval": "semantic-vector",
        "embedding_model": MODEL_NAME,
        "chunks_loaded": len(CHUNKS),
        "model_loaded": MODEL is not None,
    }


@app.get("/search")
def search(
    q: str = Query(
        ...,
        min_length=2,
        description="Natural-language semantic search query",
    ),
    k: int = Query(
        5,
        ge=1,
        le=10,
        description="Number of results",
    ),
):
    from sentence_transformers import util

    ensure_model_loaded()

    query_embedding = MODEL.encode(
        q,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    scores = util.cos_sim(
        query_embedding,
        DOCUMENT_EMBEDDINGS,
    )[0]

    top_k = min(k, len(CHUNKS))
    top_results = scores.topk(k=top_k)

    results = []

    for score, index in zip(
        top_results.values.tolist(),
        top_results.indices.tolist(),
    ):
        chunk = CHUNKS[index]

        results.append(
            {
                "score": round(float(score), 6),
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

    return {
        "query": q,
        "rag_target_id": "TAI-PAC-2026",
        "retrieval": "semantic-vector",
        "embedding_model": MODEL_NAME,
        "results": results,
    }
