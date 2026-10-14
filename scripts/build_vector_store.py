from pathlib import Path
import json, chromadb
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parents[1]
rows=[json.loads(x) for x in (ROOT/"chunks"/"rag_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
model=SentenceTransformer("all-MiniLM-L6-v2")
client=chromadb.PersistentClient(path=str(ROOT/"chroma_db"))
c=client.get_or_create_collection("taming_ai_keynote")
texts=[r["embedding_text"] for r in rows]
c.upsert(ids=[r["chunk_id"] for r in rows],documents=texts,embeddings=model.encode(texts).tolist(),metadatas=[{"slide":r["slide"],"title":r["title"],"object_type":r["object_type"]} for r in rows])
print(f"Indexed {len(rows)} chunks.")
