from pathlib import Path
import sys, chromadb
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parents[1]
q=" ".join(sys.argv[1:]) or "How does the presentation describe a RAG target?"
model=SentenceTransformer("all-MiniLM-L6-v2")
c=chromadb.PersistentClient(path=str(ROOT/"chroma_db")).get_collection("taming_ai_keynote")
r=c.query(query_embeddings=[model.encode(q).tolist()],n_results=5)
for doc,meta in zip(r["documents"][0],r["metadatas"][0]):
    print(f"\nSlide {meta['slide']} — {meta['title']}\n{doc}")
