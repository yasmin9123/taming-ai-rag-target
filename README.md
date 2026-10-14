# Taming AI — PACTrans Keynote RAG Target

Machine-readable RAG target derived from **Taming AI: Start by Properly Introducing Yourself**, presented by Eva Lerner-Lam at the PACTrans Region 10 Student Conference in Anchorage, Alaska on October 10, 2026. Co-authors: Hannah Chun and Yasmin Korin.

## Purpose
This repository objectifies the presentation into a canonical parent object, structured slide-level child objects, retrieval-ready semantic chunks, and an evidence register.

## Structure
- `source/` — canonical PowerPoint
- `objects/presentation.json` — parent object
- `objects/slides.jsonl` — child objects
- `chunks/rag_chunks.jsonl` — embedding/retrieval inputs
- `evidence/evidence_index.json` — evidence register
- `manifest.json` — RAG-target manifest
- `scripts/` — optional ChromaDB vectorization/query scripts

## Source integrity
SHA-256: `82df5828b855873cf81f3df5442c6b6a6b36fa8bb155b59adfaf784c4d823b43`

## Objectification vs. vectorization
Objectification gives the presentation stable IDs, named fields, provenance, relationships, and evidence references. Vectorization converts each chunk's `embedding_text` into an embedding for semantic retrieval.

## Optional local vector store
```bash
pip install -r requirements.txt
python scripts/build_vector_store.py
python scripts/query.py "What does the presentation say about verified and validated claims?"
```

Version 1 uses one semantic retrieval chunk per slide. The canonical structured objects remain separate from the vector database so they stay inspectable and portable.

> This repository represents the supplied presentation. It does not independently validate external factual claims, QR-code destinations, or linked evidence.
