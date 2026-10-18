"""Ingestion : FAQ (CSV) + guide technique (PDF) + tickets (SQLite) -> ChromaDB.

Usage : python ingest.py
"""
import csv
import re
import sqlite3

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config


# ---------------------------------------------------------------- chargement
def load_faq() -> list[dict]:
    docs = []
    with open(config.FAQ_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs.append(
                {
                    "id": f"faq-{row['id']}",
                    "text": f"Question: {row['question']}\nRéponse: {row['answer']}",
                    "meta": {
                        "source": "faq",
                        "category": row["category"],
                        "ref": f"FAQ #{row['id']}",
                    },
                }
            )
    return docs


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def load_pdf() -> list[dict]:
    docs = []
    reader = PdfReader(str(config.PDF_PATH))
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # retire l'en-tête / pied de page répétitifs et normalise les espaces
        text = re.sub(r"Telecom\s+Technical\s+Reference\s+Guide\s*\S?\s*Internal\s+Use\s+Only", " ", text)
        text = re.sub(r"\bPage\s+\d+\s*$", " ", text.strip())
        text = re.sub(r"\s+", " ", text).strip()
        for i, chunk in enumerate(chunk_words(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP)):
            docs.append(
                {
                    "id": f"pdf-p{page_no}-{i}",
                    "text": chunk,
                    "meta": {
                        "source": "guide",
                        "category": "guide",
                        "ref": f"Guide technique, page {page_no}",
                    },
                }
            )
    return docs


def load_tickets() -> list[dict]:
    docs = []
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    for r in con.execute("SELECT * FROM tickets"):
        docs.append(
            {
                "id": f"ticket-{r['ticket_id']}",
                "text": (
                    f"Ticket {r['ticket_id']} ({r['category']}) - {r['issue_type']}\n"
                    f"Problème: {r['description']}\n"
                    f"Résolution: {r['resolution']}\n"
                    f"Statut: {r['status']}"
                ),
                "meta": {
                    "source": "ticket",
                    "category": r["category"],
                    "ref": f"Ticket {r['ticket_id']}",
                },
            }
        )
    con.close()
    return docs


# ------------------------------------------------------------------ indexage
def main() -> None:
    docs = load_faq() + load_pdf() + load_tickets()
    print(f"{len(docs)} documents chargés "
          f"(faq={sum(d['meta']['source']=='faq' for d in docs)}, "
          f"guide={sum(d['meta']['source']=='guide' for d in docs)}, "
          f"tickets={sum(d['meta']['source']=='ticket' for d in docs)})")

    print(f"Chargement du modèle d'embeddings : {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode(
        [d["text"] for d in docs], normalize_embeddings=True, show_progress_bar=True
    ).tolist()

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection(config.COLLECTION_NAME)  # réindexation propre
    except Exception:
        pass
    col = client.create_collection(config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        embeddings=embeddings,
        metadatas=[d["meta"] for d in docs],
    )
    print(f"Index créé : {col.count()} chunks dans '{config.CHROMA_DIR}'")


if __name__ == "__main__":
    main()
