"""Pipeline RAG : récupération (ChromaDB) + génération (Groq).

Usage CLI : python rag.py
"""
from functools import lru_cache

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

import config

SYSTEM_PROMPT = """Tu es un assistant du support client d'un opérateur télécom.
Réponds UNIQUEMENT à partir du CONTEXTE fourni (FAQ, tickets résolus, guide technique).
Règles :
- IMPORTANT : réponds TOUJOURS dans la langue exacte de la QUESTION (anglais -> anglais, arabe -> arabe, français -> français), même si le contexte est dans une autre langue.
- Si le contexte ne contient pas la réponse, dis clairement que tu ne sais pas
  et propose de contacter le support au 611. N'invente jamais de chiffres, prix ou procédures.
- Sois concis et donne des étapes numérotées quand il s'agit d'une procédure.
- Cite tes sources entre crochets à la fin, par exemple [FAQ #4], [Ticket TK-002]."""


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_collection(config.COLLECTION_NAME)


@lru_cache(maxsize=1)
def _llm() -> Groq:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY manquante : copie .env.example vers .env et remplis-la.")
    return Groq(api_key=config.GROQ_API_KEY)


def retrieve(query: str, k: int = config.TOP_K) -> list[dict]:
    emb = _embedder().encode([query], normalize_embeddings=True).tolist()
    res = _collection().query(query_embeddings=emb, n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "meta": meta, "score": round(1 - dist, 3)})
    return hits


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{h['meta']['ref']}]\n{h['text']}" for h in hits)


def answer(query: str, history: list[dict] | None = None, k: int = config.TOP_K) -> dict:
    """Retourne {'answer': str, 'sources': list[dict]}."""
    hits = retrieve(query, k)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += history[-6:]  # un peu de mémoire de conversation
    messages.append(
        {"role": "user", "content": f"CONTEXTE:\n{build_context(hits)}\n\nQUESTION: {query}"}
    )
    resp = _llm().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1500,  # large : les modèles "reasoning" consomment des tokens de réflexion
        extra_body={"reasoning_effort": "low"},  # réponses plus rapides (gpt-oss)
    )
    return {"answer": resp.choices[0].message.content, "sources": hits}


if __name__ == "__main__":
    print("Assistant télécom (Ctrl+C pour quitter)\n")
    history: list[dict] = []
    while True:
        try:
            q = input("Vous > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        out = answer(q, history)
        print(f"\nAssistant > {out['answer']}\n")
        print("Sources :", ", ".join(f"{h['meta']['ref']} ({h['score']})" for h in out["sources"]), "\n")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": out["answer"]}]
