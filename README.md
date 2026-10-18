# RAG Support Télécom

Assistant de support client basé sur le RAG (Retrieval-Augmented Generation).
Il répond aux questions en s'appuyant sur trois sources :

| Source | Fichier | Contenu |
|---|---|---|
| FAQ | `faq.csv` | 25 questions/réponses (data, voix, roaming, SIM, facturation…) |
| Tickets | `tickets.db` (SQLite) | 20 tickets résolus avec leur résolution |
| Guide technique | `telecom_guide.pdf` | Référence 2G/3G/4G/5G, dépannage (9 pages) |

## Architecture

```
faq.csv ─┐
tickets.db ─┼─► ingest.py ─► embeddings (MiniLM) ─► ChromaDB
guide.pdf ─┘                                           │
                       question ─► embedding ─► top-k ─┘
                                                │
                                   prompt + contexte ─► Groq (GPT-OSS 120B) ─► réponse + sources
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # puis remplir GROQ_API_KEY et HF_TOKEN
```

- `GROQ_API_KEY` : clé gratuite sur https://console.groq.com
- `HF_TOKEN` : token Hugging Face (téléchargement du modèle d'embeddings)

## Utilisation

```bash
python ingest.py            # 1. indexer les données (une seule fois)
python rag.py               # 2a. chat en terminal
streamlit run app.py        # 2b. interface web
```

## Configuration

Variables optionnelles dans `.env` : `LLM_MODEL` (défaut `openai/gpt-oss-120b`),
`EMBEDDING_MODEL` (défaut `sentence-transformers/all-MiniLM-L6-v2`).
Les paramètres de découpage et le `TOP_K` sont dans `config.py`.

## Exemples de questions

- How do I activate international roaming?
- Mon internet est très lent, que faire ?
- Que faire si le solde de data ne se met pas à jour ?
- Quelle est la différence entre la 4G et la 5G ?
