# Week 4 RAG Assignment

A fully local RAG application: point it at a folder of `.pdf`/`.docx`
files, and it automatically keeps its vector index in sync as files are
added, removed, or edited — no manual re-upload step. Every answer is
grounded only in the watched folder's content, and comes with a live
3-metric "confidence report card."

## How it works

1. Point the sidebar at a folder (defaults to `sample_docs/`).
2. Every ~5 seconds, the app hashes each file in that folder and diffs
   it against a manifest of what's already indexed — new files get
   chunked/embedded/added, removed files get their vectors deleted,
   changed files get their old vectors replaced with new ones.
3. Ask a question — the app retrieves the most relevant chunks and asks
   a local Ollama model to answer using only that context.
4. Every answer shows 3 self-computed metrics (no eval framework, no
   extra LLM calls): retrieval relevance, groundedness, and answer
   relevance — plus an expandable list of the exact source chunks used.

## Tools & tech stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Runtime |
| Streamlit | Web UI, including `st.fragment` for periodic folder polling |
| ChromaDB | Local persistent vector database |
| pypdf | `.pdf` text extraction |
| python-docx | `.docx` text extraction |
| langchain-text-splitters | Chunking with overlap |
| sentence-transformers (`all-MiniLM-L6-v2`) | Local embeddings — no API key |
| Ollama (`llama3.2:3b`) | Local, open-source LLM — no API key |

No API keys are required anywhere in this stack.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install Ollama if you don't have it: https://ollama.com
ollama serve &      # if not already running
ollama pull llama3.2:3b

streamlit run app.py
```

## Document lifecycle handling

A `manifest.json` (gitignored, local runtime state) maps each tracked
filename to a SHA-256 hash of its contents. On every periodic check:
added files are new entries, deleted files are missing entries, and
modified files are entries whose hash changed — each case maps to an
add/delete/update against the ChromaDB collection.

## A note on the LLM choice

The assignment specifies "an open-source LLM of your choice." This
project deliberately runs one fully locally via Ollama rather than a
hosted API (Groq, OpenAI, etc.), so no external API key or network
dependency is required at all. `llama3.2:3b` was chosen after live
testing showed the smaller `llama3.2:1b` genuinely unreliable at basic
fact extraction — it would restate a fact present in the retrieved
context and then still answer "I don't know" in the same response.
`3b` (~2GB) fixed this while remaining far lighter than a "heavy" model.

## Evaluation metrics — how they're computed

No third-party eval framework is used — the assignment grants freedom
to "select the evaluation framework and metrics... yourself," so all 3
metrics are computed directly from data already available at answer
time, using the same local embedding model as retrieval:

- **Retrieval relevance** — average cosine similarity between the
  question and each retrieved chunk.
- **Groundedness** — fraction of the answer's distinct content words
  that also appear in the retrieved context.
- **Answer relevance** — cosine similarity between the answer and the
  original question.

Retrieval relevance is the most reliable signal for distinguishing a
grounded answer from a refusal (it dropped from ~0.47 on an answerable
question to ~0.03-0.10 on unrelated ones in testing) — groundedness can
still score moderately on a well-explained refusal, since explaining
*why* the context doesn't cover something naturally reuses some of the
context's own vocabulary.
