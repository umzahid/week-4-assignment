from rag import embeddings, store
from rag.llm import generate

TOP_K = 4

SYSTEM_PROMPT = (
    "You answer questions about the user's documents using only the "
    "context provided below, whether the user asks a direct question or "
    "makes a request like 'tell me about this' or 'summarize this'. "
    "If the context does not contain the information needed, say you "
    "don't know. Do not use outside knowledge."
)


def ask(collection, question: str) -> dict:
    query_embedding = embeddings.embed_texts([question])[0]
    matches = store.query(collection, query_embedding, top_k=TOP_K)
    if not matches:
        return {"answer": "I don't know.", "sources": []}
    context = "\n\n".join(f"[{i + 1}] {m['text']}" for i, m in enumerate(matches))
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    answer = generate(prompt, system=SYSTEM_PROMPT)
    return {"answer": answer, "sources": matches}
