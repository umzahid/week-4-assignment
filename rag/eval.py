import re

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "for", "and", "or", "it", "this", "that", "with", "as", "by", "at",
    "be", "has", "have", "had", "its", "from", "not", "you", "your",
}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def compute_metrics(question: str, answer: str, sources: list[dict], embed_fn) -> dict:
    if not sources:
        return {"retrieval_relevance": 0.0, "groundedness": 0.0, "answer_relevance": 0.0}

    question_embedding = embed_fn([question])[0]
    answer_embedding = embed_fn([answer])[0]
    source_embeddings = embed_fn([source["text"] for source in sources])

    retrieval_relevance = sum(
        _cosine_similarity(question_embedding, source_embedding)
        for source_embedding in source_embeddings
    ) / len(source_embeddings)

    context_words: set[str] = set()
    for source in sources:
        context_words |= _content_words(source["text"])
    answer_words = _content_words(answer)
    groundedness = len(answer_words & context_words) / len(answer_words) if answer_words else 0.0

    answer_relevance = _cosine_similarity(question_embedding, answer_embedding)

    return {
        "retrieval_relevance": round(retrieval_relevance, 3),
        "groundedness": round(groundedness, 3),
        "answer_relevance": round(answer_relevance, 3),
    }
