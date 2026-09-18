import chromadb

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "week4_rag"


def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)


def add_chunks(collection, source: str, chunks: list[str], vectors: list[list[float]]) -> None:
    ids = [f"{source}::{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)


def delete_by_source(collection, source: str) -> None:
    collection.delete(where={"source": source})


def query(collection, query_embedding: list[float], top_k: int = 4) -> list[dict]:
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, count))
    matches = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        matches.append({"text": doc, "source": metadata["source"], "chunk_index": metadata["chunk_index"]})
    return matches
