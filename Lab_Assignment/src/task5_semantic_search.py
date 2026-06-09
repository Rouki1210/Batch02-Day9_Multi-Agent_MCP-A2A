"""
Task 5 - Semantic Search Module.

Dense retrieval over the local JSONL vector store created in Task 4.
"""

import json
import math

from src.task4_chunking_indexing import INDEX_FILE, hashing_embedding, run_pipeline


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity for two vectors."""
    if not a or not b:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def load_index() -> list[dict]:
    """Load indexed chunks from Task 4, creating the index if needed."""
    if not INDEX_FILE.exists():
        run_pipeline()

    chunks = []
    if not INDEX_FILE.exists():
        return chunks

    with INDEX_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks by vector similarity.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    chunks = load_index()
    if not chunks:
        return []

    query_embedding = hashing_embedding(query)
    results = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk.get("embedding", []))
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
