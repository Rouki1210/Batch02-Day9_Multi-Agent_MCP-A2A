"""
Task 7 - Reranking Module.

Implemented methods:
    - RRF: Reciprocal Rank Fusion for merging multiple ranked lists.
    - MMR-style rerank: relevance plus diversity using local hashing embeddings.

The main rerank() function defaults to a lightweight MMR-style reranker so it
runs locally without external APIs.
"""

from src.task4_chunking_indexing import hashing_embedding, tokenize
from src.task5_semantic_search import cosine_similarity


def lexical_overlap(query: str, content: str) -> float:
    """Return normalized token overlap between query and content."""
    query_tokens = set(tokenize(query))
    content_tokens = set(tokenize(content))
    if not query_tokens or not content_tokens:
        return 0.0
    return len(query_tokens & content_tokens) / len(query_tokens)


def normalize_scores(candidates: list[dict]) -> list[float]:
    """Normalize candidate scores into 0..1 for blending."""
    scores = [float(candidate.get("score", 0.0)) for candidate in candidates]
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 if max_score > 0 else 0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Cross-encoder-compatible fallback.

    README suggests Jina/Qwen cross-encoders, but those require API keys or heavy
    model downloads. This local fallback approximates reranking with query-token
    overlap and original retriever score, preserving the same public interface.
    """
    if not candidates:
        return []

    normalized = normalize_scores(candidates)
    reranked = []
    for candidate, original_score in zip(candidates, normalized):
        overlap = lexical_overlap(query, candidate.get("content", ""))
        final_score = 0.65 * overlap + 0.35 * original_score
        reranked.append(
            {
                **candidate,
                "score": float(final_score),
                "metadata": candidate.get("metadata", {}),
            }
        )

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance.

    MMR = lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected_docs))
    """
    if not candidates:
        return []

    remaining = []
    for candidate in candidates:
        embedding = candidate.get("embedding") or hashing_embedding(candidate.get("content", ""))
        relevance = cosine_similarity(query_embedding, embedding)
        remaining.append({**candidate, "embedding": embedding, "_relevance": relevance})

    selected = []
    while remaining and len(selected) < top_k:
        best_index = 0
        best_score = float("-inf")

        for index, candidate in enumerate(remaining):
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    cosine_similarity(candidate["embedding"], chosen["embedding"])
                    for chosen in selected
                )

            mmr_score = lambda_param * candidate["_relevance"] - (
                1 - lambda_param
            ) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index

        chosen = remaining.pop(best_index)
        chosen["_mmr_score"] = float(best_score)
        selected.append(chosen)

    results = []
    for chosen in selected:
        item = chosen.copy()
        item.pop("_relevance", None)
        item.pop("_mmr_score", None)
        item.pop("embedding", None)
        item["score"] = float(chosen["_mmr_score"])
        item["metadata"] = item.get("metadata", {})
        results.append(item)

    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    RRF(d) = sum(1 / (k + rank_r(d))) across rankers.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        item["metadata"] = item.get("metadata", {})
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "mmr",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        method: "cross_encoder" | "mmr"
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        return rerank_mmr(hashing_embedding(query), candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Dieu 248: Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Nghe si bi bat vi su dung ma tuy", "score": 0.7, "metadata": {}},
        {"content": "Python programming", "score": 0.6, "metadata": {}},
    ]
    for result in rerank("hinh phat tang tru ma tuy", dummy_candidates, top_k=2):
        print(f"[{result['score']:.3f}] {result['content']}")
