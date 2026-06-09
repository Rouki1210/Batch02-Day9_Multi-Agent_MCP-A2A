"""
Task 9 - Complete Retrieval Pipeline.

Pipeline:
    semantic_search + lexical_search -> RRF merge -> rerank -> PageIndex fallback.
"""

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank, rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "mmr"


def normalize_hybrid_score(score: float) -> float:
    """Map small RRF/MMR scores into a simple 0..1-ish confidence range."""
    if score <= 0:
        return 0.0
    return min(1.0, score)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieve relevant chunks with fallback logic.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': str}
    """
    dense_results = semantic_search(query, top_k=top_k * 3)
    sparse_results = lexical_search(query, top_k=top_k * 3)

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 3)
    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    best_score = normalize_hybrid_score(final_results[0]["score"]) if final_results else 0.0
    if not final_results or best_score < score_threshold:
        return pageindex_search(query, top_k=top_k)

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy",
        "Nghe si nao bi bat vi su dung ma tuy nam 2024",
        "Luat phong chong ma tuy quy dinh gi ve cai nghien",
    ]

    for question in test_queries:
        print(f"\nQuery: {question}")
        for index, result in enumerate(retrieve(question, top_k=3), 1):
            print(
                f"  {index}. [{result['score']:.3f}] "
                f"[{result['source']}] {result['content'][:80]}..."
            )
