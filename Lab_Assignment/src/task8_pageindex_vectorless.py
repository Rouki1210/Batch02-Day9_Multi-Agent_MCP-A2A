"""
Task 8 - PageIndex Vectorless RAG fallback.

The real PageIndex SDK requires an account/API key. For this local assignment
pipeline, pageindex_search() provides a vectorless fallback over Markdown files
using structural metadata and token overlap. Results are marked source=pageindex
so Task 9 can exercise fallback logic.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

from src.task4_chunking_indexing import STANDARDIZED_DIR, tokenize

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")


def upload_documents() -> list[dict]:
    """
    Collect local Markdown documents that would be uploaded to PageIndex.

    Returns metadata for demonstration without requiring a PageIndex account.
    """
    uploaded = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        uploaded.append(
            {
                "filename": md_file.name,
                "type": md_file.parent.name,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)).replace("\\", "/"),
            }
        )
    return uploaded


def document_score(query: str, content: str) -> float:
    """Vectorless lexical/structural score for PageIndex fallback."""
    query_tokens = set(tokenize(query))
    content_tokens = tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0

    content_set = set(content_tokens)
    overlap = len(query_tokens & content_set) / len(query_tokens)
    density = sum(1 for token in content_tokens if token in query_tokens) / len(content_tokens)
    return overlap + density


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval fallback.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    results = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        score = document_score(query, content)
        metadata = {
            "source": md_file.name,
            "type": md_file.parent.name,
            "path": str(md_file.relative_to(STANDARDIZED_DIR)).replace("\\", "/"),
            "retriever": "local-pageindex-fallback",
        }
        results.append(
            {
                "content": content[:1500],
                "score": float(score),
                "metadata": metadata,
                "source": "pageindex",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in pageindex_search("hinh phat su dung ma tuy", top_k=3):
        print(f"[{result['score']:.3f}] {result['metadata']['source']}")
