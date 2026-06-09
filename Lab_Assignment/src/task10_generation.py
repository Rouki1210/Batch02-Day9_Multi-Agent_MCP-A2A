"""
Task 10 - Generation with Gemini API and citations.

Run:
    python src/task10_generation.py

Required for Gemini generation:
    GEMINI_API_KEY=...

Optional:
    GEMINI_MODEL=gemini-2.0-flash
"""

import os
import re

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

from src.task4_chunking_indexing import tokenize
from src.task9_retrieval_pipeline import retrieve

load_dotenv()

# top_k=5 keeps enough evidence for a useful answer without making the context
# too noisy. top_p=0.9 gives Gemini natural wording while temperature=0.3 keeps
# the answer factual and conservative for legal/news RAG.
TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

SYSTEM_PROMPT = """You are a Vietnamese RAG legal/news assistant.

Answer in Vietnamese using ONLY the provided context.
For every factual claim, include an immediate citation in square brackets using
the source label shown in the context, for example [nghi-dinh-105-2021.md, legal].
If the context does not explicitly support the answer, say:
"Tôi không thể xác minh thông tin này từ nguồn hiện có."

Rules:
- Do not invent facts, article numbers, penalties, dates, names, or sources.
- Do not cite sources that are not present in the provided context.
- Prefer a complete but concise answer.
- If multiple sources support the answer, mention the most relevant ones.
"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to reduce the "lost in the middle" effect.

    Input by score: [1, 2, 3, 4, 5]
    Output:         [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks

    front = [chunks[index] for index in range(0, len(chunks), 2)]
    back = [chunks[index] for index in range(len(chunks) - 1, 0, -2)]
    return front + back


def source_label(chunk: dict, fallback_index: int) -> str:
    """Build a compact citation label from chunk metadata."""
    metadata = chunk.get("metadata", {})
    source = metadata.get("source") or f"Source {fallback_index}"
    doc_type = metadata.get("type", "unknown")
    return f"{source}, {doc_type}"


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks as citation-ready context.
    """
    context_parts = []
    for index, chunk in enumerate(chunks, 1):
        label = source_label(chunk, index)
        score = float(chunk.get("score", 0.0))
        context_parts.append(
            f"[Document {index}]\n"
            f"Source label: [{label}]\n"
            f"Retrieval score: {score:.3f}\n"
            f"Content:\n{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(context_parts)


def build_user_prompt(query: str, context: str) -> str:
    """Build the Gemini user prompt."""
    return f"""Context:
{context}

---

Question:
{query}

Answer with citations. If the context is insufficient, state that you cannot
verify the information from the available sources."""


def call_gemini(prompt: str) -> str:
    """
    Call Gemini using the official google-genai SDK.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env.")

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError("Install Gemini SDK: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        ),
    )

    text = getattr(response, "text", "") or ""
    if not text.strip():
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units for local fallback."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def sentence_relevance(query: str, sentence: str) -> float:
    """Score a sentence by query-token overlap for local fallback."""
    query_tokens = set(tokenize(query))
    sentence_tokens = set(tokenize(sentence))
    if not query_tokens or not sentence_tokens:
        return 0.0
    return len(query_tokens & sentence_tokens) / len(query_tokens)


def extractive_fallback_answer(query: str, chunks: list[dict], max_sentences: int = 4) -> str:
    """
    Local fallback answer when Gemini is unavailable.
    """
    candidates = []
    for index, chunk in enumerate(chunks, 1):
        label = source_label(chunk, index)
        for sentence in split_sentences(chunk.get("content", "")):
            score = sentence_relevance(query, sentence)
            if score > 0:
                candidates.append((score, sentence, label))

    if not candidates:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    candidates.sort(key=lambda item: item[0], reverse=True)
    used = set()
    cited_sentences = []
    for _, sentence, label in candidates:
        normalized = sentence.lower()
        if normalized in used:
            continue
        used.add(normalized)
        cited_sentences.append(f"{sentence} [{label}]")
        if len(cited_sentences) >= max_sentences:
            break

    return " ".join(cited_sentences)


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    use_gemini: bool = True,
) -> dict:
    """
    End-to-end RAG generation with Gemini citation.

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str,
            'context': str,
            'generation_model': str
        }
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    prompt = build_user_prompt(query, context)

    generation_model = GEMINI_MODEL
    error = None
    if use_gemini:
        try:
            answer = call_gemini(prompt)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            answer = extractive_fallback_answer(query, reordered)
            generation_model = "extractive-fallback"
    else:
        answer = extractive_fallback_answer(query, reordered)
        generation_model = "extractive-fallback"

    result = {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
        "context": context,
        "prompt": prompt,
        "generation_model": generation_model,
        "generation_config": {
            "top_k": top_k,
            "top_p": TOP_P,
            "temperature": TEMPERATURE,
        },
    }
    if error:
        result["generation_error"] = error
    return result


if __name__ == "__main__":
    result = generate_with_citation("Những nghệ sĩ nào liên quan tới ma túy?")
    print(result["answer"])
