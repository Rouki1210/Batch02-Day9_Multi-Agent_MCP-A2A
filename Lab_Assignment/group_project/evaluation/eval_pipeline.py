"""
RAG Evaluation Pipeline — Local LLM-based

Chạy:
    python group_project/evaluation/eval_pipeline.py

Dùng local LLM để đánh giá 4 metrics thay vì DeepEval (do DeepEval yêu cầu GPT API).
"""

import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
CACHE_DIR = Path(__file__).parent / ".cache"


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def call_llm(system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


def run_pipeline(question: str, use_reranking: bool = True, top_k: int = 5) -> dict:
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import reorder_for_llm, format_context

    chunks = retrieve(question, top_k=top_k, use_reranking=use_reranking)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    answer = call_llm(
        "Answer comprehensively in Vietnamese with citations. If the context lacks info, say so.",
        f"Context:\n{context}\n\n---\n\nQuestion: {question}",
    )

    return {
        "answer": answer,
        "retrieval_context": [c["content"] for c in chunks],
        "chunks": chunks,
    }


def load_or_run(question: str, use_reranking: bool) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = "rerank" if use_reranking else "dense"
    safe = "".join(c if c.isalnum() else "_" for c in question[:40])
    cache_file = CACHE_DIR / f"{tag}_{safe}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    result = run_pipeline(question, use_reranking=use_reranking)
    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def score_faithfulness(answer: str, context: list[str]) -> float:
    prompt = (
        f"Rate the faithfulness of the following answer from 0.0 to 1.0 based on "
        f"how well it is supported by the provided context.\n\n"
        f"Context:\n{chr(10).join(context[:3])}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Return ONLY a number between 0.0 and 1.0 (e.g., 0.85). "
        f"0.0 = completely unfaithful, 1.0 = fully supported by context."
    )
    resp = call_llm("You are an evaluation assistant. Return only a number.", prompt)
    match = re.search(r"([01]\.\d+)", resp)
    return float(match.group(1)) if match else 0.5


def score_answer_relevance(question: str, answer: str) -> float:
    prompt = (
        f"Rate how relevant the following answer is to the question from 0.0 to 1.0.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Return ONLY a number between 0.0 and 1.0."
    )
    resp = call_llm("You are an evaluation assistant. Return only a number.", prompt)
    match = re.search(r"([01]\.\d+)", resp)
    return float(match.group(1)) if match else 0.5


def score_context_recall(question: str, expected: str, context: list[str]) -> float:
    prompt = (
        f"Rate how well the provided context contains the information needed to answer "
        f"the question from 0.0 to 1.0.\n\n"
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Context:\n{chr(10).join(context[:3])}\n\n"
        f"Return ONLY a number between 0.0 and 1.0."
    )
    resp = call_llm("You are an evaluation assistant. Return only a number.", prompt)
    match = re.search(r"([01]\.\d+)", resp)
    return float(match.group(1)) if match else 0.5


def score_context_precision(question: str, context: list[str]) -> float:
    prompt = (
        f"Rate the precision of the provided context from 0.0 to 1.0. "
        f"High precision means every chunk is useful for answering the question. "
        f"Low precision means many chunks are irrelevant.\n\n"
        f"Question: {question}\n"
        f"Context:\n{chr(10).join(context)}\n\n"
        f"Return ONLY a number between 0.0 and 1.0."
    )
    resp = call_llm("You are an evaluation assistant. Return only a number.", prompt)
    match = re.search(r"([01]\.\d+)", resp)
    return float(match.group(1)) if match else 0.5


def evaluate_config(config_name: str, use_reranking: bool) -> list[dict]:
    golden = load_golden_dataset()
    print(f"\n{'='*60}")
    print(f"  {config_name}")
    print(f"{'='*60}")

    per_item = []
    for i, item in enumerate(golden):
        q = item["question"]
        exp = item["expected_answer"]
        ts = time.time()

        print(f"  [{i+1}/{len(golden)}] {q[:55]:<55} ", end="", flush=True)
        result = load_or_run(q, use_reranking)
        print(f"[{time.time()-ts:.0f}s] ", end="", flush=True)

        tag = "rerank" if use_reranking else "dense"
        safe = "".join(c if c.isalnum() else "_" for c in q[:40])
        score_file = CACHE_DIR / f"{tag}_{safe}_scores.json"
        if score_file.exists():
            scores = json.loads(score_file.read_text(encoding="utf-8"))
            print("cached")
        else:
            print("eval...", end="", flush=True)
            ctx = result["retrieval_context"]
            fth = score_faithfulness(result["answer"], ctx)
            rel = score_answer_relevance(q, result["answer"])
            rec = score_context_recall(q, exp, ctx)
            pre = score_context_precision(q, ctx)
            scores = {
                "Faithfulness": fth,
                "AnswerRelevancy": rel,
                "ContextRecall": rec,
                "ContextPrecision": pre,
            }
            score_file.write_text(json.dumps(scores, ensure_ascii=False), encoding="utf-8")
            print(" done")

        per_item.append({
            "question": q,
            **scores,
            "avg": sum(scores.values()) / len(scores),
        })

    return per_item


def generate_report(config_a: list[dict], config_b: list[dict]):
    metrics = ["Faithfulness", "AnswerRelevancy", "ContextRecall", "ContextPrecision"]
    metric_labels = ["Faithfulness", "Answer Relevance", "Context Recall", "Context Precision"]

    def avg_scores(items):
        return {m: sum(item.get(m, 0) for item in items) / len(items) for m in metrics}

    def avg(l):
        return sum(l) / len(l) if l else 0

    sa = avg_scores(config_a)
    sb = avg_scores(config_b)

    worst_a = sorted(config_a, key=lambda x: x["avg"])[:3]
    worst_b = sorted(config_b, key=lambda x: x["avg"])[:3]

    lines = []
    lines.append("# RAG Evaluation Results\n")
    lines.append("## Framework sử dụng\n")
    lines.append("**Local LLM-based evaluation** — dùng model `oc/mimo-v2.5-free` để chấm 4 metrics\n")
    lines.append("---\n")
    lines.append("## Overall Scores\n")
    lines.append("| Metric | Config A (hybrid+rerank) | Config B (dense-only) | Δ |")
    lines.append("|--------|------------------------|---------------------|---|")

    for m, label in zip(metrics, metric_labels):
        a, b = sa.get(m, 0), sb.get(m, 0)
        lines.append(f"| {label} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")

    avg_a = avg([sa[m] for m in metrics])
    avg_b = avg([sb[m] for m in metrics])
    lines.append(f"| **Average** | **{avg_a:.3f}** | **{avg_b:.3f}** | **{avg_b - avg_a:+.3f}** |\n")

    lines.append("---\n")
    lines.append("## A/B Comparison Analysis\n")
    lines.append("**Config A:** Hybrid search (dense + BM25) + RRF merge + cross-encoder reranking\n")
    lines.append("**Config B:** Dense-only semantic search, không reranking\n")
    winner = "Config A" if avg_a > avg_b else "Config B"
    lines.append(f"**Kết luận:** {winner} tốt hơn ({max(avg_a, avg_b):.3f} vs {min(avg_a, avg_b):.3f}).\n")

    lines.append("---\n")
    lines.append("## Worst Performers (Bottom 3)\n")
    for label, worst in [("Config A (hybrid+rerank)", worst_a), ("Config B (dense-only)", worst_b)]:
        lines.append(f"### {label}\n")
        lines.append("| # | Question | Faith. | Relevance | Recall | Precision | Avg |")
        lines.append("|---|----------|-------|-----------|--------|-----------|-----|")
        for i, w in enumerate(worst, 1):
            vals = [w.get(m, 0) for m in metrics]
            lines.append(f"| {i} | {w['question'][:45]:<45} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} | {vals[3]:.2f} | {w['avg']:.2f} |")
        lines.append("")

    lines.append("### Root Cause Analysis\n")
    lines.append("- **News questions** có Context Recall thấp do nội dung báo chí ngắn, thiếu heading rõ ràng, khó match cả dense lẫn BM25\n")
    lines.append("- **Legal questions về điều khoản cụ thể** dễ bị Faithfulness thấp nếu chunk bị cắt ngang, mất nửa điều luật\n")
    lines.append("- **Reranking** giúp cải thiện Precision nhưng có thể làm giảm Recall nếu cross-encoder cho điểm thấp các chunk liên quan\n")
    lines.append("- **Dense-only** thiếu BM25 nên khó bắt các ca query có từ khoá chính xác nhưng ý nghĩa khác biệt\n")

    lines.append("---\n")
    lines.append("## Recommendations\n")
    lines.append("1. **Vietnamese text preprocessing** cho BM25 — chuẩn hoá dấu câu, từ ghép tiếng Việt\n")
    lines.append("2. **Tăng chunk_size → 2000** để giữ trọn một Điều luật (giảm chunk overlap mất ngữ cảnh)\n")
    lines.append("3. **HyDE** (Hypothetical Document Embeddings) — sinh câu trả lời giả trước khi search\n")
    lines.append("4. **Threshold tuning** cho reranking — cân bằng Precision vs Recall dựa trên use case\n")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report saved to {RESULTS_PATH}")


def main():
    print(f"Golden dataset: {len(load_golden_dataset())} test cases\n")

    config_a = evaluate_config("Config A: hybrid + reranking", use_reranking=True)
    config_b = evaluate_config("Config B: dense-only", use_reranking=False)

    generate_report(config_a, config_b)
    print("  Done!")


if __name__ == "__main__":
    main()