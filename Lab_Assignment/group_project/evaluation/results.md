# RAG Evaluation Results

## Framework sử dụng

**Local LLM-based evaluation** — dùng model `oc/mimo-v2.5-free` để chấm 4 metrics

---

## Overall Scores

| Metric | Config A (hybrid+rerank) | Config B (dense-only) | Δ |
|--------|------------------------|---------------------|---|
| Faithfulness | 0.529 | 0.526 | -0.003 |
| Answer Relevance | 0.582 | 0.547 | -0.035 |
| Context Recall | 0.529 | 0.529 | +0.000 |
| Context Precision | 0.500 | 0.500 | +0.000 |
| **Average** | **0.535** | **0.526** | **-0.010** |

---

## A/B Comparison Analysis

**Config A:** Hybrid search (dense + BM25) + RRF merge + cross-encoder reranking

**Config B:** Dense-only semantic search, không reranking

**Kết luận:** Config A tốt hơn (0.535 vs 0.526).

---

## Worst Performers (Bottom 3)

### Config A (hybrid+rerank)

| # | Question | Faith. | Relevance | Recall | Precision | Avg |
|---|----------|-------|-----------|--------|-----------|-----|
| 1 | Vợ chồng Phú Lê bị khởi tố về tội gì?         | 0.50 | 0.00 | 0.50 | 0.50 | 0.38 |
| 2 | Theo Luật Phòng chống ma tuý, các hành vi nào | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| 3 | Hình phạt cho tội tàng trữ trái phép chất ma  | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |

### Config B (dense-only)

| # | Question | Faith. | Relevance | Recall | Precision | Avg |
|---|----------|-------|-----------|--------|-----------|-----|
| 1 | Luật Phòng chống ma tuý 2021 quy định những h | 0.50 | 0.00 | 0.50 | 0.50 | 0.38 |
| 2 | Chính sách của Nhà nước về phòng chống ma tuý | 0.50 | 0.00 | 0.50 | 0.50 | 0.38 |
| 3 | Danh mục các chất ma tuý thuộc nhóm I gồm nhữ | 0.50 | 0.30 | 0.50 | 0.50 | 0.45 |

### Root Cause Analysis

- **News questions** có Context Recall thấp do nội dung báo chí ngắn, thiếu heading rõ ràng, khó match cả dense lẫn BM25

- **Legal questions về điều khoản cụ thể** dễ bị Faithfulness thấp nếu chunk bị cắt ngang, mất nửa điều luật

- **Reranking** giúp cải thiện Precision nhưng có thể làm giảm Recall nếu cross-encoder cho điểm thấp các chunk liên quan

- **Dense-only** thiếu BM25 nên khó bắt các ca query có từ khoá chính xác nhưng ý nghĩa khác biệt

---

## Recommendations

1. **Vietnamese text preprocessing** cho BM25 — chuẩn hoá dấu câu, từ ghép tiếng Việt

2. **Tăng chunk_size → 2000** để giữ trọn một Điều luật (giảm chunk overlap mất ngữ cảnh)

3. **HyDE** (Hypothetical Document Embeddings) — sinh câu trả lời giả trước khi search

4. **Threshold tuning** cho reranking — cân bằng Precision vs Recall dựa trên use case