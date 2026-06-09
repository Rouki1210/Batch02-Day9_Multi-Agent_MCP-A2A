"""
Streamlit RAG chatbot for the Day 8 drug-law/news pipeline.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.task10_generation import generate_with_citation


st.set_page_config(
    page_title="Drug Law RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
)


def initialize_state() -> None:
    """Initialize chat state."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Chào bạn. Mình có thể trả lời câu hỏi về văn bản pháp luật "
                    "phòng chống ma túy và các bài báo đã crawl, kèm citation."
                ),
                "sources": [],
                "retrieval_source": "none",
            }
        ]


def reset_chat() -> None:
    """Clear chat history."""
    st.session_state.messages = []
    initialize_state()


def render_source_card(source: dict, index: int) -> None:
    """Render one retrieved source chunk."""
    metadata = source.get("metadata", {})
    title = metadata.get("source", f"Source {index}")
    doc_type = metadata.get("type", "unknown")
    score = float(source.get("score", 0.0))
    retrieval_source = source.get("source", "hybrid")

    with st.expander(f"{index}. {title} · {doc_type} · {retrieval_source} · {score:.3f}"):
        st.caption(metadata.get("path", ""))
        st.write(source.get("content", ""))


def answer_question(question: str, top_k: int) -> dict:
    """Call the RAG generation layer and normalize error handling."""
    try:
        return generate_with_citation(question, top_k=top_k)
    except Exception as exc:
        return {
            "answer": f"Không thể tạo câu trả lời lúc này: {type(exc).__name__}: {exc}",
            "sources": [],
            "retrieval_source": "error",
            "context": "",
        }


initialize_state()

with st.sidebar:
    st.title("RAG Controls")
    top_k = st.slider("Số nguồn truy xuất", min_value=1, max_value=10, value=5)
    st.button("Xóa hội thoại", on_click=reset_chat, use_container_width=True)

    st.divider()
    st.subheader("Pipeline")
    st.write("Task 5: semantic search")
    st.write("Task 6: BM25 lexical search")
    st.write("Task 7: reranking")
    st.write("Task 9: hybrid retrieval")
    st.write("Task 10: citation answer")

    st.divider()
    st.caption("App dùng dữ liệu trong `data/standardized/` và index/cache trong `data/index/`.")

st.title("Drug Law & News RAG Chatbot")
st.caption("Hỏi về pháp luật phòng chống ma túy hoặc các bài báo nghệ sĩ liên quan tới ma túy.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        sources = message.get("sources") or []
        if sources:
            st.caption(f"Retrieval: {message.get('retrieval_source', 'unknown')}")
            for index, source in enumerate(sources, 1):
                render_source_card(source, index)

suggestions = [
    "Luật phòng chống ma túy 2021 quy định gì về cai nghiện?",
    "Những nghệ sĩ nào trong dữ liệu liên quan tới ma túy?",
    "Nghị định 57/2022 quy định gì về danh mục chất ma túy?",
]

with st.container():
    columns = st.columns(len(suggestions))
    for column, suggestion in zip(columns, suggestions):
        if column.button(suggestion, use_container_width=True):
            st.session_state.pending_prompt = suggestion

prompt = st.chat_input("Nhập câu hỏi...")
if not prompt and "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và tạo câu trả lời..."):
            result = answer_question(prompt, top_k=top_k)

        st.write(result["answer"])
        sources = result.get("sources", [])
        if sources:
            st.caption(f"Retrieval: {result.get('retrieval_source', 'unknown')}")
            for index, source in enumerate(sources, 1):
                render_source_card(source, index)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "retrieval_source": result.get("retrieval_source", "unknown"),
        }
    )
