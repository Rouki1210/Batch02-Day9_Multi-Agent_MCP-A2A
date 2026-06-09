"""
Task 4 - Chunking & Indexing with LangChain, Gemini embeddings, and Weaviate.

Run:
    python src/task4_chunking_indexing.py

Required .env values:
    GEMINI_API_KEY=...

Optional .env values:
    WEAVIATE_URL=https://xxx.weaviate.network
    WEAVIATE_API_KEY=...

If WEAVIATE_URL is not set, the script tries to connect to local Weaviate.
"""

import hashlib
import json
import math
import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# JSONL is kept as a local cache/debug artifact. Weaviate is the real vector
# store for this Task 4 version.
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
INDEX_FILE = INDEX_DIR / "chunks.jsonl"

# RecursiveCharacterTextSplitter is the README-recommended safe default for
# mixed Markdown converted from legal PDFs and news JSON. It tries paragraph and
# line boundaries before falling back to sentence/word boundaries.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "RecursiveCharacterTextSplitter"

# Gemini embedding model for RAG retrieval. The model supports configurable
# output dimensionality; 1536 is a balanced size for retrieval quality while
# keeping Weaviate vectors smaller than the full 3072-dimensional output.
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))

# Weaviate is used as the primary vector store because it supports vector search
# and can later support hybrid retrieval with BM25.
VECTOR_STORE = "weaviate"
WEAVIATE_COLLECTION = "DrugLawDocs"


def load_documents() -> list[dict]:
    """
    Read all Markdown files from data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, ...}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(relative_path).replace("\\", "/"),
                    "type": doc_type,
                },
            }
        )

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents with LangChain RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ImportError(
            "Task 4 requires langchain-text-splitters. "
            "Run: pip install langchain-text-splitters"
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for index, chunk_text in enumerate(splits):
            clean_text = chunk_text.strip()
            if not clean_text:
                continue

            chunks.append(
                {
                    "content": clean_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": index,
                        "chunking_method": CHUNKING_METHOD,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                    },
                }
            )

    return chunks


def embed_chunks(chunks: list[dict], batch_size: int = 64) -> list[dict]:
    """
    Embed chunks with Gemini embedding API.

    Returns:
        Each chunk dict with an extra 'embedding': list[float]
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env before embedding.")

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError("Task 4 requires google-genai. Run: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk["content"].replace("\n", " ") for chunk in batch]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIM,
            ),
        )

        for chunk, embedding_item in zip(batch, response.embeddings):
            chunk["embedding"] = embedding_item.values
            chunk["metadata"]["embedding_model"] = EMBEDDING_MODEL
            chunk["metadata"]["embedding_dim"] = EMBEDDING_DIM

    return chunks


def connect_weaviate():
    """
    Connect to Weaviate Cloud when WEAVIATE_URL is set; otherwise local Weaviate.
    """
    try:
        import weaviate
        from weaviate.auth import Auth
    except ImportError as exc:
        raise ImportError("Task 4 requires weaviate-client. Run: pip install weaviate-client") from exc

    url = os.getenv("WEAVIATE_URL", "").strip()
    api_key = os.getenv("WEAVIATE_API_KEY", "").strip()

    if url:
        if not api_key:
            raise RuntimeError("WEAVIATE_API_KEY is required when WEAVIATE_URL is set.")
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
        )

    return weaviate.connect_to_local()


def self_provided_vector_config():
    """Return a Weaviate self-provided vector config across v4 client versions."""
    from weaviate.classes.config import Configure

    if hasattr(Configure, "Vectors") and hasattr(Configure.Vectors, "self_provided"):
        return Configure.Vectors.self_provided()
    if hasattr(Configure, "Vectorizer") and hasattr(Configure.Vectorizer, "none"):
        return Configure.Vectorizer.none()
    raise RuntimeError("Unsupported weaviate-client vector configuration API.")


def create_or_replace_collection(client, reset_collection: bool = True):
    """Create the Weaviate collection used by this RAG pipeline."""
    from weaviate.classes.config import DataType, Property

    if client.collections.exists(WEAVIATE_COLLECTION):
        if not reset_collection:
            return client.collections.get(WEAVIATE_COLLECTION)
        client.collections.delete(WEAVIATE_COLLECTION)

    return client.collections.create(
        name=WEAVIATE_COLLECTION,
        vector_config=self_provided_vector_config(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="path", data_type=DataType.TEXT),
            Property(name="doc_type", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
        ],
    )


def write_local_cache(chunks: list[dict]) -> Path:
    """Write embedded chunks to JSONL for debugging and downstream local modules."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with INDEX_FILE.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return INDEX_FILE


def index_to_vectorstore(chunks: list[dict]) -> str:
    """
    Index embedded chunks into Weaviate and write a local JSONL cache.
    """
    if not chunks:
        write_local_cache(chunks)
        return WEAVIATE_COLLECTION

    client = connect_weaviate()
    try:
        collection = create_or_replace_collection(client)

        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": metadata.get("source", ""),
                        "path": metadata.get("path", ""),
                        "doc_type": metadata.get("type", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                    },
                    vector=chunk["embedding"],
                )

        write_local_cache(chunks)
        return WEAVIATE_COLLECTION
    finally:
        client.close()


def run_pipeline() -> str:
    """Run the full pipeline: load -> chunk -> Gemini embed -> Weaviate index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: Gemini {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} collection={WEAVIATE_COLLECTION}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    collection_name = index_to_vectorstore(chunks)
    print(f"Indexed to Weaviate collection: {collection_name}")
    print(f"Local cache: {INDEX_FILE}")
    return collection_name


# Compatibility helpers retained for the local Task 5/7 fallback modules.
def tokenize(text: str) -> list[str]:
    """Normalize text into simple lowercase tokens."""
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


def hashing_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic fallback embedding for local-only modules."""
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


if __name__ == "__main__":
    run_pipeline()
