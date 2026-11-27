from typing import Any, List

import chromadb
import requests

from config import Config
from documents import DocumentChunk


def get_chroma_client():
    """
    Create and return a persistent Chroma client.

    TODO:
    - Use Config.CHROMA_PATH as the local storage path.
    - Return a chromadb.PersistentClient.
    """
    return chromadb.PersistentClient(path=Config.CHROMA_PATH)


def get_or_create_collection():
    """
    Get or create the Chroma collection for the knowledge assistant.

    TODO:
    - Use get_chroma_client().
    - Use Config.COLLECTION_NAME as the collection name.
    - Return the collection.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(name=Config.COLLECTION_NAME)


def get_embedding(text: str) -> list[float]:
    """
    Create an embedding for a piece of text using the local model service.

    TODO:
    - Send a POST request to the Ollama embed endpoint (https://docs.ollama.com/api/embed).
    - Use Config.OLLAMA_BASE_URL.
    - Use Config.EMBEDDING_MODEL.
    - Return the embedding list from the response.

    Endpoint:
        POST {OLLAMA_BASE_URL}/api/embed

    Example request body:
        {
            "model": Config.EMBEDDING_MODEL,
            "prompt": text
        }
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")

    try:
        response = requests.post(
            f"{Config.OLLAMA_BASE_URL}/api/embed",
            json={
                "model": Config.EMBEDDING_MODEL,
                "input": text,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as e:
        raise RuntimeError(f"Failed to generate embedding: {e}") from e

    embeddings = data.get("embeddings")
    if not embeddings or not isinstance(embeddings, list) or not embeddings[0]:
        raise RuntimeError("Missing or empty embeddings in JSON response.")

    return embeddings[0]


def seed_vector_store(chunks: List[DocumentChunk]) -> int:
    """
    Add document chunks to the Chroma collection.

    TODO:
    - Get or create the collection.
    - Convert each chunk into:
        - id
        - document text
        - metadata with source, title, and chunk_index
        - embedding
    - Add or update the chunks in Chroma (recommend using collection.upsert(...) to prevent duplicating existing records).
    - Return the number of chunks added.

    Keep source metadata because the frontend needs to display sources.
    """
    if not chunks:
        return 0

    collection = get_or_create_collection()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []

    for chunk in chunks:
        ids.append(chunk.id)
        documents.append(chunk.text)
        metadatas.append(
            {
                "source": chunk.source,
                "title": chunk.title,
                "chunk_index": chunk.chunk_index,
            }
        )
        embeddings.append(get_embedding(chunk.text))

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks)


def retrieve_relevant_chunks(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Retrieve relevant chunks for a user question.

    TODO:
    - Create an embedding for the question.
    - Query the Chroma collection.
    - Return a list of dictionaries with:
        - text
        - source
        - title
        - chunk_index
        - optional distance or score

    The RAG workflow expects a list shaped like this:

        [
            {
                "text": "Relevant source text...",
                "source": "product_support.txt",
                "title": "Product Support Guide",
                "chunk_index": 0
            }
        ]
    """
    if not isinstance(question, str) or not question.strip():
        return []

    k = top_k if top_k is not None else Config.TOP_K
    collection = get_or_create_collection()

    if collection.count() == 0:
        return []

    query_embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    chunks: list[dict[str, Any]] = []
    for index, doc_text in enumerate(documents):
        meta = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        chunk_data: dict[str, Any] = {
            "text": doc_text,
            "source": meta.get("source", "unknown"),
            "title": meta.get("title", "Unknown Source"),
            "chunk_index": meta.get("chunk_index", 0),
        }
        if index < len(distances):
            chunk_data["distance"] = distances[index]
        chunks.append(chunk_data)

    return chunks
