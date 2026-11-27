from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
import requests

import vector_store
from config import Config


# ---------------------------------------------------------------------------
# Tests for get_chroma_client & get_or_create_collection
# ---------------------------------------------------------------------------

def test_get_chroma_client_uses_config_path():
    """Creates a PersistentClient using Config.CHROMA_PATH."""
    with patch("vector_store.chromadb.PersistentClient") as mock_client_cls:
        client = vector_store.get_chroma_client()

    mock_client_cls.assert_called_once_with(path=Config.CHROMA_PATH)
    assert client == mock_client_cls.return_value


def test_get_or_create_collection_uses_config_name():
    """Gets or creates the collection using Config.COLLECTION_NAME."""
    mock_client = MagicMock()
    with patch("vector_store.get_chroma_client", return_value=mock_client):
        collection = vector_store.get_or_create_collection()

    mock_client.get_or_create_collection.assert_called_once_with(
        name=Config.COLLECTION_NAME
    )
    assert collection == mock_client.get_or_create_collection.return_value


# ---------------------------------------------------------------------------
# Tests for get_embedding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_text", ["", "   ", None, 42])
def test_get_embedding_invalid_input(bad_text):
    """Raises ValueError when text is empty, whitespace, or not a string."""
    with pytest.raises(ValueError, match="Text must be a non-empty string."):
        vector_store.get_embedding(bad_text)


def test_get_embedding_success():
    """Posts to /api/embed with 'input' and returns the first embedding vector."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model": Config.EMBEDDING_MODEL,
        "embeddings": [[0.1, 0.2, 0.3]],
    }

    with patch("vector_store.requests.post", return_value=mock_response) as mock_post:
        result = vector_store.get_embedding("Why is the sky blue?")

    assert result == [0.1, 0.2, 0.3]
    mock_post.assert_called_once_with(
        f"{Config.OLLAMA_BASE_URL}/api/embed",
        json={
            "model": Config.EMBEDDING_MODEL,
            "input": "Why is the sky blue?",
        },
        timeout=60,
    )
    mock_response.raise_for_status.assert_called_once()


def test_get_embedding_request_exception():
    """Wraps requests.RequestException in a RuntimeError."""
    with patch(
        "vector_store.requests.post",
        side_effect=requests.RequestException("Connection refused"),
    ):
        with pytest.raises(RuntimeError, match="Failed to generate embedding"):
            vector_store.get_embedding("Hello")


def test_get_embedding_invalid_json():
    """Wraps JSON decoding errors in a RuntimeError."""
    mock_response = MagicMock()
    mock_response.json.side_effect = ValueError("Invalid JSON")

    with patch("vector_store.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Failed to generate embedding"):
            vector_store.get_embedding("Hello")


@pytest.mark.parametrize(
    "bad_payload",
    [
        {},
        {"embeddings": []},
        {"embeddings": [[]]},
        {"embeddings": "not-a-list"},
    ],
)
def test_get_embedding_missing_or_empty_embeddings(bad_payload):
    """Raises RuntimeError when 'embeddings' is missing or empty."""
    mock_response = MagicMock()
    mock_response.json.return_value = bad_payload

    with patch("vector_store.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Missing or empty embeddings"):
            vector_store.get_embedding("Hello")


# ---------------------------------------------------------------------------
# Tests for seed_vector_store
# ---------------------------------------------------------------------------

def test_seed_vector_store_empty_chunks():
    """Returns 0 and skips Chroma calls when given an empty list."""
    with patch("vector_store.get_or_create_collection") as mock_get_col:
        count = vector_store.seed_vector_store([])

    assert count == 0
    mock_get_col.assert_not_called()


def test_seed_vector_store_upserts_chunks():
    """Embeds each chunk, upserts into Chroma, and returns chunk count."""
    mock_collection = MagicMock()
    chunks = [
        SimpleNamespace(
            id="doc1_0",
            text="First chunk text.",
            source="doc1.md",
            title="Doc One",
            chunk_index=0,
        ),
        SimpleNamespace(
            id="doc1_1",
            text="Second chunk text.",
            source="doc1.md",
            title="Doc One",
            chunk_index=1,
        ),
    ]

    with patch("vector_store.get_or_create_collection", return_value=mock_collection), \
         patch("vector_store.get_embedding", side_effect=[[0.1, 0.2], [0.3, 0.4]]) as mock_embed:
        count = vector_store.seed_vector_store(chunks)

    assert count == 2
    assert mock_embed.call_count == 2
    mock_collection.upsert.assert_called_once_with(
        ids=["doc1_0", "doc1_1"],
        documents=["First chunk text.", "Second chunk text."],
        metadatas=[
            {"source": "doc1.md", "title": "Doc One", "chunk_index": 0},
            {"source": "doc1.md", "title": "Doc One", "chunk_index": 1},
        ],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
    )


# ---------------------------------------------------------------------------
# Tests for retrieve_relevant_chunks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_question", ["", "   ", None])
def test_retrieve_relevant_chunks_blank_question(bad_question):
    """Returns an empty list without querying Chroma for blank questions."""
    with patch("vector_store.get_or_create_collection") as mock_get_col:
        result = vector_store.retrieve_relevant_chunks(bad_question)

    assert result == []
    mock_get_col.assert_not_called()


def test_retrieve_relevant_chunks_empty_collection():
    """Returns an empty list when the Chroma collection has 0 items."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    with patch("vector_store.get_or_create_collection", return_value=mock_collection), \
         patch("vector_store.get_embedding") as mock_embed:
        result = vector_store.retrieve_relevant_chunks("What is the policy?")

    assert result == []
    mock_embed.assert_not_called()
    mock_collection.query.assert_not_called()


def test_retrieve_relevant_chunks_success():
    """Queries Chroma with question embedding and formats returned chunks."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_collection.query.return_value = {
        "documents": [["PTO is 15 days.", "Carryover limit is 5 days."]],
        "metadatas": [[
            {"source": "handbook.pdf", "title": "Employee Handbook", "chunk_index": 0},
            {"source": "handbook.pdf", "title": "Employee Handbook", "chunk_index": 1},
        ]],
        "distances": [[0.12, 0.34]],
    }

    with patch("vector_store.get_or_create_collection", return_value=mock_collection), \
         patch("vector_store.get_embedding", return_value=[0.5, 0.6]) as mock_embed:
        result = vector_store.retrieve_relevant_chunks("How much PTO?", top_k=2)

    mock_embed.assert_called_once_with("How much PTO?")
    mock_collection.query.assert_called_once_with(
        query_embeddings=[[0.5, 0.6]],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )
    assert result == [
        {
            "text": "PTO is 15 days.",
            "source": "handbook.pdf",
            "title": "Employee Handbook",
            "chunk_index": 0,
            "distance": 0.12,
        },
        {
            "text": "Carryover limit is 5 days.",
            "source": "handbook.pdf",
            "title": "Employee Handbook",
            "chunk_index": 1,
            "distance": 0.34,
        },
    ]


def test_retrieve_relevant_chunks_caps_n_results_to_collection_count():
    """Caps n_results to collection.count() when top_k exceeds available chunks."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        "documents": [["Only doc."]],
        "metadatas": [[{"source": "a.txt", "title": "Doc A", "chunk_index": 0}]],
        "distances": [[0.05]],
    }

    with patch("vector_store.get_or_create_collection", return_value=mock_collection), \
         patch("vector_store.get_embedding", return_value=[0.1, 0.2]):
        vector_store.retrieve_relevant_chunks("Question", top_k=5)

    mock_collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2]],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )
