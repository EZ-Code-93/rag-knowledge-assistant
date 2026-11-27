from unittest.mock import MagicMock, patch
import pytest
import requests

import rag_service
from config import Config


# ---------------------------------------------------------------------------
# Tests for answer_question
# ---------------------------------------------------------------------------

def test_answer_question_no_chunks():
    """Returns fallback message and empty sources when no chunks are retrieved."""
    with patch("rag_service.retrieve_relevant_chunks", return_value=[]) as mock_retrieve:
        result = rag_service.answer_question("What is the PTO policy?")

    mock_retrieve.assert_called_once_with("What is the PTO policy?", top_k=Config.TOP_K)
    assert result == {
        "answer": (
            "I could not find enough relevant information in the provided "
            "knowledge base to answer that question."
        ),
        "sources": [],
    }


def test_answer_question_success():
    """Runs full RAG flow: retrieve -> build prompt -> generate -> format sources."""
    sample_chunks = [
        {
            "title": "Employee Handbook",
            "source": "handbook.pdf",
            "chunk_index": 0,
            "text": "Employees receive 15 days of PTO per year.",
        }
    ]

    with patch("rag_service.retrieve_relevant_chunks", return_value=sample_chunks) as mock_retrieve, \
         patch("rag_service.call_generation_model", return_value="You get 15 days of PTO.") as mock_gen:
        result = rag_service.answer_question("How much PTO do I get?")

    mock_retrieve.assert_called_once_with("How much PTO do I get?", top_k=Config.TOP_K)
    mock_gen.assert_called_once()
    assert result["answer"] == "You get 15 days of PTO."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["title"] == "Employee Handbook"


# ---------------------------------------------------------------------------
# Tests for build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_formats_chunks_and_question():
    """Includes source headers, chunk text, and user question in the prompt."""
    chunks = [
        {"title": "Doc A", "source": "a.md", "text": "First fact."},
        {"text": "Second fact without metadata."},
    ]

    prompt = rag_service.build_prompt("What are the facts?", chunks)

    assert "[Source 1: Doc A | a.md]\nFirst fact." in prompt
    assert "[Source 2: Unknown Source | unknown]\nSecond fact without metadata." in prompt
    assert "User question:\nWhat are the facts?" in prompt
    assert prompt.endswith("Answer:")


# ---------------------------------------------------------------------------
# Tests for call_generation_model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_prompt", ["", "   ", None, 123])
def test_call_generation_model_invalid_prompt(bad_prompt):
    """Raises ValueError when prompt is empty, whitespace, or not a string."""
    with pytest.raises(ValueError, match="Prompt must be a non-empty string."):
        rag_service.call_generation_model(bad_prompt)


def test_call_generation_model_success():
    """Sends expected payload to Ollama and returns stripped response text."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "  Generated answer text.  "}

    with patch("rag_service.requests.post", return_value=mock_response) as mock_post:
        result = rag_service.call_generation_model("Test prompt")

    assert result == "Generated answer text."
    mock_post.assert_called_once_with(
        f"{Config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": Config.GENERATION_MODEL,
            "prompt": "Test prompt",
            "stream": False,
            "options": {"temperature": Config.TEMPERATURE},
        },
        timeout=60,
    )
    mock_response.raise_for_status.assert_called_once()


def test_call_generation_model_request_exception():
    """Wraps requests.RequestException in a RuntimeError."""
    with patch(
        "rag_service.requests.post",
        side_effect=requests.RequestException("Connection refused"),
    ):
        with pytest.raises(RuntimeError, match="Failed to generate answer"):
            rag_service.call_generation_model("Test prompt")


def test_call_generation_model_http_error():
    """Wraps HTTP status errors (e.g., 500) in a RuntimeError."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

    with patch("rag_service.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Failed to generate answer"):
            rag_service.call_generation_model("Test prompt")


def test_call_generation_model_invalid_json():
    """Wraps JSON decoding errors in a RuntimeError."""
    mock_response = MagicMock()
    mock_response.json.side_effect = ValueError("No JSON object could be decoded")

    with patch("rag_service.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Failed to generate answer"):
            rag_service.call_generation_model("Test prompt")


def test_call_generation_model_missing_response_key():
    """Raises RuntimeError if Ollama JSON lacks the 'response' key."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"unexpected_key": "value"}

    with patch("rag_service.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Missing response in JSON."):
            rag_service.call_generation_model("Test prompt")


# ---------------------------------------------------------------------------
# Tests for format_sources
# ---------------------------------------------------------------------------

def test_format_sources_truncates_long_excerpts():
    """Truncates excerpts longer than 280 characters and keeps short ones intact."""
    long_text = "A" * 300
    short_text = "Short excerpt."
    chunks = [
        {"title": "Long Doc", "source": "long.txt", "chunk_index": 1, "text": long_text},
        {"text": short_text},
    ]

    formatted = rag_service.format_sources(chunks)

    assert formatted[0] == {
        "title": "Long Doc",
        "source": "long.txt",
        "chunk_index": 1,
        "excerpt": ("A" * 280) + "...",
    }
    assert formatted[1] == {
        "title": "Unknown Source",
        "source": "unknown",
        "chunk_index": None,
        "excerpt": short_text,
    }
