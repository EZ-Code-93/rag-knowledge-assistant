from unittest.mock import patch
import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """GET /api/health returns 200 and status ok."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ok",
        "message": "Backend is running.",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": None},
        {"question": 123},
    ],
)
def test_ask_question_missing_or_invalid_question(client, payload):
    """POST /api/ask returns 400 when question is missing, blank, or non-string."""
    response = client.post("/api/ask", json=payload)
    assert response.status_code == 400
    assert response.get_json() == {"error": "Question is required."}


def test_ask_question_success(client):
    """POST /api/ask calls answer_question with stripped text and returns 200."""
    mock_result = {
        "answer": "You get 15 days of PTO.",
        "sources": [{"title": "Handbook", "source": "handbook.pdf", "chunk_index": 0, "excerpt": "..."}],
    }

    with patch("app.answer_question", return_value=mock_result) as mock_ask:
        response = client.post("/api/ask", json={"question": "  How much PTO?  "})

    mock_ask.assert_called_once_with("How much PTO?")
    assert response.status_code == 200
    assert response.get_json() == mock_result


def test_ask_question_model_service_failure(client):
    """POST /api/ask returns 502 when rag_service raises RuntimeError."""
    with patch("app.answer_question", side_effect=RuntimeError("Failed to generate answer")):
        response = client.post("/api/ask", json={"question": "How much PTO?"})

    assert response.status_code == 502
    data = response.get_json()
    assert "Failed to generate answer" in data["error"]
    assert data["sources"] == []
