import pytest
from fastapi.testclient import TestClient

import api
from conftest import make_pdf


@pytest.fixture
def client(fake_pipeline_factory, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server")
    monkeypatch.setattr(api, "pipeline_factory", fake_pipeline_factory)
    api._sessions.clear()
    return TestClient(api.app)


def upload(client, data, name="doc.pdf", headers=None):
    return client.post("/api/upload", files={"file": (name, data, "application/pdf")}, headers=headers or {})


def test_index_and_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200 and "Ask your PDF" in page.text
    assert client.get("/static/style.css").status_code == 200


def test_config(client):
    config = client.get("/api/config").json()
    assert config["server_key"] is True
    assert config["max_pages"] == api.MAX_PAGES


def test_upload_and_ask(client, sample_pdf):
    response = upload(client, sample_pdf.read_bytes(), "atlantis.pdf")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["filename"] == "atlantis.pdf" and body["pages"] == 2

    answer = client.post("/api/ask", json={"session_id": body["session_id"], "question": "Capital?"})
    assert answer.status_code == 200, answer.text
    result = answer.json()
    assert result["answer"] == "Poseidonia."
    assert result["sources"] and result["questions_left"] == api.MAX_QUESTIONS - 1


def test_requires_a_key(client, sample_pdf, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    assert upload(client, sample_pdf.read_bytes()).status_code == 400
    ok = upload(client, sample_pdf.read_bytes(), headers={"X-OpenAI-Key": "sk-user"})
    assert ok.status_code == 200


def test_user_key_is_passed_to_pipeline(client, sample_pdf, fake_pipeline_factory, monkeypatch):
    seen = {}

    def factory(api_key=None):
        seen["key"] = api_key
        return fake_pipeline_factory(api_key=api_key)

    monkeypatch.setattr(api, "pipeline_factory", factory)
    upload(client, sample_pdf.read_bytes(), headers={"X-OpenAI-Key": "sk-user"})
    assert seen["key"] == "sk-user"


def test_rejects_bad_uploads(client, monkeypatch):
    assert upload(client, b"hello, not a pdf").status_code == 400
    assert upload(client, b"%PDF-1.4 broken").status_code == 400
    monkeypatch.setattr(api, "MAX_PAGES", 1)
    assert upload(client, make_pdf(["one", "two"])).status_code == 413
    monkeypatch.setattr(api, "MAX_PDF_BYTES", 10)
    assert upload(client, make_pdf(["one"])).status_code == 413


def test_scanned_pdf_gives_clear_error(client):
    response = upload(client, make_pdf([""]))
    assert response.status_code == 422
    assert "No text" in response.json()["detail"]


def test_unknown_session_and_question_limit(client, sample_pdf, monkeypatch):
    assert client.post("/api/ask", json={"session_id": "nope", "question": "x"}).status_code == 404
    monkeypatch.setattr(api, "MAX_QUESTIONS", 1)
    session_id = upload(client, sample_pdf.read_bytes()).json()["session_id"]
    first = client.post("/api/ask", json={"session_id": session_id, "question": "a"})
    second = client.post("/api/ask", json={"session_id": session_id, "question": "b"})
    assert first.status_code == 200 and second.status_code == 429


def test_old_sessions_are_evicted(client, sample_pdf, monkeypatch):
    monkeypatch.setattr(api, "MAX_SESSIONS", 2)
    ids = [upload(client, sample_pdf.read_bytes()).json()["session_id"] for _ in range(3)]
    assert client.post("/api/ask", json={"session_id": ids[0], "question": "x"}).status_code == 404
    assert client.post("/api/ask", json={"session_id": ids[2], "question": "x"}).status_code == 200
