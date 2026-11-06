"""
RAG PDF Chatbot - web app (FastAPI).

Run locally:
    uvicorn api:app --reload --port 7860
then open http://localhost:7860

Configuration (environment variables):
    OPENAI_API_KEY     server key used when visitors don't supply their own (optional)
    OPENAI_MODEL       chat model (default gpt-4o-mini)
    MAX_PDF_MB         largest upload accepted (default 10)
    MAX_PAGES          most pages indexed per PDF (default 100)
    MAX_SESSIONS       PDFs kept in memory at once (default 20)
    MAX_QUESTIONS      questions allowed per uploaded PDF (default 30)
    SESSION_TTL_MIN    minutes before an idle PDF is dropped (default 60)
"""

import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag_pipeline import RAGPipeline

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


MAX_PDF_BYTES = _env_int("MAX_PDF_MB", 10) * 1024 * 1024
MAX_PAGES = _env_int("MAX_PAGES", 100)
MAX_SESSIONS = _env_int("MAX_SESSIONS", 20)
MAX_QUESTIONS = _env_int("MAX_QUESTIONS", 30)
SESSION_TTL = _env_int("SESSION_TTL_MIN", 60) * 60

app = FastAPI(title="RAG PDF Chatbot", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Swappable so tests can inject a pipeline with fake models.
pipeline_factory = RAGPipeline


class Session:
    def __init__(self, pipeline, filename: str, pages: int, chunks: int):
        self.pipeline = pipeline
        self.filename = filename
        self.pages = pages
        self.chunks = chunks
        self.questions = 0
        self.last_used = time.time()


_sessions: "OrderedDict[str, Session]" = OrderedDict()
_lock = threading.Lock()


def _store(session: Session) -> str:
    session_id = uuid.uuid4().hex
    with _lock:
        _sessions[session_id] = session
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)  # drop the oldest
    return session_id


def _get(session_id: str) -> Session:
    with _lock:
        now = time.time()
        for key in [k for k, s in _sessions.items() if now - s.last_used > SESSION_TTL]:
            del _sessions[key]
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "This PDF session has expired. Please upload the PDF again.")
        session.last_used = now
        _sessions.move_to_end(session_id)
        return session


def _resolve_key(user_key: Optional[str]) -> Optional[str]:
    key = (user_key or "").strip() or os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(400, "An OpenAI API key is required. Add yours in the settings box.")
    return key


def _openai_error(error: Exception) -> HTTPException:
    if isinstance(error, openai.AuthenticationError):
        return HTTPException(401, "The OpenAI API key was rejected. Check it and try again.")
    if isinstance(error, openai.RateLimitError):
        return HTTPException(429, "OpenAI rate limit or quota reached. Try again later.")
    if isinstance(error, openai.OpenAIError):
        return HTTPException(502, f"OpenAI request failed: {error.__class__.__name__}")
    return HTTPException(500, "Something went wrong while processing the request.")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return {
        "server_key": bool(os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "max_pdf_mb": MAX_PDF_BYTES // (1024 * 1024),
        "max_pages": MAX_PAGES,
        "max_questions": MAX_QUESTIONS,
    }


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    x_openai_key: Optional[str] = Header(default=None),
):
    api_key = _resolve_key(x_openai_key)

    data = await file.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, f"PDF is larger than {MAX_PDF_BYTES // (1024 * 1024)} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "That file isn't a PDF.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        try:
            pages = len(PdfReader(path).pages)
        except (PdfReadError, ValueError):
            raise HTTPException(400, "The PDF could not be read. It may be damaged or encrypted.")
        if pages > MAX_PAGES:
            raise HTTPException(413, f"PDF has {pages} pages; the limit is {MAX_PAGES}.")

        pipeline = pipeline_factory(api_key=api_key)
        try:
            stats = await run_in_threadpool(pipeline.initialize_from_pdf, path)
        except ValueError as error:
            raise HTTPException(422, str(error))
        except Exception as error:
            raise _openai_error(error)
    finally:
        os.unlink(path)

    filename = Path(file.filename or "document.pdf").name
    session_id = _store(Session(pipeline, filename, stats["pages"], stats["chunks"]))
    return {"session_id": session_id, "filename": filename, **stats}


class Question(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=1000)


@app.post("/api/ask")
async def ask(body: Question):
    session = _get(body.session_id)
    if session.questions >= MAX_QUESTIONS:
        raise HTTPException(429, f"Question limit ({MAX_QUESTIONS}) reached for this PDF.")
    session.questions += 1
    try:
        result = await run_in_threadpool(session.pipeline.query_with_sources, body.question.strip())
    except Exception as error:
        raise _openai_error(error)
    return {**result, "questions_left": MAX_QUESTIONS - session.questions}
