FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Hugging Face Spaces runs containers as user 1000
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app rag_pipeline.py api.py ./
COPY --chown=app:app static ./static

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
