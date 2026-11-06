# rag-pdf-chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions about a PDF.

## How it works

1. **Load** – the PDF is read page by page with `PyPDFLoader`.
2. **Chunk** – pages are split into ~1,000-character chunks (200-character overlap).
3. **Embed & index** – chunks are embedded with OpenAI embeddings and stored in a FAISS vector index.
4. **Retrieve & answer** – for each question, the 3 most relevant chunks are retrieved and passed to an OpenAI chat model, which answers using only that context.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your OpenAI API key
```

## Usage

```bash
python app.py path/to/document.pdf
```

Type a question at the prompt, or `quit` to exit.

The chat model defaults to `gpt-4o-mini`. To use a different one, pass `--model` or set `OPENAI_MODEL` in `.env`:

```bash
python app.py report.pdf --model gpt-4o
```

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Command-line interface |
| `rag_pipeline.py` | `RAGPipeline` class: loading, chunking, indexing and querying |
| `requirements.txt` | Pinned dependencies |
| `.env.example` | Template for environment variables |

## Using the pipeline in your own code

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.initialize_from_pdf("report.pdf")
print(pipeline.query("What are the key findings?"))
```
