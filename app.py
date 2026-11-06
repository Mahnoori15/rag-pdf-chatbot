"""
RAG PDF Chatbot - command-line interface.

Usage:
    python app.py path/to/document.pdf
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from rag_pipeline import RAGPipeline


def main():
    """Load a PDF and start an interactive question-answering session."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ask questions about a PDF.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI chat model (default: OPENAI_MODEL env var or gpt-4o-mini)",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    print("RAG PDF Chatbot")
    print("=" * 50)

    try:
        pipeline = RAGPipeline(model=args.model)
        pipeline.initialize_from_pdf(args.pdf_path)
    except FileNotFoundError as error:
        sys.exit(str(error))

    while True:
        try:
            question = input("\nAsk a question about the PDF (or 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        answer = pipeline.query(question)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
