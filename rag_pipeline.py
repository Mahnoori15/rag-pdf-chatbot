"""
RAG Pipeline Module
Handles document loading, embedding, and retrieval-augmented generation
"""

import os
from typing import List, Optional

from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGPipeline:
    """Manages RAG pipeline for PDF documents"""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0,
        api_key: Optional[str] = None,
        embeddings=None,
        llm=None,
        top_k: int = 3,
    ):
        """
        Initialize RAG pipeline

        Args:
            model: OpenAI model to use (defaults to the OPENAI_MODEL env var,
                   or "gpt-4o-mini" if that isn't set)
            temperature: Temperature for LLM responses
            api_key: OpenAI API key (defaults to the OPENAI_API_KEY env var)
            embeddings: custom LangChain embeddings object (optional, e.g. for tests)
            llm: custom LangChain chat model (optional, e.g. for tests)
            top_k: number of chunks retrieved per question
        """
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        key_args = {"api_key": api_key} if api_key else {}
        self.embeddings = embeddings or OpenAIEmbeddings(**key_args)
        self.llm = llm or ChatOpenAI(model=model, temperature=temperature, **key_args)
        self.vector_store = None
        self.qa_chain = None

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load PDF document

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of documents
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {pdf_path}")
        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of documents

        Returns:
            List of chunked documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = [doc for doc in text_splitter.split_documents(documents) if doc.page_content.strip()]
        print(f"Split into {len(split_docs)} chunks")
        return split_docs

    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create FAISS vector store from documents

        Args:
            documents: List of documents
        """
        if not documents:
            raise ValueError("No text could be extracted from this PDF (is it a scanned image?).")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        print(f"Created vector store with {len(documents)} documents")

    def setup_qa_chain(self) -> None:
        """Setup QA chain using vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not created. Call create_vector_store first.")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": self.top_k}),
            return_source_documents=True,
        )
        print("QA chain initialized")

    def query_with_sources(self, question: str) -> dict:
        """
        Query the RAG pipeline and return the answer with the chunks it used

        Args:
            question: User question

        Returns:
            {"answer": str, "sources": [{"page": int, "text": str}, ...]}
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call setup_qa_chain first.")

        result = self.qa_chain.invoke({"query": question})
        sources = []
        for doc in result.get("source_documents", []):
            page = doc.metadata.get("page")
            sources.append({
                "page": page + 1 if isinstance(page, int) else None,
                "text": doc.page_content[:500],
            })
        return {"answer": result["result"], "sources": sources}

    def query(self, question: str) -> str:
        """
        Query the RAG pipeline

        Args:
            question: User question

        Returns:
            Answer from RAG pipeline
        """
        return self.query_with_sources(question)["answer"]

    def initialize_from_pdf(self, pdf_path: str) -> dict:
        """
        Complete initialization from PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            {"pages": int, "chunks": int}
        """
        # Load and process
        documents = self.load_pdf(pdf_path)
        processed_docs = self.process_documents(documents)

        # Create vector store and QA chain
        self.create_vector_store(processed_docs)
        self.setup_qa_chain()

        print(f"RAG pipeline ready for {pdf_path}")
        return {"pages": len(documents), "chunks": len(processed_docs)}
