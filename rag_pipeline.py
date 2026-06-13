"""
RAG Pipeline Module
Handles document loading, embedding, and retrieval-augmented generation
"""

import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


class RAGPipeline:
    """Manages RAG pipeline for PDF documents"""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0):
        """
        Initialize RAG pipeline
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for LLM responses
        """
        self.model = model
        self.temperature = temperature
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=model, temperature=temperature)
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
        split_docs = text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks")
        return split_docs
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create FAISS vector store from documents
        
        Args:
            documents: List of documents
        """
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        print(f"Created vector store with {len(documents)} documents")
    
    def setup_qa_chain(self) -> None:
        """Setup QA chain using vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not created. Call create_vector_store first.")
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
        )
        print("QA chain initialized")
    
    def query(self, question: str) -> str:
        """
        Query the RAG pipeline
        
        Args:
            question: User question
            
        Returns:
            Answer from RAG pipeline
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call setup_qa_chain first.")
        
        answer = self.qa_chain.run(question)
        return answer
    
    def initialize_from_pdf(self, pdf_path: str) -> None:
        """
        Complete initialization from PDF file
        
        Args:
            pdf_path: Path to PDF file
        """
        # Load and process
        documents = self.load_pdf(pdf_path)
        processed_docs = self.process_documents(documents)
        
        # Create vector store and QA chain
        self.create_vector_store(processed_docs)
        self.setup_qa_chain()
        
        print(f"RAG pipeline ready for {pdf_path}")
