import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

def setup_rag_pipeline(pdf_path):
    """Setup RAG pipeline with PDF document"""
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Setup QA chain
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    
    return qa_chain

def main():
    """Main function"""
    print("RAG PDF Chatbot")
    print("=" * 50)
    
    # Example usage
    pdf_path = "sample.pdf"  # Replace with your PDF path
    
    if os.path.exists(pdf_path):
        qa_chain = setup_rag_pipeline(pdf_path)
        
        # Interactive chat
        while True:
            question = input("\nAsk a question about the PDF (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            
            answer = qa_chain.run(question)
            print(f"\nAnswer: {answer}")
    else:
        print(f"PDF file not found: {pdf_path}")

if __name__ == "__main__":
    main()
