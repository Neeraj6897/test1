import os
from dotenv import load_dotenv
from github_fetcher import GithubIssuesCollector
from vector_store import VectorStoreManager
from query_engine import QueryEngine

def main():
    # Load environment variables
    load_dotenv()
    
    # Configuration
    GITHUB_REPO = "llvm/llvm-project"
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Initialize components
    collector = GithubIssuesCollector(github_token=GITHUB_TOKEN)
    vector_store = VectorStoreManager(persist_directory="github_issues_db")
    query_engine = QueryEngine(groq_api_key=GROQ_API_KEY)
    
    # Check for existing vector store
    if not vector_store.load_vectorstore():
        # Fetch and process issues
        issues = collector.fetch_issues(repo=GITHUB_REPO, max_issues=50)
        documents = vector_store.prepare_documents(issues)
        vector_store.create_vectorstore(documents)
    
    # Setup retriever and RAG chain
    retriever = vector_store.get_retriever(k=1)
    rag_chain = query_engine.create_rag_chain(retriever)
    
    # Example query
    query = "Explain the error caused by O2 flag and also share the source from where error was observed."
    response = rag_chain.invoke(query)
    
    # Process and display results
    results = query_engine.process_response(response)
    print("\nQuery Results:")
    print("-------------")
    print(f"Final Summary: {results['final_summary']}")

if __name__ == "__main__":
    main()