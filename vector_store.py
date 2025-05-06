import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./github_issues_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None

    def prepare_documents(self, issues: List[Dict[str, Any]]) -> List[Document]:
        """Convert issues to documents for embedding."""
        documents = []
        for issue in issues:
            content = (
                f"Title: {issue['title']}\n"
                f"Body: {issue['body']}\n"
                f"Fix Info: {' '.join(issue['fix_info'])}\n"
                f"Comments: {' '.join(issue['comments'])}"
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={"id": issue["id"]}
                )
            )
        return documents

    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create and persist vector store."""
        logger.info("Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        #self.vectorstore.persist()
        logger.info(f"Vector store created and persisted to {self.persist_directory}")

    def load_vectorstore(self) -> Optional[Chroma]:
        """Load existing vector store."""
        if Path(self.persist_directory).exists():
            logger.info("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return self.vectorstore
        logger.warning("No existing vector store found")
        return None

    def get_retriever(self, k: int = 1):
        """Get retriever for querying."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})