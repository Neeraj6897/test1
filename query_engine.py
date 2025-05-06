import logging
from typing import Any, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryEngine:
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model="deepseek-r1-distill-llama-70b"
        )

    def create_rag_chain(self, retriever):
        """Create RAG chain for querying."""
        message = """
        Answer this question using the provided context only.
        
        {question}
        
        Context:
        {context}
        """
        prompt = ChatPromptTemplate.from_messages([("human", message)])
        return {"context": retriever, "question": RunnablePassthrough()} | prompt | self.llm

    def process_response(self, response: Any) -> dict:
        """Process and summarize response."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        chunks = splitter.split_text(response_text)
        
        summaries = []
        for chunk in chunks:
            chunk_prompt = f"Summarize this concisely:\n\n{chunk}"
            summary = self.llm.invoke(chunk_prompt)
            summaries.append(summary.content)
        
        final_prompt = (
            "Based on these summaries, provide one concise summary "
            "with key fix details and resolution. Also provide details about the github issue number and title.:\n\n" + "\n".join(summaries)
        )
        final_summary = self.llm.invoke(final_prompt)
        
        return {
            "raw_response": response_text,
            "summaries": summaries,
            "final_summary": final_summary.content
        }