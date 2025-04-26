"""
query_router.py

Routes user queries to either SQL or Pinecone RAG, using a simple keyword heuristic.
"""
# Suppress deprecation warnings
import warnings
# Ignore standard DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Ignore LangChain memory deprecation notice (migration guide)
warnings.filterwarnings("ignore", message=r".*Please see the migration guide.*")
import os
import re
from dotenv import load_dotenv
from tools.sql_tool import query_sql_database
from tools.pinecone_tool import query_pinecone_documents

# LangChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# Environment
load_dotenv()
#openaiApiKey = os.getenv("OPENAI_API_KEY")
import openai
openai.api_key = st.secrets["openai"]["api_key"]
openai.api_key = openaiApiKey
pinecone_api_key = st.secrets["pinecone"]["api_key"]
pinecone_environment = st.secrets["pinecone"]["environment"]

# Initialize Pinecone with the retrieved API key and environment
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

# Memory setup
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer",
)

# Embedding + Chat model
embeddingModel = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=openaiApiKey
)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openaiApiKey
)

import pinecone
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from pydantic import ConfigDict, Extra
from typing import Any

class PineconeV3Retriever(BaseRetriever):
    """Custom Pinecone v3 retriever for LangChain ConversationalRetrievalChain."""
    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True,
        extra=Extra.allow,
    )

    index: Any
    embedder: Any
    k: int = 3
    namespace: str = None
    text_key: str = "text"
    
    def get_relevant_documents(self, query: str) -> list[Document]:
        # Embed the query
        query_embedding = self.embedder.embed_query(query)
        # Query Pinecone
        res = self.index.query(
            vector=query_embedding,
            top_k=self.k,
            include_metadata=True,
            namespace=self.namespace,
        )
        docs: list[Document] = []
        for match in res.get("matches", []):
            metadata = match.get("metadata", {}) or {}
            text = metadata.pop(self.text_key, None)
            if text is None:
                continue
            # Filter metadata to primitive types
            filtered = {k: v for k, v in metadata.items() if isinstance(v, (str, bool, int, float))}
            docs.append(Document(page_content=text, metadata=filtered))
        return docs

def _run_structured_query(userQuery: str) -> str:
    """Run the SQL pipeline on the user query."""
    return query_sql_database(userQuery)

def _run_unstructured_query(userQuery: str) -> str:
    """Run the Pinecone RAG pipeline or general LLM fallback for the user query."""
    # Lazy initialize Pinecone client and retriever (v3)
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    client = pinecone.Pinecone(api_key=pinecone_api_key)
    index = client.Index(pineconeIndexName)
    retriever = PineconeV3Retriever(
        index=index,
        embedder=embeddingModel,
        k=3,
        namespace="supply-unstructured",
        text_key="text",
    )
    # Attempt retrieval; if no relevant docs, fallback to general LLM
    try:
        docs = retriever.get_relevant_documents(userQuery)
    except Exception:
        docs = []
    if not docs:
        # General LLM fallback for open-ended queries
        try:
            resp = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": userQuery},
                ],
                temperature=0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"❌ Error in general LLM fallback: {e}"
    # Otherwise, route through RAG
    return query_pinecone_documents(
        query=userQuery,
        llm=llm,
        retriever=retriever,
        memory=memory,
        index_name=pineconeIndexName,
    )

def handle_user_query(userQuery: str, hybrid: bool = False) -> str:
    """
    Routes user queries to SQL, RAG, or a hybrid merge of both.
    If hybrid=True, runs both pipelines and synthesizes a combined answer.
    """
    # Hybrid mode: run both pipelines and synthesize
    if hybrid:
        structured_ans = _run_structured_query(userQuery)
        unstructured_ans = _run_unstructured_query(userQuery)
        try:
            # Synthesize using LLM
            messages = [
                {"role": "system", "content": (
                    "You are a helpful assistant that synthesizes structured SQL results "
                    "and unstructured document insights into a single, coherent response."
                )},
                {"role": "user", "content": (
                    f"Structured result:\n{structured_ans}\n\n"
                    f"Unstructured result:\n{unstructured_ans}\n\n"
                    "Please combine these into one comprehensive answer."
                )},
            ]
            resp = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"❌ Hybrid synthesis error: {e}"
    # Simple rule-based decision for non-hybrid queries
    if re.search(r"sku|demand|stock|order|quantity", userQuery.lower()):
        return _run_structured_query(userQuery)
    else:
        return _run_unstructured_query(userQuery)
