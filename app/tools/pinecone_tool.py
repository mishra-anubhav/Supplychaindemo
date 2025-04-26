# tools/pinecone_tool.py

from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain.memory import ConversationBufferMemory


def query_pinecone_documents(
    query: str,
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
    memory: ConversationBufferMemory = None,
    index_name: str = "test"
) -> str:
    """
    Runs a RAG pipeline using a Pinecone-backed retriever and a selected LLM.
    Optionally supports conversation memory for multi-turn interactions.

    Args:
        query (str): The user's question.
        retriever (BaseRetriever): The Pinecone retriever.
        llm (BaseLanguageModel): The language model (OpenAI / LLaMA).
        memory (ConversationBufferMemory, optional): LangChain memory.
        k (int): Number of documents to retrieve. Default is 3.

    Returns:
        str: Assistant response
    """
    try:
        # Initialize the LangChain RAG pipeline
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )

        # Use invoke instead of deprecated run; extract answer from output dict
        output = qa_chain.invoke({"question": query})
        if isinstance(output, dict):
            # Return the 'answer' field if available
            return output.get("answer", output)
        return output

    except Exception as e:
        return f"❌ Error in Pinecone RAG pipeline: {e}"
