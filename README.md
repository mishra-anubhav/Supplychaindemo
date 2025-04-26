 # Predictive Supply Chain Assistant (Final-Supply-GenAI)

    Final-Supply-GenAI is a Streamlit-powered chat application that integrates structured SQL analytics, retrieval-augmented generation (RAG) over unstructured documents, and large language models (LLMs) to provide actionable insights for supply chain management.

    ---

## Table of Contents
    1. [End-to-End Flow](#end-to-end-flow)
    2. [Key Components](#key-components)
       - [Streamlit UI](#streamlit-ui-appmainpy)
       - [Query Router](#query-router-appquery_routerpy)
       - [Tools & Pipelines](#tools--pipelines)
       - [Data Ingestion](#data-ingestion-appingestion)
       - [Models & Memory](#models--memory)
    3. [Directory Structure](#directory-structure)
    4. [Setup & Installation](#setup--installation)
    5. [Running the App](#running-the-app)
    6. [Usage Examples](#usage-examples)
       - [Demand Forecasting](#demand-forecasting)
       - [Document Summarization (RAG)](#document-summarization-rag)
       - [General LLM Queries](#general-llm-queries)
    7. [Extending & Customization](#extending--customization)
    8. [Troubleshooting](#troubleshooting)
    9. [Contributing](#contributing)
    10. [License & Acknowledgements](#license--acknowledgements)

## End-to-End Flow

    1. **User Interaction**
       User enters a query or uploads data via the Streamlit interface (`app/main.py`).

    2. **Query Routing**
       `app/query_router.py` inspects the message and chooses one of three pipelines:
       - **SQL Pipeline** for structured data queries.
       - **RAG Pipeline** for unstructured document queries.
       - **LLM Fallback** for general or unsupported queries.

    3. **Pipeline Execution**
       - **SQL Pipeline** (`app/tools/sql_tool.py`): Generates SQL via the LLM, executes against the SQLite database, and formats the results.
       - **RAG Pipeline** (`app/tools/pinecone_tool.py`): Retrieves top-k relevant chunks from Pinecone, then uses the LLM to generate the final
    answer.
       - **LLM Fallback** (`app/model/openai_llms.py` or `app/model/local_llm.py`): Sends the query directly to the LLM for open-ended responses.

    4. **Response Rendering**
       The response is rendered back in the Streamlit chat interface, preserving session history.

## Key Components

### Streamlit UI (app/main.py)
    - Sets up the page, uploads, and chat input.
    - Maintains `st.session_state.history` to persist the conversation.

    Example:
    ```python
    import streamlit as st
    from query_router import QueryRouter

    router = QueryRouter()
    if "history" not in st.session_state:
        st.session_state.history = []

    prompt = st.text_input("Enter your question:")
    if prompt:
        response = router.route(prompt, st.session_state.history)
        st.session_state.history.append((prompt, response))

    for user, bot in st.session_state.history:
        st.chat_message("user").write(user)
        st.chat_message("assistant").write(bot)
    ```

### Query Router (app/query_router.py)
    - Determines which pipeline to invoke based on keywords or patterns.
    - Centralizes error handling and tool initialization.

    Sample logic:
    ```python
    class QueryRouter:
        def __init__(self):
            self.sql_tool = SQLTool()
            self.pinecone_tool = PineconeTool()
            self.llm_tool = OpenAILLM()

        def route(self, query, history):
            if query.lower().startswith(('what','show','forecast')):
                return self.sql_tool.run(query)
            elif query.lower().startswith(('summarize','extract','review')):
                return self.pinecone_tool.run(query)
            else:
                return self.llm_tool.chat(query, history)
    ```

### Tools & Pipelines (app/tools/)
    - **SQL Tool** (`sql_tool.py`): LLM-driven SQL generation, execution against SQLite, and result formatting.
    - **Pinecone Tool** (`pinecone_tool.py`): RAG chain that handles embedding lookup and LLM summarization.

### Data Ingestion (app/ingestion/)
    - **ingest_sql.py**: Loads `data/demand_data.csv` into `database/supplychain.db`.
    - **ingest_unstructured.py**: Reads files from `data/uploaded_docs/`, splits text, generates embeddings, and upserts them into Pinecone.

    ```bash
    python app/ingestion/ingest_sql.py
    python app/ingestion/ingest_unstructured.py
    ```

### Models & Memory
    - **OpenAI LLM Wrapper** (`app/model/openai_llms.py`): Interfaces with OpenAI ChatCompletion and Embeddings.
    - **Local LLM Stub** (`app/model/local_llm.py`): Placeholder for on-premise models.
    - **Conversation Memory** (`app/memory/memory.py`): Infrastructure for future session memory.

## Directory Structure
    ```
    Final-Supply-GenAI/
    ├── app/
    │   ├── main.py
    │   ├── query_router.py
    │   ├── ingestion/
    │   ├── tools/
    │   ├── model/
    │   └── memory/
    ├── data/
    ├── database/
    ├── requirements.txt
    └── README.md
    ```

## Setup & Installation

    1. Clone & Virtual Env:
       ```bash
    git clone <repo_url>
    cd Final-Supply-GenAI
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

    2. Create `.env` with:
    ```dotenv
    OPENAI_API_KEY=...
    PINECONE_API_KEY=...
    PINECONE_ENV=...
    PINECONE_INDEX=...
    ```

    ## Running the App

    - Local: `streamlit run app/main.py`
    - Docker:
      ```bash
    docker build -t supplychain-assistant .
    docker run -d --env-file .env -p 8501:8501 supplychain-assistant
    ```

    ## Usage Examples

    ### Demand Forecasting
    **Query:** "What was the total demand for SKU-101 in Q1 2023?"
    **Flow:** SQL Tool → SQLite → Result table displayed.

    ### Document Summarization (RAG)
    **Query:** "Summarize key points from the supply memos."
    **Flow:** Retrieve embeddings → LLM summary.

    ### General LLM Queries
    **Query:** "Write an email to procurement about a delay."
    **Flow:** Direct LLM chat.

    ## Extending & Customization
    - Add tools in `app/tools/` and update `query_router.py`.
    - Swap LLM or vector store in `app/model/`.
    - Enhance UI components in `app/main.py`.

    ## Troubleshooting
    - **No response:** Check your API keys and internet connection.
    - **DB errors:** Ensure `supplychain.db` exists; run ingestion.
    - **Pinecone issues:** Verify your environment variables and index name.

    ## Contributing
    Contributions welcome! Please fork and issue a pull request.

    ## License & Acknowledgements
    Licensed under MIT.
    Built with [Streamlit], [LangChain], [OpenAI], and [Pinecone].
    EOF

    command.stdout (code: 127, duration: 0s)

    bash: cat: command not found


    codex
    Below is a completely rewritten README.md that walks through the entire code‐flow step-by-step, shows you where everything lives, and gives
    concrete examples of how each part behaves. You can copy & paste this into your existing README.md (or replace it entirely).

    ---------------------------------------------------------------------------------------------------------------------------------------------------
    ---

    # Predictive Supply Chain Assistant (Final-Supply-GenAI)

    Final-Supply-GenAI is a Streamlit-powered chat application that fuses:

      • Structured SQL analytics on your demand data
      • Retrieval-Augmented Generation (RAG) over unstructured memos and documents
      • Large Language Models for fallback and unbounded queries

    to give you real-time, actionable supply-chain insights in a conversational UI.

    ---------------------------------------------------------------------------------------------------------------------------------------------------
    ---

    ## Table of Contents

        1. [End-to-End Flow](#end-to-end-flow)
        2. [Key Components & Code Walkthrough](#key-components-and-code-walkthrough)
            * [Streamlit UI (app/main.py)](#1-streamlit-ui-appmainpy)

            * [Query Routing (app/query_router.py)](#2-query-routing-appquery_routerpy)

            * [SQL Tool (app/tools/sql_tool.py)](#3-sql-tool-apptoolssql_toolpy)

            * [Pinecone RAG Tool (app/tools/pinecone_tool.py)](#4-pinecone-rag-tool-apptoolspinecone_toolpy)

            * [Data Ingestion (app/ingestion/)](#5-data-ingestion-appingestion)

            * [Models & Memory (app/model, app/memory)](#6-models-and-memory)
        3. [Directory Structure](#directory-structure)
        4. [Setup & Installation](#setup-and-installation)
        5. [Running the App](#running-the-app)
        6. [Usage Examples](#usage-examples)
        7. [Extending & Customization](#extending-and-customization)
        8. [Troubleshooting](#troubleshooting)
        9. [Contributing](#contributing)
        10. [License & Acknowledgements](#license-and-acknowledgements)

    ---------------------------------------------------------------------------------------------------------------------------------------------------
    ---

    ## End-to-End Flow

        1. **User Interaction**


            * Via `app/main.py`, users upload CSVs or docs, then type questions in the chat box.
        2. **Query Routing**


            * `app/query_router.py` inspects each prompt and dispatches it to one of:

                * **SQL pipeline** (for structured/demand queries)


                * **RAG pipeline** (for unstructured doc queries)


                * **LLM fallback** (for anything else)
        3. **Pipeline Execution**


            * **SQL Pipeline** (`sql_tool.py`):

                * The LLM drafts an SQL statement


                * Executed against `database/supplychain.db`


                * Results are formatted back to text or a table

            * **RAG Pipeline** (`pinecone_tool.py`):

                * Semantic search in Pinecone over your uploaded docs


                * Top-k chunks + prompt → LLM generates answer

            * **LLM Fallback** (`openai_llms.py` or `local_llm.py`):

                * Direct call to ChatGPT (or a local model) for general Q&A
        4. **Response Rendering**


            * Streamlit displays the assistant’s reply, preserving history in `st.session_state.history`.

    ---------------------------------------------------------------------------------------------------------------------------------------------------
    ---

    ## Key Components and Code Walkthrough

    ### 1. Streamlit UI (app/main.py)

    Sets up the page, file-upload widgets, chat input & history.

        import streamlit as st
        from query_router import QueryRouter

        st.set_page_config(page_title="Supply-Chain Assistant")
        router = QueryRouter()

        if "history" not in st.session_state:
            st.session_state.history = []

        # Sidebar: CSV / Document upload logic goes here...

        prompt = st.text_input("Ask your supply-chain assistant...")
        if prompt:
            answer = router.route(prompt, st.session_state.history)
            st.session_state.history.append((prompt, answer))

        for user_msg, bot_msg in st.session_state.history:
            st.chat_message("user").write(user_msg)
            st.chat_message("assistant").write(bot_msg)

    ### 2. Query Routing (app/query_router.py)

    Simple rules (and pluggable overrides) decide which tool to invoke.

        class QueryRouter:
            def __init__(self):
                self.sql_tool = SQLTool()
                self.rag_tool = PineconeTool()
                self.llm_tool = OpenAILLM()

            def route(self, query: str, history: List[Tuple[str,str]]) -> str:
                text = query.lower()
                if any(text.startswith(k) for k in ("show","list","forecast","what")):
                    return self.sql_tool.run(query)
                if any(text.startswith(k) for k in ("summarize","review","extract")):
                    return self.rag_tool.run(query)
                return self.llm_tool.chat(query, history)

    ### 3. SQL Tool (app/tools/sql_tool.py)

        * Uses LangChain’s SQL utilities
        * LLM drafts SQL → run on SQLite → format back

        from langchain_experimental.sql import SQLDatabaseChain

        class SQLTool:
            def __init__(self):
                self.chain = SQLDatabaseChain.from_llm(llm=OpenAI(), db_uri="sqlite:///database/supplychain.db")

            def run(self, question: str) -> str:
                result = self.chain.run(question)
                return result

    ### 4. Pinecone RAG Tool (app/tools/pinecone_tool.py)

        * Splits & embeds documents
        * Upserts to Pinecone
        * `RetrievalQA` chain to answer

        from langchain.chains import RetrievalQA
        from langchain.vectorstores import Pinecone

        class PineconeTool:
            def __init__(self, index_name="test"):
                self.store = Pinecone.from_existing_index(index_name, OpenAIEmbeddings())
                self.chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=self.store.as_retriever())

            def run(self, question: str) -> str:
                return self.chain.run(question)

    ### 5. Data Ingestion (app/ingestion/)

        * **ingest_sql.py**
          Loads `data/demand_data.csv` into your SQLite DB.      python app/ingestion/ingest_sql.py
        * **ingest_unstructured.py**
          Reads everything in `data/uploaded_docs/`, chunks it, embeds and upserts to Pinecone.      python app/ingestion/ingest_unstructured.py

    ### 6. Models & Memory

        * **app/model/openai_llms.py**: ChatCompletion & Embeddings wrapper
        * **app/model/local_llm.py**: (Placeholder) for on-prem models
        * **app/memory/memory.py**: (Future) conversation memory store

    ---------------------------------------------------------------------------------------------------------------------------------------------------
    ---

    ## Directory Structure

        Final-Supply-GenAI/
        ├── app/
        │   ├── main.py
        │   ├── query_router.py
        │   ├── ingestion/
        │   ├── tools/
        │   ├── model/
        │   └── memory/
        ├── data/
        │   ├── demand_data.csv
        │   └── uploaded_docs/
        ├── database/
        │   └── supplychain.db
        ├── requirements.txt
        └── README.md     ← (this file)

    ---------------------------------------------------------------------------------------------------------------------------------------------------
    ---

    ## Setup and Installation

        git clone <repo_url>
        cd Final-Supply-GenAI
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt

    Create a .env in the project root:

        OPENAI_API_KEY=your_openai_key
        PINECONE_API_KEY=your_pinecone_key
        PINECONE_ENV=your_pinecone_region
        PINECONE_INDEX=your_index_name

    ---------------------------------------------------------------------------------------------------------------------------------------------------
    ---

    ## Running the App

    Locally:

        streamlit run app/main.py

    With Docker:

        docker build -t supplychain-assistant .
        docker run -d --env-file .env -p 8501:8501 supplychain-assistant

    ---------------------------------------------------------------------------------------------------------------------------------------------------
    ---

    ## Usage Examples

        1. **Demand Forecasting**


            * **Query:** “What was the total demand for SKU-101 in Q1 2023?”

            * **Flow:** SQL Tool → SQLite → Table output in chat
        2. **Document Summarization (RAG)**


            * **Query:** “Summarize the key points from the supply memos.”

            * **Flow:** Pinecone retrieves relevant chunks → LLM generates summary
        3. **General LLM Queries**


            * **Query:** “Write an email to procurement about a shipment delay.”

            * **Flow:** Direct LLM chat pipeline → Draft email returned

    ---------------------------------------------------------------------------------------------------------------------------------------------------
    ---

    ## Extending and Customization

        * Drop new tools into `app/tools/` and register them in `query_router.py`
        * Swap out your LLM or Vector Store in `app/model/`
        * Enhance the UI in `app/main.py` or add dashboard pages

    ---------------------------------------------------------------------------------------------------------------------------------------------------
    ---

    ## Troubleshooting

        * **No response?** Check your API keys and network.
        * **DB errors?** Ensure `supplychain.db` exists (run the SQL ingestion script).
        * **Pinecone failures?** Verify `.env` vars and your index name.

    ---------------------------------------------------------------------------------------------------------------------------------------------------
    ---

    ## Contributing

    Pull requests, issues and feature requests are welcome! Feel free to fork and open a PR.

    ---------------------------------------------------------------------------------------------------------------------------------------------------
    ---

    ## License and Acknowledgements

    Licensed under MIT.