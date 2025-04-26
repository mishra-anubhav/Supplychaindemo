# ingestion/ingest_unstructured.py

import os
from typing import Literal
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone, ServerlessSpec
import pinecone as _pinecone_module
import pinecone.data.index as _pinecone_index_module
# Monkey-patch pinecone.Index to satisfy LangChain v2 type check (v3 reorg workaround)
_pinecone_module.Index = _pinecone_index_module.Index
from dotenv import load_dotenv

load_dotenv()

# 📁 Paths and config
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/uploaded_docs"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = "test"
NAMESPACE = "supply-unstructured"

# 📦 Pinecone v3 setup
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

def embed_and_store_unstructured_files(file_paths: list[str] = None):
    """
    Embed and store text/PDF files into Pinecone.
    If file_paths is provided, ingest only those files; otherwise ingest all files
    found in the UPLOAD_DIR.
    """
    all_docs = []

    # Determine files to ingest
    if file_paths:
        # Use provided file paths (relative or absolute)
        paths = [os.path.abspath(p) for p in file_paths]
    else:
        # Ingest all files in upload directory
        print(f"📁 Upload directory: {UPLOAD_DIR}")
        entries = os.listdir(UPLOAD_DIR)
        print(f"📄 Files found: {entries}")
        paths = [os.path.join(UPLOAD_DIR, fn) for fn in entries]

    for file_path in paths:
        filename = os.path.basename(file_path)
        ext = filename.split(".")[-1].lower()

        print(f"📌 Processing file: {filename} ({ext})")

        if ext == "txt":
            loader = TextLoader(file_path)
        elif ext == "pdf":
            loader = PyPDFLoader(file_path)
        else:
            print(f"❌ Skipping unsupported file type: {filename}")
            continue

        docs = loader.load()
        all_docs.extend(docs)

    if not all_docs:
        print("⚠️ No valid documents found to embed.")
        return

    # Chunking
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    print(f"✂️ Chunks created: {len(chunks)}")

    # Embedding
    embedder = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    # Initialize Pinecone client and index (v3)
    client = Pinecone(api_key=PINECONE_API_KEY)
    index = client.Index(PINECONE_INDEX_NAME)
    # Create vector store and upsert documents
    store = PineconeStore(
        index=index,
        embedding=embedder,
        text_key="text",
        namespace=NAMESPACE,
    )
    # Prepare texts and metadatas
    texts = [doc.page_content for doc in chunks]
    metadatas = [getattr(doc, 'metadata', {}) for doc in chunks]
    store.add_texts(
        texts,
        metadatas=metadatas,
    )

    print(f"✅ Embedded and stored {len(chunks)} chunks into Pinecone ({NAMESPACE})")
