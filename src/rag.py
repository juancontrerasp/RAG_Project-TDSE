"""
RAG (Retrieval-Augmented Generation) Implementation
Using LangChain + Google Gemini + Pinecone
"""

import os
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Pinecone
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ──────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────
GOOGLE_API_KEY   = os.environ["GOOGLE_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = "rag-demo-index"

# Gemini embedding model outputs 768-dim vectors
EMBEDDING_MODEL  = "models/gemini-embedding-001"

# Free Gemini chat model
LLM_MODEL        = "gemini-2.5-flash"


# ──────────────────────────────────────────────
# 2. Initialize clients
# ──────────────────────────────────────────────
def get_embeddings():
    """Return a Gemini embeddings model."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def get_llm():
    """Return a Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )


def get_or_create_pinecone_index():
    """
    Connect to Pinecone and create the index if it does not exist yet.
    Returns the index name ready for use with PineconeVectorStore.

    IMPORTANT: dimension=768 matches Gemini's embedding-001 output size.
    If you previously created an index with dimension=1536 (OpenAI),
    delete it in the Pinecone dashboard first, then re-run.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{INDEX_NAME}' ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,   
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Using existing Pinecone index '{INDEX_NAME}'.")

    return INDEX_NAME


# ──────────────────────────────────────────────
# 3. Ingest documents
# ──────────────────────────────────────────────
def ingest_documents(urls: list[str]) -> PineconeVectorStore:
    """
    Load web pages, split them into chunks, embed them with Gemini,
    and upsert them into Pinecone.

    Args:
        urls: List of web page URLs to ingest.

    Returns:
        A PineconeVectorStore ready for retrieval.
    """
    print(f"\n📥 Loading {len(urls)} document(s)...")
    loader = WebBaseLoader(urls)
    docs = loader.load()
    print(f"   Loaded {len(docs)} raw document(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"   Split into {len(chunks)} chunk(s).")

    embeddings = get_embeddings()
    index_name = get_or_create_pinecone_index()

    print("⬆️  Upserting chunks into Pinecone ...")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    print("✅ Ingestion complete.\n")
    return vector_store


# ──────────────────────────────────────────────
# 4. Build the RAG chain
# ──────────────────────────────────────────────
def build_rag_chain(vector_store: PineconeVectorStore):
    """
    Wire together the retriever, prompt, and LLM into a RAG chain.

    The chain:
      1. Converts the user question into a Gemini embedding.
      2. Retrieves the top-k most similar chunks from Pinecone.
      3. Stuffs those chunks into the prompt context.
      4. Asks Gemini to answer using only that context.

    Args:
        vector_store: An initialised PineconeVectorStore.

    Returns:
        A LangChain retrieval chain callable.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    system_prompt = (
        "You are a helpful assistant. Use ONLY the retrieved context below "
        "to answer the user's question. If the answer is not contained in "
        "the context, say 'I don't know based on the provided documents'.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm       = get_llm()
    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    return rag_chain


# ──────────────────────────────────────────────
# 5. Connect to an existing Pinecone index
# ──────────────────────────────────────────────
def load_existing_vector_store() -> PineconeVectorStore:
    """Load a PineconeVectorStore from an already-populated index."""
    embeddings = get_embeddings()
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )


# ──────────────────────────────────────────────
# 6. Main – demo flow
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # ---- Step A: choose some URLs to ingest ----
    URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
    ]

    # ---- Step B: ingest (comment out after first run to avoid re-ingesting) ----
    vector_store = ingest_documents(URLS)

    # ---- Step C: build chain and query ----
    chain = build_rag_chain(vector_store)

    questions = [
        "What is Task Decomposition?",
        "What are the main types of memory in LLM agents?",
        "How does ReAct work?",
    ]

    print("=" * 60)
    for question in questions:
        print(f"\n❓ Question: {question}")
        result = chain.invoke({"input": question})
        print(f"💬 Answer  : {result['answer']}")
        print(f"📚 Sources : {[d.metadata.get('source', 'N/A') for d in result['context']]}")
        print("-" * 60)