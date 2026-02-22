#  RAG Project — Retrieval-Augmented Generation with LangChain + Gemini + Pinecone

## Juan Pablo Contreras Parra

---

A fully-functional Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Google Gemini**, and **Pinecone**. The system ingests documents from the web, stores their vector embeddings in Pinecone, and answers natural-language questions using only the retrieved context — no hallucinations, no made-up facts.

---

## Architecture & Components

The pipeline is made up of six components that work together in sequence.

The **Document Loader** (`WebBaseLoader`) fetches and parses HTML pages from any URL you provide. The **Text Splitter** (`RecursiveCharacterTextSplitter`) breaks those pages into overlapping chunks of 1000 characters with a 200-character overlap, so no information is lost at chunk boundaries. The **Embedding Model** (`models/embedding-001` via Google Gemini) converts each chunk into a 768-dimensional vector that captures its semantic meaning. Those vectors are then upserted into a **Pinecone** serverless index where they can be searched by cosine similarity.

At query time, the user's question is embedded using the same Gemini model and the **Pinecone Retriever** finds the top 4 most semantically similar chunks. Those chunks are injected as context into a prompt sent to **Gemini 1.5 Flash**, which generates a grounded answer using only the retrieved content. The whole chain is wired together using LangChain's `create_retrieval_chain`.

---

##  Repository Structure

```
rag-project/
├── src/
│   ├── rag.py          # Core pipeline: ingest + chain building
│   └── query_cli.py    # Interactive CLI for querying
├── .env.example        # Environment variable template 
├── .env                # Your real API keys 
├── .gitignore          # Excludes .env and venv/
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

##  Installation

### Prerequisites

- Python 3.10-3.11
- A free [Google Gemini API key](https://aistudio.google.com/apikey) 
- A free [Pinecone API key](https://app.pinecone.io/) 

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/rag-project.git
cd rag-project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate.bat       # Windows CMD
venv\Scripts\Activate.ps1       # Windows PowerShell

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and fill in your GOOGLE_API_KEY and PINECONE_API_KEY
```

---

##  Environment Variables

Your `.env` file should contain the following:

```
GOOGLE_API_KEY=AIza...
PINECONE_API_KEY=pcsk_...
USER_AGENT=rag-project/1.0
```

---

##  Running the Project

### Step 1 — Ingest documents

Open `src/rag.py` and update the `URLS` list with any web pages you want the system to learn from. Then run:

```bash
cd src
python rag.py
```

The script will fetch the pages, split them into chunks, embed them with Gemini, create a Pinecone index (if one doesn't exist yet), and upsert all the vectors. It will then automatically answer three sample questions to confirm everything is working.

Expected output:

```
📥 Loading 1 document(s)...
   Loaded 1 raw document(s).
   Split into 66 chunk(s).
Creating Pinecone index 'rag-demo-index' ...
Index created.
⬆️  Upserting chunks into Pinecone ...
✅ Ingestion complete.

============================================================
❓ Question: What is Task Decomposition?
💬 Answer  : Task decomposition is the process of breaking down a complex task into smaller, manageable subgoals...
📚 Sources : ['https://lilianweng.github.io/posts/2023-06-23-agent/']
------------------------------------------------------------
```


### Step 2 — Query interactively

After ingestion is complete, use the interactive CLI to ask questions in real time:

```bash
python query_cli.py
```

Example session:

```
🤖 RAG Interactive Query CLI
==================================================
Type your question and press Enter. Type 'exit' to quit.

❓ Your question: How does ReAct work?

💬 Answer:
ReAct combines reasoning and acting by prompting the model to generate
both verbal reasoning traces and actions in an interleaved fashion...

📚 Sources: ['https://lilianweng.github.io/posts/2023-06-23-agent/']

--------------------------------------------------
❓ Your question: exit
Goodbye!
```

---

## ❓ What Can It Answer?

The system can only answer questions about the documents it has ingested. By default it indexes one article — Lilian Weng's post on LLM-powered autonomous agents — so out of the box you can ask things like:

- What is Task Decomposition?
- What are the types of memory in LLM agents?
- How does ReAct work?
- What is chain-of-thought prompting?
- What tools can agents use?

If you ask about anything outside the ingested content, it will respond with "I don't know based on the provided documents." To expand what it knows, simply add more URLs to the `URLS` list in `rag.py` and re-run the ingestion.

---

## Evidence

![Question](/images/image.png)

![Question2](/images/image-1.png)

![Question3](/images/image-2.png)

![CustomQuestion](/images/image-3.png)

##  Configuration

All tuneable parameters are at the top of `src/rag.py`:

| Parameter | Default | Description |
|---|---|---|
| `INDEX_NAME` | `rag-demo-index` | Pinecone index name |
| `EMBEDDING_MODEL` | `models/embedding-001` | Gemini embedding model (768-dim output) |
| `LLM_MODEL` | `gemini-1.5-flash` | Gemini chat model |
| `chunk_size` | `1000` | Characters per chunk |
| `chunk_overlap` | `200` | Overlap between chunks |
| `k` (retriever) | `4` | Number of chunks retrieved per query |

---

## How RAG Works

Traditional LLMs answer from their training data alone and hallucinate when asked about documents they haven't seen. RAG solves this by separating knowledge storage from generation. During indexing, your documents are converted into vector embeddings and stored in Pinecone. At query time, the user's question is embedded and used to search for the most relevant chunks. Those chunks are injected into the LLM's prompt as context, and the model generates an answer grounded entirely in the retrieved material.

---

##  References

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain Pinecone Integration](https://python.langchain.com/docs/integrations/vectorstores/pinecone)
- [Google Gemini API Docs](https://ai.google.dev/gemini-api/docs)
- [Pinecone Quickstart](https://docs.pinecone.io/guides/getting-started/quickstart)

