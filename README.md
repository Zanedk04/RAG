# RAG

A small collection of Retrieval-Augmented Generation (RAG) experiments built with
[LangChain](https://python.langchain.com/), [Ollama](https://ollama.com/), and
[Chroma](https://www.trychroma.com/). Documents are embedded locally with Ollama's
`nomic-embed-text` model, stored in a persistent Chroma vector store, and queried with
`llama3.1:8b` for answer generation.

Sample documents (`docs/`) are plain-text company overviews for Google, Microsoft,
Nvidia, SpaceX, and Tesla, used to test retrieval and generation.

## Scripts

| Script | Purpose |
|---|---|
| [1_ingestion_pipeline.py](1_ingestion_pipeline.py) | Loads `.txt` files from `docs/`, splits them into chunks, embeds them, and persists them to a local Chroma vector store at `db/chroma_db`. Run this first. |
| [2_retrieval_pipeline.py](2_retrieval_pipeline.py) | Loads the vector store and runs a similarity search for a sample query, printing the retrieved chunks. |
| [3_answer_generation.py](3_answer_generation.py) | Retrieves relevant chunks for a query and passes them to an LLM to generate a grounded answer. |
| [4_history_aware_generation.py](4_history_aware_generation.py) | Interactive chat loop that rewrites follow-up questions using conversation history before retrieving, then answers with context. |
| [6_semantic_chunking.py](6_semantic_chunking.py) | Demonstrates splitting text into chunks based on semantic similarity (via `SemanticChunker`) rather than fixed size. |
| [7_agent_chunking.py](7_agent_chunking.py) | Demonstrates using an LLM itself to decide where to split text into chunks ("agentic chunking"). |

## Setup

1. Install [Ollama](https://ollama.com/) and pull the models used here:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.1:8b
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. (Optional) create a `.env` file for any environment variables the scripts load via `python-dotenv`.

## Usage

Build the vector store first, then run any of the retrieval/generation scripts:

```bash
python 1_ingestion_pipeline.py
python 2_retrieval_pipeline.py
python 3_answer_generation.py
python 4_history_aware_generation.py
```

The chunking demos (`6_semantic_chunking.py`, `7_agent_chunking.py`) are standalone and
don't require the vector store to be built first.

Note: `db/` (the generated vector store) and `venv/` are gitignored — regenerate `db/`
by running the ingestion pipeline.
