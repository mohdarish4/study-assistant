# Study Assistant: RAG Chatbot for Your Documents

One line: Ask questions about your PDFs and get answers with page-level citations.

## Demo
(screenshot ya GIF yahan)

## Features
- PDF ingestion with chunking and embeddings
- Answers only from your documents, with source and page number
- Streaming responses
- Says "not found" instead of guessing

## Tech Stack
LangChain, ChromaDB, Gemini embeddings, Groq LLM, Python, uv

## How It Works
PDF → chunks → embeddings → ChromaDB → retrieve top 4 → LLM → answer + sources

## Setup
1. `uv sync`
2. Copy `.env.example` to `.env` and add your keys
3. Put PDFs in `data/`
4. `uv run ingest.py`
5. `uv run ask.py`

## Evaluation
(15-20 test questions, kitne sahi aaye)

## Roadmap
- [x] Stage 1: RAG pipeline
- [ ] Stage 2: Streamlit UI and chat memory
- [ ] Stage 3: LangGraph agent with web search
