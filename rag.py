"""Core RAG logic used by the Streamlit app: indexing, retrieval, and answering."""
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("data")
DB_DIR = "chroma_db"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # "groq" or "gemini"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # set GROQ_MODEL in .env to override
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # set GEMINI_MODEL in .env to override
BATCH_SIZE = 50  # small batches keep us inside free-tier rate limits
HISTORY_TURNS = 6  # how many recent messages the model remembers
NOT_FOUND = "I could not find this in your documents."


# ---------- setup ----------

def get_db() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def get_llm():
    """Chat model. Set LLM_PROVIDER=gemini in .env if Groq is blocked on your network."""
    if LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    return ChatGroq(model=GROQ_MODEL, temperature=0)


# ---------- indexing ----------

def indexed_sources(db: Chroma) -> set[str]:
    """Source paths of every PDF already stored in the database."""
    data = db.get(include=["metadatas"])
    return {m["source"] for m in data["metadatas"] if m and "source" in m}


def pending_pdfs(db: Chroma) -> list[Path]:
    """PDFs in the data folder that are not indexed yet."""
    done = indexed_sources(db)
    return [p for p in sorted(DATA_DIR.glob("*.pdf")) if str(p) not in done]


def ingest_pdf(db: Chroma, path: Path) -> int:
    """Split one PDF into chunks, embed them, and store them. Returns the chunk count."""
    pages = PyPDFLoader(str(path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    for i in range(0, len(chunks), BATCH_SIZE):
        db.add_documents(chunks[i : i + BATCH_SIZE])
        if i + BATCH_SIZE < len(chunks):
            time.sleep(1)
    return len(chunks)


# ---------- question answering ----------

condense_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest question as a standalone question, using the chat "
            "history to resolve words like 'it' or 'that'. Do NOT answer it. If it is "
            "already standalone, return it unchanged. Output only the question.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a study assistant. Answer ONLY from the context below. "
            "If the answer is not in the context, reply exactly: '{not_found}' "
            "Keep answers clear and short.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
).partial(not_found=NOT_FOUND)


def to_messages(history: list[dict]) -> list:
    """Turn [{'role','content'}, ...] into LangChain messages (recent turns only)."""
    out = []
    for m in history[-HISTORY_TURNS:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        out.append(cls(content=m["content"]))
    return out


def standalone_question(llm: ChatGroq, history: list[dict], question: str) -> str:
    """Make a follow-up like 'give an example' searchable by adding the missing context."""
    if not history:
        return question
    result = (condense_prompt | llm).invoke(
        {"history": to_messages(history), "question": question}
    )
    return (result.content or "").strip() or question


def retrieve(db: Chroma, query: str, k: int = 4):
    return db.as_retriever(search_kwargs={"k": k}).invoke(query)


def stream_answer(llm: ChatGroq, docs, history: list[dict], question: str):
    """Yield the answer piece by piece."""
    context = "\n\n".join(d.page_content for d in docs)
    chain = answer_prompt | llm
    for chunk in chain.stream(
        {"context": context, "history": to_messages(history), "question": question}
    ):
        if isinstance(chunk.content, str):
            yield chunk.content


def format_sources(docs) -> list[str]:
    """Unique 'file.pdf, page N' strings (PyPDFLoader pages start from 0)."""
    pairs = {(Path(d.metadata.get("source", "?")).name, d.metadata.get("page", 0) + 1) for d in docs}
    return [f"{name}, page {page}" for name, page in sorted(pairs)]