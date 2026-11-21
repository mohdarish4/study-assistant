"""Stage 1, step 1: read PDFs from data/, split into chunks, store embeddings in ChromaDB."""
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("data")
DB_DIR = "chroma_db"
BATCH_SIZE = 50  # small batches to stay inside free-tier rate limits


def main():
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found. Put your PDF files inside the 'data' folder first.")
        return

    # 1. Load every page of every PDF
    pages = []
    for pdf in pdfs:
        loaded = PyPDFLoader(str(pdf)).load()
        print(f"Loaded {pdf.name}: {len(loaded)} pages")
        pages.extend(loaded)

    # 2. Split pages into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")

    # 3. Start fresh so re-running does not create duplicates
    if Path(DB_DIR).exists():
        shutil.rmtree(DB_DIR)

    # 4. Embed and store in batches
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    for i in range(0, len(chunks), BATCH_SIZE):
        db.add_documents(chunks[i : i + BATCH_SIZE])
        print(f"Stored {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")
        time.sleep(1)

    print("Done. Now run: uv run ask.py")


if __name__ == "__main__":
    main()