"""Stage 2: Streamlit UI with PDF upload, auto-indexing, chat memory, streaming, and sources."""
import os
from pathlib import Path

import streamlit as st

from rag import (
    DATA_DIR,
    format_sources,
    get_db,
    get_llm,
    indexed_sources,
    ingest_pdf,
    pending_pdfs,
    retrieve,
    standalone_question,
    stream_answer,
)

st.set_page_config(page_title="Study Assistant", page_icon="📚")
st.title("📚 Study Assistant")
st.caption("Ask questions about your PDFs. Answers come only from your documents, with sources.")

# Stop early with a clear message if API keys are missing
missing = [k for k in ("GOOGLE_API_KEY", "GROQ_API_KEY") if not os.getenv(k)]
if missing:
    st.error(f"Missing in your .env file: {', '.join(missing)}")
    st.stop()


@st.cache_resource
def load_db():
    return get_db()


@st.cache_resource
def load_llm():
    return get_llm()


db = load_db()
llm = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- sidebar: upload and index documents ----------
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploads:
        DATA_DIR.mkdir(exist_ok=True)
        for f in uploads:
            target = DATA_DIR / f.name
            if not target.exists():
                target.write_bytes(f.getvalue())

    # Index only the PDFs that are new (uploaded now, or dropped into data/ by hand)
    for pdf in pending_pdfs(db):
        with st.spinner(f"Indexing {pdf.name} ..."):
            try:
                n = ingest_pdf(db, pdf)
            except Exception as e:
                st.error(f"Could not index {pdf.name}: {e}")
                continue
        if n == 0:
            st.warning(f"{pdf.name}: no text found (scanned PDF?). Use a text-based PDF.")
        else:
            st.success(f"Indexed {pdf.name} ({n} chunks)")

    st.subheader("Indexed")
    sources = sorted(indexed_sources(db))
    if sources:
        for s in sources:
            st.write(f"📄 {Path(s).name}")
    else:
        st.write("Nothing yet.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- chat ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.write(f"- {s}")

if not indexed_sources(db):
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

question = st.chat_input("Ask about your documents")
if question:
    history = list(st.session_state.messages)  # earlier turns only
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            search_query = standalone_question(llm, history, question)
            docs = retrieve(db, search_query)
            answer = st.write_stream(stream_answer(llm, docs, history, question))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        # Show sources only when the answer actually came from the documents
        found = "could not find" not in answer.lower()
        sources = format_sources(docs) if found else []
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )