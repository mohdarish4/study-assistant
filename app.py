"""Stage 3: Streamlit UI with an agent that chooses between your documents and web search."""
import os
from pathlib import Path

import streamlit as st

from agent import build_agent, collect_sources, run_agent, web_search_available
from rag import (
    DATA_DIR,
    LLM_PROVIDER,
    get_db,
    get_llm,
    indexed_sources,
    ingest_pdf,
    pending_pdfs,
)

st.set_page_config(page_title="Study Assistant", page_icon="📚")
st.title("📚 Study Assistant")
st.caption("An agent that answers from your PDFs first, and searches the web only when needed.")

# Stop early with a clear message if API keys are missing
required_keys = ["GOOGLE_API_KEY"] + (["GROQ_API_KEY"] if LLM_PROVIDER == "groq" else [])
missing = [k for k in required_keys if not os.getenv(k)]
if missing:
    st.error(f"Missing in your .env file: {', '.join(missing)}")
    st.stop()


@st.cache_resource
def load_db():
    return get_db()


@st.cache_resource
def load_llm():
    return get_llm()


@st.cache_resource
def load_agent(_db, _llm, use_web: bool):
    return build_agent(_db, _llm, use_web)


db = load_db()
llm = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- sidebar ----------
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

    st.subheader("Agent")
    if web_search_available():
        use_web = st.toggle("Allow web search", value=True)
    else:
        use_web = False
        st.caption("Web search is off. Add TAVILY_API_KEY to your .env file to enable it.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

agent = load_agent(db, llm, use_web)

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
        answer, new_messages = "", []
        try:
            with st.status("Thinking ...", expanded=True) as status:
                for event in run_agent(agent, history, question):
                    if event[0] == "step":
                        st.write(event[1])
                    else:
                        _, answer, new_messages = event
                status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        if not answer:
            answer = "Sorry, I could not produce an answer. Please try again."
        st.markdown(answer)

        sources = collect_sources(answer, new_messages)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )