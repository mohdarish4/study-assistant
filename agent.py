"""Stage 3: a LangGraph-based agent that decides between your documents and the web."""
import os
import re

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from rag import indexed_sources, retrieve

try:  # LangChain 1.x
    from langchain.agents import create_agent

    def _make_agent(llm, tools, prompt):
        return create_agent(model=llm, tools=tools, system_prompt=prompt)

except ImportError:  # older LangGraph-only setups
    from langgraph.prebuilt import create_react_agent

    def _make_agent(llm, tools, prompt):
        return create_react_agent(llm, tools, prompt=prompt)


NOT_FOUND = "I could not find this in your documents."
RECURSION_LIMIT = 12  # stops the agent from looping forever

SYSTEM_PROMPT = """You are a study assistant with tools.

Rules:
1. For any question that may relate to the user's documents, call search_documents FIRST.
2. If the documents contain the answer, answer ONLY from them and cite as (file, page N).
3. If they do not: {fallback}
4. Never invent facts, file names, or page numbers. Keep answers short and clear.
5. For greetings or chit-chat, answer directly without using tools.
"""

FALLBACK_WITH_WEB = (
    "say so in one short sentence, then call web_search and answer from its results. "
    "Start that part with 'From the web:' and mention the source URLs."
)
FALLBACK_NO_WEB = f"reply exactly: '{NOT_FOUND}'"


def web_search_available() -> bool:
    return bool(os.getenv("TAVILY_API_KEY"))


def build_tools(db, use_web: bool):
    @tool
    def list_documents() -> str:
        """List the names of the PDF documents the user has uploaded."""
        names = sorted(indexed_sources(db))
        return "\n".join(os.path.basename(n) for n in names) or "No documents uploaded."

    @tool
    def search_documents(query: str) -> str:
        """Search the user's uploaded PDF documents. Use this FIRST for any question that
        could be answered from their documents. Returns passages labelled [file, page N]."""
        docs = retrieve(db, query)
        if not docs:
            return "No relevant passages found in the documents."
        parts = []
        for d in docs:
            name = os.path.basename(d.metadata.get("source", "?"))
            page = d.metadata.get("page", 0) + 1
            parts.append(f"[{name}, page {page}]\n{d.page_content}")
        return "\n\n".join(parts)

    tools = [list_documents, search_documents]

    if use_web:
        from langchain_tavily import TavilySearch

        @tool
        def web_search(query: str) -> str:
            """Search the internet. Use ONLY when the documents do not contain the answer,
            or the question needs current or external information."""
            try:
                result = TavilySearch(max_results=4).invoke({"query": query})
            except Exception as e:  # keep the agent alive if the search fails
                return f"Web search failed: {e}"
            if isinstance(result, str):
                return result
            items = result.get("results", []) if isinstance(result, dict) else []
            return (
                "\n\n".join(f"{r['title']} ({r['url']})\n{r['content']}" for r in items)
                or "No web results."
            )

        tools.append(web_search)

    return tools


def build_agent(db, llm, use_web: bool):
    fallback = FALLBACK_WITH_WEB if use_web else FALLBACK_NO_WEB
    prompt = SYSTEM_PROMPT.format(fallback=fallback)
    return _make_agent(llm, build_tools(db, use_web), prompt)


# ---------- helpers for the UI ----------

def text_of(content) -> str:
    """Message content can be a string or a list of blocks (depends on the provider)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def run_agent(agent, history: list[dict], question: str):
    """Run the agent and yield ('step', text) as it works, then ('final', answer, messages)."""
    messages_in = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
    messages_in.append({"role": "user", "content": question})

    new_messages = []
    for update in agent.stream(
        {"messages": messages_in},
        config={"recursion_limit": RECURSION_LIMIT},
        stream_mode="updates",
    ):
        for node_output in update.values():
            for msg in (node_output or {}).get("messages", []):
                new_messages.append(msg)
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        arg = next(iter(call["args"].values()), "")
                        yield ("step", f"🔧 Using **{call['name']}**: {arg}")
                elif isinstance(msg, ToolMessage):
                    yield ("step", f"✅ **{msg.name}** returned {len(text_of(msg.content))} characters")

    final = next(
        (
            text_of(m.content)
            for m in reversed(new_messages)
            if isinstance(m, AIMessage) and not m.tool_calls and text_of(m.content).strip()
        ),
        "",
    )
    yield ("final", final, new_messages)


def collect_sources(answer: str, messages) -> list[str]:
    """Build the source list from what the tools returned, but only for answers that used them."""
    lowered = answer.lower()
    doc_sources, web_sources = [], []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        text = text_of(m.content)
        if m.name == "search_documents":
            for name, page in re.findall(r"^\[(.+?), page (\d+)\]", text, flags=re.M):
                doc_sources.append(f"📄 {name}, page {page}")
        elif m.name == "web_search":
            for url in re.findall(r"\((https?://[^)\s]+)\)", text):
                web_sources.append(f"🌐 {url}")

    def unique(items):
        return list(dict.fromkeys(items))

    if "from the web" in lowered:
        return unique(web_sources)
    if "could not find" in lowered:
        return []
    return unique(doc_sources)