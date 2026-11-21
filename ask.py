"""Stage 1, step 2: ask questions about your PDFs. Answers stream live and show sources."""
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

DB_DIR = "chroma_db"
GROQ_MODEL = "openai/gpt-oss-120b"  # use the chat model that worked for you

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})  # top 4 most similar chunks

llm = ChatGroq(model=GROQ_MODEL, temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a study assistant. Answer ONLY from the context below. "
            "If the answer is not in the context, say: 'I could not find this in your documents.' "
            "Keep answers clear and short.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)
chain = prompt | llm


def main():
    print("Study Assistant ready. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break

        # 1. Retrieve relevant chunks
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        # 2. Generate the answer, streaming it word by word
        answer = ""
        print("\nAssistant: ", end="", flush=True)
        for chunk in chain.stream({"context": context, "question": question}):
            print(chunk.content, end="", flush=True)
            answer += chunk.content

        # 3. Show sources only when the answer came from the documents
        if "could not find" not in answer.lower():
            sources = sorted(
                {(d.metadata.get("source", "?"), d.metadata.get("page", 0) + 1) for d in docs}
            )
            print("\n\nSources:")
            for src, page in sources:
                print(f"  - {src}, page {page}")
        print()


if __name__ == "__main__":
    main()