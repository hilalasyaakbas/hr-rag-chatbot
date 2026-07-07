import os

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from vector_store import get_retriever

# Modele her seferinde arama yaptirmak icin biraz zorluyoruz,
# yoksa bazen tool'u atlayip kafasindan cevap vermeye calisiyor
SYSTEM_PROMPT = (
    "You are an HR assistant. For EVERY question, you MUST first call the "
    "search_hr_documents tool to look up the answer — never answer from memory "
    "without searching. If the first search does not contain the answer, try "
    "calling the tool again with different keywords before giving up. "
    "Keep answers SHORT (2-3 sentences). ALWAYS cite the source document using "
    "the file_name metadata, e.g. (source: leave_policy.docx)."
)


@tool
def search_hr_documents(query: str) -> str:
    """Search the HR documents and return the relevant parts with their sources."""
    docs = get_retriever(k=4).invoke(query)
    if not docs:
        return "No relevant documents found."

    bloklar = []
    for d in docs:
        kaynak = d.metadata.get("file_name", "unknown")
        bloklar.append(f"[source: {kaynak}]\n{d.page_content}")
    return "\n\n".join(bloklar)


def get_chat_model():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY yok, .env dosyasina ekle")

    # not: odevdeki flash-lite artik OpenRouter'da yok, flash kullaniyoruz
    # max_tokens dusuk cunku ucretsiz kredide limit var
    return init_chat_model(
        "openai:google/gemini-2.5-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        temperature=0,
        max_tokens=1024,
    )


def build_agent(checkpointer):
    return create_agent(
        get_chat_model(),
        tools=[search_hr_documents],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def ask(agent, question, thread_id):
    # ayni thread_id => ayni konusma, model onceki mesajlari hatirliyor
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    return result["messages"][-1].content