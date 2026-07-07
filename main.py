import os
import sys

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

from document_loader import process_documents
from vector_store import build_vectorstore
from rag_agent import build_agent, ask

load_dotenv()

DB_URI = os.getenv("DB_URI", "postgresql://localhost:5432/rag_chatbot")
DOCS = "./hr_documents_pack/initial_docs/"

SORULAR = [
    "What is the company's leave policy?",
    "How many vacation days do employees get?",
    "What are the steps in the offboarding process?",
    "What are the IT security requirements for new employees?",
    "What is the performance review process?",
    "How do I submit travel expenses for reimbursement?",
]


def cmd_setup():
    print("Belgeler yukleniyor ve ChromaDB'ye kaydediliyor...\n")
    docs = process_documents(DOCS)
    build_vectorstore(docs)
    print("\nBitti.")


def cmd_test():
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        agent = build_agent(checkpointer)

        print(">>> Test sorulari\n")
        # her soruyu ayri thread'de soruyorum ki birbirine karismasin
        for i, soru in enumerate(SORULAR, 1):
            print(f"{i}. {soru}")
            print("   ->", ask(agent, soru, thread_id=f"test-{i}"), "\n")

        # burada tek thread kullaniyorum, follow-up sorular
        # onceki konusmayi hatirlamali (kisa donem bellek)
        print(">>> Hafiza testi\n")
        for soru in ["What is the leave policy?",
                     "What about sick leave?",
                     "What about vacation?"]:
            print("Sen:", soru)
            print("Bot:", ask(agent, soru, thread_id="memory-test"), "\n")


def cmd_chat():
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        agent = build_agent(checkpointer)

        print("Sohbet basladi (cikmak icin quit)\n")
        while True:
            try:
                user = input("Sen: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGorusuruz!")
                break
            if user.lower() in {"quit", "exit", "cik", "q"}:
                print("Gorusuruz!")
                break
            if not user:
                continue
            # sabit thread => tum sohbet boyunca hafiza korunur
            print("Bot:", ask(agent, user, thread_id="chat-session"), "\n")


def main():
    komut = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if komut == "setup":
        cmd_setup()
    elif komut == "test":
        cmd_test()
    elif komut == "chat":
        cmd_chat()
    else:
        print("Kullanim: python main.py [setup|test|chat]")


if __name__ == "__main__":
    main()