import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = "./chroma_db"
COLLECTION = "vbo-aillm-bc-rag"


def get_embeddings():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY yok, .env dosyasina ekle")

    # OpenRouter, OpenAI ile ayni formati kullaniyor, sadece base_url degisiyor
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=key,
        openai_api_base="https://openrouter.ai/api/v1",
    )


def build_vectorstore(documents):
    # ilk kurulum - belgeleri embed edip diske yaz
    store = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
    )
    print(f"{len(documents)} parca kaydedildi -> {COLLECTION}")
    return store


def load_vectorstore():
    # daha once olusturulmus koleksiyonu tekrar ac
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
    )


def get_retriever(k=4):
    return load_vectorstore().as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    from document_loader import process_documents

    docs = process_documents("./hr_documents_pack/initial_docs/")
    build_vectorstore(docs)

    # hizli bir kontrol
    hits = get_retriever().invoke("What is the leave policy?")
    print(f"\n{len(hits)} sonuc:")
    for h in hits:
        print(" -", h.metadata.get("file_name"), "->", h.page_content[:60].replace("\n", " "))