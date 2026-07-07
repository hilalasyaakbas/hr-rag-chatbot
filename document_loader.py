import os
from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TYPE_MAP = {".pdf": "pdf", ".docx": "document", ".txt": "text"}


def _dir_loader(folder, pattern, loader, **kw):
    return DirectoryLoader(
        folder, glob=pattern, loader_cls=loader, loader_kwargs=kw or None
    )


def load_documents(folder):
    docs = []
    docs += _dir_loader(folder, "**/*.docx", Docx2txtLoader).load()
    docs += _dir_loader(folder, "**/*.pdf", PyPDFLoader).load()
    # latin-1 sectim cunku bazi txt'lerde bozuk bayt var, utf-8 patliyordu
    docs += _dir_loader(folder, "**/*.txt", TextLoader, encoding="latin-1").load()
    print(f"{len(docs)} belge okundu")
    return docs


def chunk_and_enrich(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    sonuc = []
    for doc in documents:
        path = Path(doc.metadata.get("source", ""))
        ext = path.suffix

        if path.exists():
            st = path.stat()
            boyut = st.st_size
            olusturma = datetime.fromtimestamp(st.st_ctime).isoformat()
            degistirme = datetime.fromtimestamp(st.st_mtime).isoformat()
        else:
            boyut = len(doc.page_content)
            olusturma = degistirme = datetime.now().isoformat()

        sayfa = doc.metadata.get("page", 0)
        toplam_karakter = len(doc.page_content)

        for i, chunk in enumerate(splitter.split_documents([doc])):
            chunk.metadata = {
                "file_name": path.name,
                "file_extension": ext,
                "file_size_bytes": boyut,
                "character_count": toplam_karakter,
                "chunk_index": i,
                "chunk_size": len(chunk.page_content),
                "chunk_overlap": CHUNK_OVERLAP,
                "document_type": TYPE_MAP.get(ext.lower(), "unknown"),
                "creation_date": olusturma,
                "last_modified": degistirme,
                "ingestion_timestamp": datetime.now().isoformat(),
                "page_number": sayfa,
                "section_title": path.stem,
            }
            sonuc.append(chunk)

    print(f"{len(sonuc)} parca olustu")
    return sonuc


def process_documents(folder):
    return chunk_and_enrich(load_documents(folder))


if __name__ == "__main__":
    docs = process_documents("./hr_documents_pack/initial_docs/")
    print(f"\nToplam parca: {len(docs)}")
    if docs:
        print("Metadata alan sayisi:", len(docs[0].metadata))
        print("Ornek:", docs[0].metadata)
        print("Icerik:", docs[0].page_content[:120])