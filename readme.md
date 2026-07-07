# HR RAG Chatbot

HR belgeleri üzerinde soru-cevap yapan, kısa dönem hafızaya sahip bir RAG chatbot.

## Bileşenler

- Belge yükleme: DirectoryLoader ile DOCX, PDF, TXT
- Parçalama: RecursiveCharacterTextSplitter (500 karakter, 100 overlap), 13 metadata alanı
- Vektör deposu: ChromaDB, OpenRouter text-embedding-3-small embeddings
- Ajan: LangChain create_agent, @tool ile retriever
- Hafıza: PostgreSQL üzerinde PostgresSaver checkpointer

## Dizin Yapısı

​```
hr_rag_chatbot/
├── document_loader.py
├── vector_store.py
├── rag_agent.py
├── main.py
├── requirements.txt
├── .env.example
├── README.md
└── hr_documents_pack/
    └── initial_docs/
​```

## Kurulum

Sanal ortam:

​```bash
python3 -m venv venv
source venv/bin/activate
​```

Paketler:

​```bash
pip install -r requirements.txt
​```

PostgreSQL:

​```bash
brew services start postgresql@14
createdb rag_chatbot
​```

.env dosyası (.env.example dosyasını kopyalayıp doldurun):

​```
OPENROUTER_API_KEY=...
DB_URI=postgresql://kullanici@localhost:5432/rag_chatbot
​```

HR belgelerini hr_documents_pack/initial_docs/ dizinine yerleştirin.

## Çalıştırma

Belgeleri yükleyip vektör deposunu oluşturmak için:

​```bash
python main.py setup
​```

Test sorularını çalıştırmak için:

​```bash
python main.py test
​```

İnteraktif sohbet için:

​```bash
python main.py chat
​```

## Notlar

- Ödevde belirtilen openai:google/gemini-2.5-flash-lite modeli OpenRouter üzerinde geçerli olmadığından openai:google/gemini-2.5-flash kullanılmıştır.
- Embedding ve chat model için tek sağlayıcı olarak OpenRouter kullanılmaktadır.
- Kısa dönem bellek PostgresSaver ile PostgreSQL üzerinde saklanır. checkpointer.setup() bir kez çağrılır, sabit thread_id ile aynı konuşma sürdürülür.