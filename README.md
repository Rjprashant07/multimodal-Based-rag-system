# 🧠 Multimodal RAG API

This is a FastAPI-based Retrieval-Augmented Generation (RAG) system that integrates:
- **Qdrant** for vector storage and retrieval
- **Sentence Transformers** for embeddings
- **LangChain** for smart text chunking
- **Flan-T5** model for question answering

## 🚀 Features

- 📄 **Upload and index PDFs, images, or text files**
- 🔍 **Hybrid search** combining dense (semantic) and sparse (keyword) retrieval
- 🧠 **Context-aware generation** using Flan-T5
- 🧩 **LangChain-powered document chunking**
- 💬 **Natural language querying**
- ☁️ **Deplyoment Ready FastAPI** (Render, AWS, etc.)
- 🛠 **Modular architecture** — easy to extend for new modalities

## 🧩 Project Structure
```plaintext
app/
├── main.py              # FastAPI application entry point
| init.py
├── modules/
│   ├── ingest.py        # Handles file ingestion, chunking, and embedding
│   ├── query.py         # Performs dense, sparse, and hybrid searches
├── requirements.txt     # All dependencies
├── README.md            # Project documentation






        ┌─────────────────────────────┐
        │          User Query         │
        └──────────────┬──────────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │  Embedding Generator   │
           │ (Sentence Transformers)│
           └────────────┬───────────┘
                        │
                        ▼
             ┌──────────────────┐
             │   Qdrant Vector   │
             │     Database      │
             └────────┬──────────┘
                      │
                      ▼
             ┌──────────────────┐
             │  Context Builder  │
             │   (LangChain)     │
             └────────┬──────────┘
                      │
                      ▼
             ┌──────────────────┐
             │  Generator Model  │
             │     (Flan-T5)     │
             └──────────────────┘



## 🛠 Installation
```bash
git clone <https://github.com/Rjprashant07/multimodal-Based-rag-system.git>
cd app
pip install -r requirements.txt





---
👨‍💻 Author

rjprashant07
🔗 GitHub

📧 prashantranjan1999@gmail.com