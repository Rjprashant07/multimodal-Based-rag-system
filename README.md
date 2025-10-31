# 🧠 Multimodal RAG API

This is a FastAPI-based Retrieval-Augmented Generation (RAG) system that integrates:
- **Qdrant** for vector storage and retrieval
- **Sentence Transformers** for embeddings
- **LangChain** for smart text chunking
- **Flan-T5** model for question answering

## 🚀 Features
- Upload and index PDFs, images, or text files
- Hybrid search (dense + sparse)
- Query the knowledge base using natural language
- Deployed-ready for Render or any cloud platform

## 🧩 Project Structure
app/
├── main.py              # FastAPI application entry point
├── modules/
│   ├── ingest.py        # Handles file ingestion, chunking, and embedding
│   ├── query.py         # Performs dense, sparse, and hybrid searches
├── requirements.txt     # All dependencies
├── README.md            # Project documentation

## 🛠 Installation
```bash
git clone <https://github.com/Rjprashant07/multimodal-Based-rag-system.git>
cd app
pip install -r requirements.txt
