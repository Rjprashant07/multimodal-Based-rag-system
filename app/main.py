import nest_asyncio, uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from datetime import datetime
from pyngrok import ngrok
import os, uuid, fitz, pytesseract
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.query import dense_search, sparse_search, hybrid_search

import asyncio # Import asyncio

# --- Allow async event loop reuse in Colab ---
nest_asyncio.apply()

# --- Initialize components ---
app = FastAPI(title="Multimodal RAG API")
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Initialize Qdrant client
import os
from qdrant_client import QdrantClient

# Read from environment variables
QDRANT_URL = os.getenv("url")
QDRANT_API_KEY = os.getenv("api_key")

# Initialize Qdrant client
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

collection_name = "documents"

# Create collection if missing
if collection_name not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embedder.get_sentence_embedding_dimension(), distance=models.Distance.COSINE),
    )

# Load small LLM (Flan-T5-base)
from transformers import pipeline
rag_model = pipeline("text2text-generation", model="google/flan-t5-large")

# --- Helper functions ---
def extract_text(file_path, file_type):
    text = ""
    if file_type == "pdf":
        pdf = fitz.open(file_path)
        for page in pdf:
            text += page.get_text("text")
    elif file_type in ["png", "jpg", "jpeg"]:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    else:
        with open(file_path, "r") as f:
            text = f.read()
    return text

def smart_chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", ".", "!", "?", "\n", " "]
    )
    chunks = splitter.split_text(text)
    print(f" Created {len(chunks)} chunks.")
    return chunks


def ingest_file(file_path, filename):
    file_type = filename.split(".")[-1].lower()
    text = extract_text(file_path, file_type)

    if not text.strip():
        return {"status": "error", "message": "No text extracted."}

    chunks = smart_chunk_text(text)
    print(f"🔹 Ingesting {len(chunks)} chunks from {filename}")

    points = []
    for idx, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_index": idx,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        )

    qdrant.upsert(collection_name=collection_name, points=points)
    print(f"Uploaded {len(points)} chunks to Qdrant.")
    return {"status": "success", "chunks": len(points), "filename": filename}

def query_documents(query, top_k=3):
    query_vector = embedder.encode(query).tolist()
    search_result = qdrant.search(collection_name=collection_name, query_vector=query_vector, limit=top_k)
    return [{"text": hit.payload["text"], "score": hit.score, "filename": hit.payload["filename"]} for hit in search_result]

def generate_answer(question):
    results = query_documents(question)
    context = " ".join([r["text"] for r in results])
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    answer = rag_model(prompt, max_new_tokens=200)[0]["generated_text"]
    return answer, results

# --- Endpoints ---
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    result = ingest_file(file_path, file.filename)
    return JSONResponse({"status": "success", "metadata": result})

@app.post("/query")
async def query_api(question: str = Form(...)):
    answer, docs = generate_answer(question)
    return JSONResponse({"question": question, "answer": answer, "sources": docs})

# --- Hybrid Search Endpoints ---


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# Dense search only (existing vector embeddings)
@app.post("/dense")
def dense_endpoint(request: QueryRequest):
    results = query_documents(request.query, top_k=request.top_k)
    return {"method": "dense", "results": results}

# Sparse search only (BM25)
@app.post("/sparse")
def sparse_endpoint(request: QueryRequest):
    # Example: placeholder BM25 search — integrate real one if built
    return {"method": "sparse", "results": ["(Add BM25 results here)"]}

# Hybrid search (dense + sparse combination)
@app.post("/hybrid")
def hybrid_endpoint(request: QueryRequest):
    dense_results = query_documents(request.query, top_k=request.top_k)
    # Combine or weight with BM25 results if available
    hybrid_results = dense_results  # for now, use dense until BM25 added
    return {"method": "hybrid", "results": hybrid_results}



