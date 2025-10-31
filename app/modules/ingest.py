import os
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from datetime import datetime
from modules.query import initialize_bm25
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

# --- Initialize models ---
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# --- Initialize Qdrant ---
qdrant = QdrantClient(
    url="https://bc7fde05-251b-4940-b11a-683326ab9396.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.CTZmywE4q_bNllGjFY5_ILpWDulZxyvrINCL15v7LVM"
)
collection_name = "documents"
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config={"size": 768, "distance": "Cosine"},  # 768 for mpnet
)


def smart_chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", ".", "!", "?", "\n", " "]
    )
    chunks = splitter.split_text(text)
    print(f" Created {len(chunks)} semantic chunks.")
    return chunks


def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"OCR failed: {str(e)}"

def extract_from_pdf(pdf_path):
    """Extract text and OCR from all pages and images in a PDF."""
    text_content = []
    pdf_document = fitz.open(pdf_path)

    for page_index, page in enumerate(pdf_document):
        # --- Extract visible text ---
        page_text = page.get_text("text")
        if page_text.strip():
            text_content.append(f"[Page {page_index+1} Text]\n{page_text}")

        # --- Extract and OCR images ---
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                os.makedirs("data/processed", exist_ok=True)
                image_filename = f"{uuid.uuid4()}.{image_ext}"
                image_path = os.path.join("data/processed", image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # OCR on image
                ocr_text = extract_text_from_image(image_path)
                if ocr_text.strip():
                    text_content.append(f"[Page {page_index+1} Image {img_index+1} OCR]\n{ocr_text}")

            except Exception as e:
                print(f" Error extracting image {img_index} on page {page_index}: {e}")

    full_text = "\n".join(text_content)
    return full_text.strip()


"""Main ingestion function for any supported file type."""

def process_file(file_path, file_type):
    if file_type.lower() in ["txt", "text"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    elif file_type.lower() in ["jpg", "jpeg", "png"]:
        content = extract_text_from_image(file_path)
    elif file_type.lower() == "pdf":
        content = extract_from_pdf(file_path)
    else:
        raise ValueError("Unsupported file type")


    # --- Create semantic chunks ---
    chunks = smart_chunk_text(content)

    all_chunks = []
    for idx, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        all_chunks.append({"text": chunk})

        qdrant.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "file_path": file_path,
                        "file_type": file_type,
                        "chunk_index": idx,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            ],
        )

    print(f"Ingested {len(chunks)} chunks into Qdrant.")
    initialize_bm25(all_chunks)  # build BM25 for hybrid search
    return {"status": "success", "chunks": len(chunks), "path": file_path}
