import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler

# --- Initialize models ---
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# --- Connect to persistent Qdrant Cloud ---
qdrant = QdrantClient(
    url="https://bc7fde05-251b-4940-b11a-683326ab9396.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.CTZmywE4q_bNllGjFY5_ILpWDulZxyvrINCL15v7LVM"
)

collection_name = "documents"

# --- Sparse Retriever (BM25) ---
bm25_index = None
bm25_texts = []

def initialize_bm25():
    """
    Initialize BM25 dynamically from stored text chunks in Qdrant.
    """
    global bm25_index, bm25_texts

    print("📥 Loading text chunks from Qdrant for BM25...")
    points, _ = qdrant.scroll(collection_name=collection_name, limit=5000)
    bm25_texts = [p.payload.get("text", "") for p in points if p.payload.get("text")]

    if not bm25_texts:
        print(" No text chunks found in Qdrant for BM25 initialization.")
        return

    tokenized_corpus = [text.split() for text in bm25_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    print(f"BM25 index initialized with {len(bm25_texts)} text chunks.")


# --- Dense Retrieval ---
def dense_search(query_text, top_k=5):
    """Semantic dense retrieval using embeddings from Qdrant."""
    query_vector = embedder.encode(query_text).tolist()
    search_result = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
    )

    dense_results = [
        {
            "text": hit.payload.get("text", ""),
            "file_path": hit.payload.get("file_path"),
            "file_type": hit.payload.get("file_type"),
            "score": float(hit.score),
            "type": "dense"
        }
        for hit in search_result
    ]
    return dense_results


# --- Sparse Retrieval ---
def sparse_search(query_text, top_k=5):
    """Keyword-based BM25 retrieval over stored chunks."""
    global bm25_index

    if bm25_index is None:
        initialize_bm25()

    if bm25_index is None:
        return []

    tokenized_query = query_text.split()
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    sparse_results = [
        {
            "text": bm25_texts[i],
            "file_path": None,
            "score": float(scores[i]),
            "type": "sparse"
        }
        for i in top_indices
    ]
    return sparse_results


# --- Hybrid Retrieval ---
def hybrid_search(query_text, top_k=5, alpha=0.5):
    """
    Combine dense (semantic) and sparse (keyword) retrieval.
    alpha balances weight: 0.5 = equal.
    """
    dense_results = dense_search(query_text, top_k)
    sparse_results = sparse_search(query_text, top_k)

    combined = dense_results + sparse_results
    if not combined:
        return []

    # Normalize and combine scores
    scores = np.array([r["score"] for r in combined]).reshape(-1, 1)
    normalized = MinMaxScaler().fit_transform(scores).flatten()

    for i, r in enumerate(combined):
        r["hybrid_score"] = (
            normalized[i] * (alpha if r["type"] == "dense" else (1 - alpha))
        )

    sorted_results = sorted(combined, key=lambda x: x["hybrid_score"], reverse=True)
    return sorted_results[:top_k]

