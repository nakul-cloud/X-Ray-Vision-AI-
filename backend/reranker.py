# reranker.py

from sentence_transformers import CrossEncoder
import os

# Local model path
LOCAL_RERANKER_PATH = r"D:\X_Ray\models\marco-MiniLM-L-6-v2"

reranker = None

def get_reranker():
    global reranker
    if reranker is None:
        if not os.path.exists(LOCAL_RERANKER_PATH):
            raise FileNotFoundError(f"[ERROR] Reranker model not found at {LOCAL_RERANKER_PATH}")
        reranker = CrossEncoder(LOCAL_RERANKER_PATH)
    return reranker

from text_embedding_utils import sanitize_text

def rerank_chunks(query, retrieved_chunks):
    """
    Rerank retrieved chunks using local cross-encoder.
    ✔ Safe input handling
    ✔ Robust error handling & Empty list check
    ✔ Logs scores and IDs
    ✔ Fallback to original order
    """
    
    # 4. Fallback if reranker fails -> return original chunks unmodified
    try:
        # Check empty input
        if not retrieved_chunks:
            print("[DEBUG] rerank_chunks: No chunks to rerank.")
            return []

        clean_query = sanitize_text(query)
        if not clean_query:
            print("[WARN] rerank_chunks: Query empty after sanitization.")
            return retrieved_chunks

        # Create pairs of (query, chunk_text)
        pairs = []
        doc_ids = []
        for c in retrieved_chunks:
            clean_chunk = sanitize_text(c.get("cleaned_text", ""))
            pairs.append([clean_query, clean_chunk])
            doc_ids.append(c.get("id", "unknown"))

        # Get similarity scores
        print(f"[DEBUG] Reranking {len(pairs)} chunks...")
        scores = get_reranker().predict(pairs)

        # Attach scores to chunks
        reranked = []
        for i, (item, score) in enumerate(zip(retrieved_chunks, scores)):
            item["rerank_score"] = float(score)
            reranked.append(item)
            # Log individual score for debugging
            if i < 3: # Log first 3 only to avoid spam
                print(f"   Score: {score:.4f} | ID: {doc_ids[i]}")

        # Sort highest score first
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Log Top ID
        if reranked:
             print(f"[DEBUG] Top Reranked Chunk ID: {reranked[0].get('id')} (Score: {reranked[0]['rerank_score']:.4f})")

        return reranked

    except Exception as e:
        print(f"[WARN] Reranker failed: {e}. Returning original chunks.")
        # Fallback
        return retrieved_chunks
