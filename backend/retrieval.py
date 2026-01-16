# retrieval.py

from supabase_client import supabase
from text_embedding_utils import embed_text, sanitize_text


def retrieve_chunks(query: str, top_k: int = 5):
    """
    Retrieves relevant document chunks from Supabase using pgvector similarity.
    ✔ Logs sanitized query & embedding shape
    ✔ Validates inputs before RPC
    ✔ Handles RPC failures gracefully
    """
    # 1. Sanitize & Log
    clean_query = sanitize_text(query)
    print(f"[DEBUG] match_chunks sanitized query: '{clean_query}'")
    
    if not clean_query:
        print("[WARN] Query was empty after sanitization.")
        return []

    # 2. Embed Query
    try:
        query_embedding = embed_text(clean_query)
        
        # Validation: Must be a list of floats
        if not query_embedding or not isinstance(query_embedding, list):
            print("[ERROR] Embeddings generation failed to return a list.")
            return []
            
        print(f"[DEBUG] Embedding generated. Shape: {len(query_embedding)}, Sample: {query_embedding[:3]}...")
            
    except Exception as e:
        print(f"[ERROR] Error embedding query: {e}")
        return []

    # 3. Call Supabase match_chunks
    try:
        print(f"[DEBUG] Calling Supabase match_chunks with top_k={top_k}")
        
        response = supabase.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": top_k
            }
        ).execute()

        # 4. Validate output
        if not response.data:
            print("[WARN] Supabase returned 0 chunks (empty response).")
            # Return empty list, let Prompt Builder handle fallback text
            return []
            
        chunks = response.data
        print(f"[DEBUG] Retrieved {len(chunks)} chunks from Supabase.")
        return chunks

    except Exception as e:
        print(f"[ERROR] Supabase RPC failed: {e}")
        # Return empty list instead of crashing
        return []
