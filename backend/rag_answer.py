# rag_answer.py

from retrieval import retrieve_chunks
from reranker import rerank_chunks
from groq_llm import GroqLLM
from prompt_builder import build_rag_prompt
from text_embedding_utils import sanitize_text

# Initialize Groq LLM globally once
llm = GroqLLM()


def answer_with_rag(query: str, top_k: int = 5, debug: bool = False):
    """
    Full RAG processing pipeline with error handling and debug output.
    """
    
    # Pre-sanitize to prevent emojis crashing logging/LLM
    clean_query = sanitize_text(query)
    if not clean_query:
        return "Please ask a text-based question."

    try:
        # Step 1: Retrieve using CLEAN query
        retrieved = retrieve_chunks(clean_query, top_k=top_k)
        if not retrieved:
            return "[WARN] No relevant context was found in Supabase."

        # Step 2: Rerank
        reranked = rerank_chunks(clean_query, retrieved)
        final_chunks = reranked[:2]

        if debug:
            print("Top chunks after reranking:")
            for c in final_chunks:
                print(c["cleaned_text"][:200])

        # Step 3: Prompt
        prompt = build_rag_prompt(clean_query, final_chunks)

        # Step 4: LLM completion
        response = llm.generate(prompt)

        if not response or response.strip() == "":
            return "[WARN] Groq returned an empty response."

        return response

    except Exception as e:
        return f"[ERROR] RAG Error: {str(e)}"

    # ----------------------------------------------------------------------
    # STEP 1 — VECTOR RETRIEVAL from Supabase
    # ----------------------------------------------------------------------
    retrieved = retrieve_chunks(query, top_k=top_k)

    if not retrieved:
        return "[ERROR] No relevant context found in database."

    if debug:
        print("\n[DEBUG] Retrieved Chunks:")
        for i, c in enumerate(retrieved):
            print(f"[{i+1}] {c['cleaned_text'][:150]}...\n")

    # ----------------------------------------------------------------------
    # STEP 2 — RERANKING (CrossEncoder)
    # ----------------------------------------------------------------------
    reranked = rerank_chunks(query, retrieved)

    # pick best 2 chunks
    final_chunks = reranked[:2]

    if debug:
        print("\n[DEBUG] Reranked Chunks:")
        for i, c in enumerate(final_chunks):
            print(f"[{i+1}] Score={c['rerank_score']:.4f}")
            print(c["cleaned_text"][:150], "\n")

    # ----------------------------------------------------------------------
    # STEP 3 — PROMPT BUILDING FOR LLM
    # ----------------------------------------------------------------------
    prompt = build_rag_prompt(query, final_chunks)

    if debug:
        print("\n[DEBUG] Final LLM Prompt >>>")
        print(prompt)

    # ----------------------------------------------------------------------
    # STEP 4 — LLM RESPONSE (GROQ)
    # ----------------------------------------------------------------------
    try:
        response = llm.generate(prompt)
    except Exception as e:
        return f"[ERROR] Groq LLM Error: {str(e)}"

    return response
