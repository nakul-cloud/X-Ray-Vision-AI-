from supabase_client import supabase
from text_embedding_utils import embed_text

def regenerate_embeddings():
    print("🔄 Fetching chunks from database...")

    chunks = supabase.table("chunks").select("*").execute().data
    print(f"📌 Total chunks found: {len(chunks)}")

    for c in chunks:
        cid = c["chunk_id"]
        text = c["cleaned_text"]

        print(f"➡️ Re-embedding chunk {cid}...")

        vec = embed_text(text)

        # Remove old embedding for this chunk (optional but cleaner)
        supabase.table("embeddings").delete().eq("chunk_id", cid).execute()

        supabase.table("embeddings").insert({
            "chunk_id": cid,
            "embedding": vec,
            "pipeline_stage": c.get("section_title", ""),
            "algorithm": "bge-large-updated"
        }).execute()

    print("\n🎉 All embeddings regenerated successfully!")

if __name__ == "__main__":
    regenerate_embeddings()
