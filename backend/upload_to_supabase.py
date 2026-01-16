import os
import sys
from supabase_client import supabase
from chunk_xray_pdfs import process_pdf
from text_embedding_utils import embed_text   # UPDATED IMPORT

# -----------------------------
# CONFIG
# -----------------------------
PDF_FOLDER = r"D:\X_Ray\pdf"


# -----------------------------
# PROCESS & UPLOAD SINGLE PDF
# -----------------------------
def upload_pdf(pdf_path):
    file_name = os.path.basename(pdf_path)
    print(f"\n🚀 Uploading PDF: {file_name}")
    print(f"📄 Path: {pdf_path}")

    # 1️⃣ INSERT INTO documents TABLE
    try:
        doc = supabase.table("documents").insert({
            "title": file_name,
            "file_name": file_name,
            "metadata": {"source": "xray_pdf_local"}
        }).execute()
    except Exception as e:
        print("❌ ERROR inserting into documents:", e)
        return

    if not doc.data:
        print("❌ ERROR: No document returned from Supabase")
        return

    doc_id = doc.data[0]["doc_id"]
    print(f"📌 Document ID: {doc_id}")

    # 2️⃣ CHUNK PDF
    try:
        chunks = process_pdf(pdf_path)
    except Exception as e:
        print("❌ ERROR during chunking:", e)
        return

    if len(chunks) == 0:
        print("⚠️ No chunks generated for this PDF")
        return

    print(f"📌 Total Chunks Found: {len(chunks)}")

    # 3️⃣ LOOP THROUGH CHUNKS
    for index, item in enumerate(chunks):
        section = item["section"]
        text = item["chunk"]

        print(f"\n➡️ Uploading Chunk {index + 1}/{len(chunks)}")
        print(f"   Section: {section}")

        # 3. INSERT CHUNK INTO chunks TABLE
        try:
            chunk_row = supabase.table("chunks").insert({
                "doc_id": doc_id,
                "section_title": section,
                "cleaned_text": text,
                "metadata": {"section": section}
            }).execute()
        except Exception as e:
            print("❌ ERROR inserting chunk:", e)
            continue

        if not chunk_row.data:
            print("❌ ERROR: No chunk_id returned")
            continue

        chunk_id = chunk_row.data[0]["chunk_id"]
        print(f"   ✔ chunk_id: {chunk_id}")

        # 4️⃣ GENERATE EMBEDDING (BGE-large)
        try:
            vec = embed_text(text)
            if not vec:
                print("⚠️ Empty embedding returned, skipping")
                continue
        except Exception as e:
            print("❌ ERROR generating embedding:", e)
            continue

        # 5️⃣ INSERT EMBEDDING INTO embeddings TABLE
        try:
            supabase.table("embeddings").insert({
                "chunk_id": chunk_id,
                "embedding": vec,
                "pipeline_stage": section,
                "algorithm": None
            }).execute()
        except Exception as e:
            print("❌ ERROR inserting embedding:", e)
            continue

        print("   ✔ Embedding uploaded")

    print(f"\n✅ Completed uploading PDF: {file_name}")


# -----------------------------
# PROCESS ALL PDFs IN FOLDER
# -----------------------------
def upload_all_pdfs():
    print(f"\n📁 Scanning folder: {PDF_FOLDER}")

    if not os.path.exists(PDF_FOLDER):
        print("❌ ERROR: PDF folder does not exist!")
        sys.exit()

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

    print("📄 PDFs found:", pdf_files)

    if not pdf_files:
        print("❌ No PDF files found in the folder!")
        return

    for file_name in pdf_files:
        full_path = os.path.join(PDF_FOLDER, file_name)
        upload_pdf(full_path)

    print("\n🎉 ALL PDFs Successfully uploaded to Supabase!")


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    upload_all_pdfs()
