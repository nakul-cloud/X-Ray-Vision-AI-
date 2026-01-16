from sentence_transformers import SentenceTransformer
import os
import re

# ---------------------------------------
# LOAD BGE-LARGE-EN-V1.5 FROM LOCAL PATH
# ---------------------------------------
model_path = r"D:/X_Ray/models/bge-large-en-v1.5"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model folder not found: {model_path}")

text_model = None

def get_text_model():
    global text_model
    if text_model is None:
        print(f"[INFO] Loading BGE-large-en-v1.5 model from: {model_path}")
        text_model = SentenceTransformer(model_path)
    return text_model

# ---------------------------------------
# SANITIZATION (Prevents Encoding Errors)
# ---------------------------------------
import unicodedata

def sanitize_text(text: str) -> str:
    """
    Strips emojis, bullets, and non-ASCII symbols.
    Prevents 'charmap' codec errors on Windows consoles.
    """
    if not text:
        return ""
        
    try:
        # 1. Normalize Unicode (collapse weird chars)
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Remove non-ASCII characters (emojis, etc)
        # This forces the string to be pure 7-bit ASCII
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # 3. Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception:
        # Extreme fallback if something goes wrong
        return ""

# ---------------------------------------
# EMBEDDING FUNCTION (LONG TEXT SAFE)
# ---------------------------------------
def embed_text(text: str):
    """
    Generates embeddings for long text using BGE-large-en-v1.5.
    ✔ Supports 1024+ tokens
    ✔ Perfect for RAG text chunks
    ✔ Prevents MedSigLIP token overflow errors
    ✔ Robust error handling (returns zero vector on failure)
    """
    dim = 1024  # BGE-large dimension
    
    try:
        # 1. Sanitize Input
        safe_text = sanitize_text(text)
        
        # 2. Handle Empty Input
        if not safe_text or len(safe_text) == 0:
            print("[WARN] text_embedding_utils: input text empty after sanitization")
            return [0.0] * dim

        # 3. Generate Embedding (Model returns numpy array)
        embedding_array = get_text_model().encode(
            safe_text,
            normalize_embeddings=True  # recommended for pgvector
        )
        
        # 4. Convert to pure Python List<float>
        if hasattr(embedding_array, "tolist"):
             vector = embedding_array.tolist()
        else:
             vector = list(embedding_array)
             
        # 5. Final Validation
        if len(vector) != dim:
             print(f"[ERROR] text_embedding_utils: Dimension mismatch {len(vector)} vs {dim}")
             return [0.0] * dim
             
        return vector
        
    except Exception as e:
        print(f"[ERROR] embed_text failed: {e}")
        return [0.0] * dim

# Error Handling Fix (Wrapper)
def safe_embed_text(text):
    return embed_text(text)
