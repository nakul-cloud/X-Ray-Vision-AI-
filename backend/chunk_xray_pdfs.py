import os
import re
from pypdf import PdfReader
import nltk
nltk.download('punkt')

# ----------------------------
# CONFIG
# ----------------------------
PDF_FOLDER = r"D:\X_Ray\pdf"
MAX_TOKENS = 350
OVERLAP = 60

# ----------------------------
# SECTION HEADER DETECTION
# ----------------------------
SECTION_PATTERNS = [
    # General academic sections
    r"^\s*\d*\.?\s*abstract[:\-]?\s*$",
    r"^\s*\d*\.?\s*introduction[:\-]?\s*$",
    r"^\s*\d*\.?\s*background[:\-]?\s*$",
    r"^\s*\d*\.?\s*related work[:\-]?\s*$",
    r"^\s*\d*\.?\s*methodology[:\-]?\s*$",
    r"^\s*\d*\.?\s*materials and methods[:\-]?\s*$",
    r"^\s*\d*\.?\s*results[:\-]?\s*$",
    r"^\s*\d*\.?\s*discussion[:\-]?\s*$",
    r"^\s*\d*\.?\s*conclusion[:\-]?\s*$",
    r"^\s*\d*\.?\s*limitations[:\-]?\s*$",
    r"^\s*references[:\-]?\s*$",
    r"^\s*index[:\-]?\s*$",

    # CHAPTER headings
    r"^\s*chapter\s+one[:\-]?\s*$",
    r"^\s*chapter\s+two[:\-]?\s*$",
    r"^\s*chapter\s+three[:\-]?\s*$",
    r"^\s*chapter\s+four[:\-]?\s*$",
    r"^\s*chapter\s+five[:\-]?\s*$",
    r"^\s*chapter\s+six[:\-]?\s*$",
    r"^\s*chapter\s+seven[:\-]?\s*$",
    r"^\s*chapter\s+eight[:\-]?\s*$",

    # Major section titles from the pages you shared
    r"^\s*basics[:\-]?\s*$",
    r"^\s*chest[:\-]?\s*$",
    r"^\s*abdomen[:\-]?\s*$",
    r"^\s*skull[:\-]?\s*$",
    r"^\s*spine[:\-]?\s*$",
    r"^\s*pelvis[, ]*hips[:\-]?\s*$",
    r"^\s*elbow[, ]*wrist[:\-]?\s*$",
    r"^\s*ankle[, ]*foot[:\-]?\s*$",

    # Subsections from Chest Imaging outline
    r"^\s*diagnostic imaging techniques[:\-]?\s*$",
    r"^\s*how to read a chest x\-ray\s*\(cxr\)[:\-]?\s*$",
    r"^\s*the normal chest x\-ray\s*\(cxr\)[:\-]?\s*$",
    r"^\s*the abnormal mediastinum[:\-]?\s*$",
    r"^\s*bone pathology on cxrs[:\-]?\s*$",
    r"^\s*pathology of the pleura and pleural cavity[:\-]?\s*$",
    r"^\s*pathology of the hilum[:\-]?\s*$",
    r"^\s*abnormal lung parenchyma[:\-]?\s*$",
    r"^\s*pulmonary oedema[:\-]?\s*$",
    r"^\s*pulmonary hypertension[:\-]?\s*$",
    r"^\s*left heart failure[:\-]?\s*$",
    r"^\s*medical devices and other foreign bodies on cxrs[:\-]?\s*$",
    r"^\s*soft tissue asymmetry on cxrs[:\-]?\s*$",
    r"^\s*the silhouette sign[:\-]?\s*$",
    r"^\s*take\-home messages[:\-]?\s*$",
    r"^\s*test your knowledge[:\-]?\s*$",
]


def detect_section_header(line):
    clean = line.strip().lower()
    return any(re.match(p, clean) for p in SECTION_PATTERNS)

# ----------------------------
# PASS 1: SPLIT INTO SECTIONS
# ----------------------------
def split_into_sections(pdf_text):
    sections = []
    current_section = {"title": "Start", "content": ""}

    for line in pdf_text.split("\n"):
        if detect_section_header(line):
            # save previous section
            if current_section["content"].strip():
                sections.append(current_section)

            # start new section
            current_section = {"title": line.strip(), "content": ""}
        else:
            current_section["content"] += line + " "

    sections.append(current_section)
    return sections

# ----------------------------
# PASS 2: SEMANTIC CHUNKING
# ----------------------------
def semantic_chunk(section_text, max_tokens=MAX_TOKENS):
    sentences = nltk.sent_tokenize(section_text)
    chunks = []
    current_chunk = []

    token_count = 0
    for sent in sentences:
        sent_tokens = sent.split()
        if token_count + len(sent_tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            token_count = 0

        current_chunk.append(sent)
        token_count += len(sent_tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ----------------------------
# PASS 3: ADD OVERLAP
# ----------------------------
def add_overlap(chunks, overlap=OVERLAP):
    overlapped = []
    for i, chunk in enumerate(chunks):
        words = chunk.split()
        if i > 0:
            prev_words = chunks[i-1].split()
            chunk = " ".join(prev_words[-overlap:] + words)
        overlapped.append(chunk)
    return overlapped

# ----------------------------
# PROCESS SINGLE PDF
# ----------------------------
def process_pdf(pdf_path):
    print(f"\n📄 Processing: {os.path.basename(pdf_path)}")

    reader = PdfReader(pdf_path)
    all_text = ""

    for page in reader.pages:
        try:
            extracted = page.extract_text() or ""
        except:
            extracted = ""
        all_text += extracted + "\n"

    # PASS 1
    sections = split_into_sections(all_text)

    # For collecting final results
    final_chunks = []

    for sec in sections:
        sec_title = sec["title"]
        sec_text = sec["content"].strip()

        if len(sec_text) < 50:
            continue

        # PASS 2
        semantic_chunks = semantic_chunk(sec_text)

        # PASS 3
        overlapped_chunks = add_overlap(semantic_chunks)

        # store with metadata
        for ch in overlapped_chunks:
            final_chunks.append({
                "section": sec_title,
                "chunk": ch.strip()
            })

    print(f"✅ {len(final_chunks)} chunks created")
    return final_chunks

# ----------------------------
# PROCESS ALL PDFs
# ----------------------------
def process_all():
    all_results = {}
    for file in os.listdir(PDF_FOLDER):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, file)
            chunks = process_pdf(pdf_path)
            all_results[file] = chunks

    return all_results

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    chunks_dict = process_all()
    print("\n🎉 Chunking Completed for All PDFs!")
