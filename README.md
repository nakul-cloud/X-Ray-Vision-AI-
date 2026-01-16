# 🩻 X-Ray Vision AI (Medicax)

**Technical X-Ray Image Analysis with Metric-Driven RAG & Explainable AI**

X-Ray Vision AI (Medicax) is an advanced medical imaging analysis platform designed to evaluate X-ray image quality, preprocessing pipelines, and structural visibility using classical computer vision, quantitative metrics, and Retrieval-Augmented Generation (RAG).

---

## ⚠️ Important Disclaimer

**This system does NOT perform medical diagnosis.**
It provides technical, image-processing–level interpretation only, intended to support radiologists, researchers, and students in understanding image quality, enhancement effectiveness, noise behavior, and segmentation reliability.

### 🎯 What This System Is Designed For

**✔ Technical & Educational Use**
- Image quality assessment
- Pipeline comparison & explainability
- Noise & artifact analysis
- Segmentation reliability analysis
- Research, education, and technical validation

**❌ NOT For Clinical Use**
- Disease diagnosis
- Clinical decision making
- Pathology detection

---

## 🧠 Core Capabilities

### 🔬 Image Processing & Metrics
- **Contrast enhancement evaluation**
- **Noise reduction effectiveness analysis**
- **Edge preservation assessment**
- **Texture & frequency analysis**
- **Bone-structure visibility proxies**

### 🧪 Pipeline Comparison Engine
- Automatically evaluates multiple enhancement + denoising pipelines
- Ranks pipelines using composite score
- Explains why one pipeline outperforms others

### 📊 Explainable Metrics
- **PSNR, SSIM, MSE**
- **Noise level** (difference statistics)
- **Gradient variance** (sharpness)
- **Entropy** (texture complexity)
- **Symmetry difference**
- **Residual noise statistics**

### 🧠 RAG-Powered Technical Reporting
- Retrieves image-processing knowledge from PDFs
- Combines retrieved context + live metrics
- Produces structured, technical explanations
- **Enforced non-diagnostic output**

---

## 🧱 System Architecture

### Frontend (Antigravity UI)
- **Pure HTML / CSS / Vanilla JS**
- No heavy frameworks
- Responsive layout for:
  - X-ray visualization
  - Histogram plots
  - Heatmaps
  - AI reports
- Communicates via REST APIs

### Backend (FastAPI – Python)
- Image processing (OpenCV, scikit-image)
- Metric computation
- Pipeline ranking engine
- RAG orchestration
- LLM integration

### Database (Supabase)
- Vector store for RAG
- Stores:
  - PDF chunks
  - Text embeddings
  - Metadata

---

## 🧠 Models Used (Actual, In-Use)

### 📘 Text Embedding Model (RAG)
- **BGE-Large-EN-v1.5**
- Library: `sentence-transformers`
- **Purpose:**
  - Embedding PDF chunks
  - Embedding user queries
- **Optimized for:**
  - Technical text
  - Long documents
  - High-quality retrieval
- Stored in Supabase (pgvector)

### 🖼️ Image Embedding Model
- **MedSigLIP**
- Vision-language model
- **Used for:**
  - Image semantic feature extraction
  - Frequency, entropy, symmetry context
- **NOT used for diagnosis**
- Supports Mode-B semantic descriptors

### 🧠 Reranker Model
- **MS-MARCO MiniLM Cross-Encoder**
- **Purpose:**
  - Rerank retrieved RAG chunks
  - Improve relevance before LLM generation

### 🤖 Large Language Model (LLM)
- **Groq API** (Llama / Mixtral family)
- **Used ONLY for:**
  - Converting metrics + retrieved context into readable technical explanations
- **Strict prompt rules enforce:**
  - ❌ No diagnosis
  - ❌ No medical claims
  - ✅ Metric-based reasoning only

---

## 🔁 Full Processing Workflow

### 1️⃣ Image Upload
- User uploads X-ray image (PNG/JPG)
- Sent to backend via REST API

### 2️⃣ Enhancement & Denoising
- Multiple pipelines are executed automatically:

  **Enhancement Methods**
  - CLAHE (8×8, 16×16)
  - Histogram Equalization

  **Noise Reduction**
  - Gaussian Filter
  - Median Filter
  - Bilateral Filter

  Each combination is evaluated independently.

### 3️⃣ Segmentation
- Otsu Thresholding (bone separation)
- Watershed Segmentation (region separation)
- Region statistics extracted

### 4️⃣ Metric Computation
- For each pipeline:
  - Contrast (std deviation)
  - PSNR / SSIM / MSE
  - Noise level
  - Gradient sharpness
  - Composite score (weighted)

### 5️⃣ Pipeline Ranking
- All pipelines ranked
- Best pipeline selected
- Explanation generated:
  - Why it performed best
  - Trade-offs involved

### 6️⃣ RAG (Retrieval-Augmented Generation)
- **This is NOT diagnosis.**
- Metrics are summarized into a technical query
- Query is embedded using BGE
- Relevant PDF chunks retrieved from Supabase
- Reranker selects top context
- LLM generates:
  - Image Quality Summary
  - Pipeline Explanation
  - Noise & Segmentation Interpretation
  - Technical Limitations

### 7️⃣ Visualization
- Frontend displays:
  - Original vs enhanced images
  - Noise & residual heatmaps
  - Histograms
  - Structured technical report

---
 
