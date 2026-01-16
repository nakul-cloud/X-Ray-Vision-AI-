# prompt_builder.py

"""
Centralized prompt builder for RAG + Groq LLM pipeline.
Generates structured, safe, medical-focused prompts.
"""

def build_context_block(top_chunks):
    """
    Build formatted context text from top RAG chunks.
    """
    if not top_chunks:
        return "No relevant context retrieved."

    formatted = []
    for i, c in enumerate(top_chunks, 1):
        formatted.append(f"[CHUNK {i} - Section: {c.get('section_title','Unknown')}]\n{c['cleaned_text']}\n")

    return "\n".join(formatted)


def build_rag_prompt(user_query, top_chunks):
    """
    Builds a more flexible RAG prompt that avoids constant fallback.
    Allows partial context and prevents empty answers.
    """
    # Build context text
    context = build_context_block(top_chunks)

    return f"""
You are a medical imaging analysis assistant. Use BOTH the user query and any provided RAG knowledge 
to generate a clear, short, technical explanation about X-ray image quality.

IMPORTANT RULES:
1. If context is available, ALWAYS use it primarily.
2. If context is partial or weak, use it anyway and expand logically.
3. Use fallback ONLY when context is truly empty or nonsensical.
4. NEVER answer with only: "No relevant technical context available."
5. If context is partial/noisy, produce a synthetic technical summary using metrics.
6. No medical diagnosis.

USER QUERY:
{user_query}

RETRIEVED KNOWLEDGE (RAG):
{context}

YOUR TASK:
- Combine user metrics + available chunks.
- Generate a concise technical explanation.
- Always produce output unless context is completely missing.

OUTPUT FORMAT:

### Technical Summary
<2–3 sentences summarizing contrast, sharpness, noise, and clarity>

### Key Insights
- Contrast: ...
- Noise: ...
- Detail: ...
- Texture/Entropy: ...

### Based on Available Context
<Explain how the retrieved text relates to the query. If context is partial, say so and still summarize it.>

### Limitations
<Mention if some information was missing or incomplete>
"""




def build_xray_metrics_prompt(metrics, rag_context):
    """
    Creates prompt for explaining CLAHE, Gaussian, Otsu metrics.
    Used inside Streamlit medical analysis.
    """

    return f"""
You are a medical image-processing expert. Explain the following results **in technical, non-clinical terms**.

CONTEXT FROM RAG:
{rag_context[:400]}

METRIC VALUES:
- Contrast Improvement: {metrics['contrast_improvement']:.2f}%
- Original Contrast: {metrics['original_contrast']:.2f}
- Enhanced Contrast: {metrics['enhanced_contrast']:.2f}
- Noise Level: {metrics['noise_level']:.4f}
- Segmentation Quality: {metrics['segmentation_quality']:.4f}
- Bone Density Proxy: {metrics['bone_density_proxy']:.2f}

Produce structured output:

Contrast Enhancement:
<one sentence>

Noise Reduction:
<one sentence>

Segmentation Result:
<one sentence>

Technical Interpretation:
<one sentence>

Limitations:
<one sentence>
"""


def build_image_pipeline_explanation():
    """
    Describes the classical X-ray processing pipeline steps.
    """

    return f"""
Explain the X-ray image processing pipeline in simple clear terms:

Pipeline:
1. Read Image
2. Contrast Enhancement (CLAHE)
3. Noise Reduction (Gaussian blur)
4. Segmentation (Otsu thresholding)
5. Region Extraction
6. Metrics computation

RULES:
- Avoid medical diagnosis
- Explain only image processing
- Use short clear sections
"""


def build_rerank_debug_prompt(query, before, after):
    """
    Optional debugging prompt for evaluating reranker effectiveness.
    """

    return f"""
RERANKING DEBUG REPORT

QUERY:
{query}

TOP RESULTS BEFORE RERANKING:
{before}

TOP RESULTS AFTER RERANKING:
{after}

Explain how reranking changed the ordering based on semantic alignment.
"""


def build_technical_report_prompt(data, rag_context):
    """
    Generates a complete technical report prompt using ALL available pipeline and metric data.
    """
    
    # Extract key data points for the prompt
    best = data.get('best_pipeline', {})
    basic = data.get('basic_metrics', {})
    semantic = data.get('semantic_metrics', {})
    seg = data.get('segmentation', {})
    noise_stats = data.get('noise_heatmap_stats', {})
    rankings = data.get('rankings', [])
    
    # Format rankings for display
    top_3_str = ""
    for i, (pid, score) in enumerate(rankings[:3], 1):
        top_3_str += f"{i}. {pid} (Score: {score:.3f})\n"

    return f"""
You are an X-ray image-processing expert AI. Your job is to produce a **complete technical report** based on the provided analysis data.

---
### INPUT DATA

**1. Best Pipeline Selected:**
- Name: {best.get('name', 'N/A')}
- Enhancement: {best.get('enhancement', 'N/A')}
- Noise Filter: {best.get('noise_filter', 'N/A')}
- Composite Score: {best.get('composite_score', 0):.3f}
- PSNR: {best.get('psnr', 0):.2f} dB
- SSIM: {best.get('ssim', 0):.3f}
- Noise Level: {best.get('noise_level', 0):.3f}
- Contrast: {best.get('contrast', 0):.3f}

**2. Pipeline Rankings (Top 3):**
{top_3_str}

**3. Basic Image Metrics:**
- Mean Intensity: {basic.get('mean_intensity', 0):.2f}
- Std Dev (Noise Distribution): {basic.get('std_intensity', 0):.2f}
- Contrast: {basic.get('contrast', 0):.2f}
- Sharpness: {basic.get('sharpness', 0):.2f}

**4. Semantic Features (Mode-B):**
- Entropy (Texture): {semantic.get('entropy', 0):.3f}
- Gradient Variance (Structural Noise): {semantic.get('gradient_variance', 0):.3f}
- Symmetry Difference: {semantic.get('symmetry_difference', 0):.3f}
- Frequency Energy: {semantic.get('frequency_energy', 0):.3f}

**5. Segmentation Info:**
- Otsu Regions: {seg.get('otsu_region_count', 'N/A')}
- Watershed Regions: {seg.get('watershed_region_count', 'N/A')}

**6. Noise Heatmap Analysis (Best Pipeline vs Original):**
- Residual Noise Mean: {noise_stats.get('residual_mean', 0):.3f} (Filtration difference)
- Local Variance Mean: {noise_stats.get('variance_mean', 0):.3f} (Noisy areas)
- Gradient Difference Mean: {noise_stats.get('gradient_mean', 0):.3f} (Edge impact)

**7. RAG Context (Medical Knowledge):**
{rag_context}

---
### STRICT FORMATTING RULES (CRITICAL)

1. **NO MARKDOWN TABLES**. Do not use `|` pipes or table syntax. Tables break the mobile UI.
2. **CARD-STYLE SECTIONS**. Use the format:
   Metric: <Name>
   Value: <Number>
   Interpretation: <Short explanation>
   ---
3. **SHORT LINES**. Keep lines under 80 characters.
4. **BULLET POINTS ONLY**. Use `•` for lists. No `*` or `-`.
5. **NO SCROLL-BREAKING CONTENT**. No code blocks, no indentation blocks.

---
### REPORT STRUCTURE

### 1. Pipeline Winner Summary
**Pipeline:** {best.get('name', 'N/A')}
**Composite Score:** {best.get('composite_score', 0):.3f}
**Why It Was Chosen:**
• List 2-3 reasons based on PSNR/SSIM.
---

### 2. Pipeline Comparison Breakdown
Compare top steps vertically.
**Comparison:**
• {best.get('enhancement', 'N/A')} vs others
• {best.get('noise_filter', 'N/A')} vs others
• Explain why this combo won.
---

### 3. Detailed Quality Metrics
**Metric:** Mean Intensity
**Value:** {basic.get('mean_intensity', 0):.2f}
**Interpretation:**
• ...
---
**Metric:** Entropy
**Value:** {semantic.get('entropy', 0):.3f}
**Interpretation:**
• ...
---

### 4. Segmentation Analysis
**Masks Generated:**
• Otsu: {seg.get('otsu_region_count', 'N/A')} regions
• Watershed: {seg.get('watershed_region_count', 'N/A')} regions
**Observation:**
• Did noise affect boundaries?
---

### 5. Noise Heatmap Interpretation
**Metric:** Residual Noise
**Value:** {noise_stats.get('residual_mean', 0):.3f}
**Meaning:**
• ...
---
**Metric:** Local Variance
**Value:** {noise_stats.get('variance_mean', 0):.3f}
**Meaning:**
• ...
---

### 6. Final Technical Interpretation
Provide a concise, professional summary of the image quality.
---

### MEDICAL DISCLAIMER
This is a technical image analysis, not a medical diagnosis. Computed values are for signal processing evaluation only.
"""

# For external imports
__all__ = [
    "build_rag_prompt",
    "build_xray_metrics_prompt",
    "build_image_pipeline_explanation",
    "build_rerank_debug_prompt",
    "build_context_block",
    "build_technical_report_prompt"
]
