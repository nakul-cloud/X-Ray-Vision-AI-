# =====================================================================
# X-RAY ADVANCED PIPELINE + HYBRID MODE-B RAG + SCALABLE PIPELINE COMPARISON
# =====================================================================

import os
import io
import cv2
import json
import base64
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import SiglipModel, SiglipProcessor
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from groq import Groq
from supabase import create_client

from rag_answer import answer_with_rag
from prompt_builder import build_rag_prompt

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# MODEL PATHS
# =====================================================================
BGE_MODEL_PATH = r"D:/X_Ray/models/bge-large-en-v1.5"
MEDSIGLIP_PATH = r"D:/X_Ray/models/medsiglip"
RERANKER_PATH = r"D:/X_Ray/models/marco-MiniLM-L-6-v2"
HF_TOKEN = "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"

# =====================================================================
# ALGORITHM DEFINITIONS - SCALABLE SYSTEM
# =====================================================================

# Enhancement Methods
ENHANCEMENT_METHODS = {
    "clahe": {
        "name": "CLAHE",
        "function": lambda img: cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img),
        "description": "CLAHE (clipLimit=3.0, tileGrid=8×8)",
        "best_for": "Local contrast enhancement, bone structures",
        "parameters": {"clip_limit": 3.0, "tile_grid": "8x8"}
    },
    "hist_eq": {
        "name": "Histogram Equalization",
        "function": lambda img: cv2.equalizeHist(img),
        "description": "Global Histogram Equalization",
        "best_for": "Overall contrast improvement, low-contrast images",
        "parameters": {}
    },
    "adaptive_clahe": {
        "name": "Adaptive CLAHE",
        "function": lambda img: cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16)).apply(img),
        "description": "Adaptive CLAHE (clipLimit=4.0, tileGrid=16×16)",
        "best_for": "Fine details, complex textures",
        "parameters": {"clip_limit": 4.0, "tile_grid": "16x16"}
    }
}

# Noise Reduction Methods
NOISE_METHODS = {
    "gaussian": {
        "name": "Gaussian Blur",
        "function": lambda img: cv2.GaussianBlur(img, (5, 5), 1),
        "description": "Gaussian Blur (5×5 kernel, sigma=1)",
        "best_for": "General noise reduction, smooth images",
        "parameters": {"kernel_size": "5x5", "sigma": 1}
    },
    "median": {
        "name": "Median Filter",
        "function": lambda img: cv2.medianBlur(img, 5),
        "description": "Median Filter (kernel size=5)",
        "best_for": "Salt-and-pepper noise removal",
        "parameters": {"kernel_size": 5}
    },
    "bilateral": {
        "name": "Bilateral Filter",
        "function": lambda img: cv2.bilateralFilter(img, 9, 75, 75),
        "description": "Bilateral Filter (d=9, sigmaColor=75, sigmaSpace=75)",
        "best_for": "Edge-preserving noise reduction",
        "parameters": {"d": 9, "sigma_color": 75, "sigma_space": 75}
    }
}

# =====================================================================
# LAZY LOADERS
# =====================================================================
_supabase = None
_text_model = None
_img_model = None
_img_proc = None
_reranker = None


def get_supabase():
    global _supabase
    if _supabase is None:
        _supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    return _supabase


def load_text_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(BGE_MODEL_PATH)
    return _text_model


def load_img_model():
    global _img_model, _img_proc
    if _img_model is None:
        _img_model = SiglipModel.from_pretrained(MEDSIGLIP_PATH, token=HF_TOKEN).to(device)
    if _img_proc is None:
        _img_proc = SiglipProcessor.from_pretrained(MEDSIGLIP_PATH, token=HF_TOKEN)
    return _img_model, _img_proc


def load_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_PATH)
    return _reranker


# =====================================================================
# EMBEDDINGS
# =====================================================================
def embed_text(text: str):
    model = load_text_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_image(img_np: np.ndarray):
    model, proc = load_img_model()
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    from PIL import Image
    pil = Image.fromarray(img_np)
    inputs = proc(images=pil, return_tensors="pt").to(device)

    with torch.no_grad():
        feats = model.get_image_features(**inputs)

    return feats.cpu().numpy().flatten().tolist()


# =====================================================================
# MODE-B SEMANTIC FEATURES
# =====================================================================
def analyze_embedding_mode_b(img_np, embedding):
    fft = np.fft.fftshift(np.fft.fft2(img_np))
    freq = float(np.mean(np.abs(fft)))

    gx = cv2.Sobel(img_np, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img_np, cv2.CV_64F, 0, 1)
    grad = float(np.var(np.sqrt(gx ** 2 + gy ** 2)))

    hist = cv2.calcHist([img_np], [0], None, [256], [0, 256])
    hist_norm = hist / np.sum(hist)
    ent = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-10)))

    h, w = img_np.shape
    left = img_np[:, : w // 2]
    right = np.flip(img_np[:, w - (w // 2):], axis=1)
    sym = float(np.mean(np.abs(left - right)))

    return {
        "frequency_energy": freq,
        "gradient_variance": grad,
        "entropy": ent,
        "symmetry_difference": sym,
        "embedding_dimensions": len(embedding)
    }


# =====================================================================
# RETRIEVAL + RERANK
# =====================================================================
def retrieve_chunks(query, top_k=5):
    supabase = get_supabase()
    vec = embed_text(query)

    result = supabase.rpc(
        "match_chunks",
        {"query_embedding": vec, "match_count": top_k}
    ).execute()

    return result.data or []


def rerank_chunks(query, chunks):
    reranker = load_reranker()
    pairs = [(query, c["cleaned_text"]) for c in chunks]
    scores = reranker.predict(pairs)

    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)

    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)


# =====================================================================
# GROQ LLM WRAPPER
# =====================================================================
class GroqLLM:
    def __init__(self, model="openai/gpt-oss-120b"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def generate(self, prompt):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.2
        )
        return r.choices[0].message.content


llm = GroqLLM()


# =====================================================================
# BASIC PROCESSING
# =====================================================================
def load_xray(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    return img


def enhance_clahe(img, clip_limit=3.0, tile_grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def segment_otsu(img):
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(img, 50, 150)
    return mask, edges


# =====================================================================
# ADVANCED NOISE ANALYSIS FUNCTIONS
# =====================================================================
def compute_residual_noise(original, filtered):
    """
    Compute residual noise with visibility enhancements.
    Returns dictionary with:
        - normalized_map: 0-255 scaled (for visibility)
        - amplified_map: x4 gain (for very low noise)
        - stats: mean, max, std
    """
    # 1. Compute absolute difference (float32 to prevent overflow/clipping during calc)
    diff = cv2.absdiff(original, filtered).astype(np.float32)
    
    # 2. Compute Statistics (on raw difference)
    mean_diff = float(np.mean(diff))
    max_diff = float(np.max(diff))
    std_diff = float(np.std(diff))
    
    # 3. Create Amplified Map (x4 gain) - Good for low noise
    # We clip to 255
    amplified = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
    
    # 4. Create Normalized Map (0-255) - Good for high dynamic range
    if max_diff > 0:
        normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
    else:
        normalized = np.zeros_like(original, dtype=np.uint8)

    return {
        "normalized_map": normalized,
        "amplified_map": amplified,
        "mean": mean_diff,
        "max": max_diff,
        "std": std_diff
    }


def compute_local_variance(img, window_size=7):
    """
    Compute local variance map using sliding window.
    Normalized to 0-255 uint8.
    """
    img_float = img.astype(np.float32)
    
    # Compute local mean
    kernel = (window_size, window_size)
    local_mean = cv2.blur(img_float, kernel)
    
    # Compute local squared mean
    local_sqr_mean = cv2.blur(img_float ** 2, kernel)
    
    # Variance = E[X^2] - (E[X])^2
    variance = local_sqr_mean - (local_mean ** 2)
    
    # Normalize to 0-255 for visualization
    # We clip high variance to make visualization useful
    v_min, v_max = variance.min(), variance.max()
    if v_max - v_min > 0:
        variance_norm = 255 * (variance - v_min) / (v_max - v_min)
    else:
        variance_norm = np.zeros_like(variance)
        
    return variance_norm.astype(np.uint8)


def compute_gradient_difference(original, filtered):
    """
    Compute difference in gradient magnitude (edge sharpness loss).
    Normalized to 0-255 uint8.
    """
    def get_gradient_magnitude(image):
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2)

    grad_orig = get_gradient_magnitude(original)
    grad_filt = get_gradient_magnitude(filtered)
    
    # Difference in gradients (shows where edges differ)
    grad_diff = np.abs(grad_orig - grad_filt)
    
    # Normalize
    g_min, g_max = grad_diff.min(), grad_diff.max()
    if g_max - g_min > 0:
        grad_norm = 255 * (grad_diff - g_min) / (g_max - g_min)
    else:
        grad_norm = np.zeros_like(grad_diff)
        
    return grad_norm.astype(np.uint8)


def analyze_basic(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    return {
        "mean_intensity": float(np.mean(img)),
        "std_intensity": float(np.std(img)),
        "sharpness": float(np.mean(grad)),
        "contrast": float(np.max(img) - np.min(img)),
    }


def process_xray(file_bytes):
    img = load_xray(file_bytes)
    enh = enhance_clahe(img)
    mask, edges = segment_otsu(enh)

    return {
        "original": img.tolist(),
        "enhanced": enh.tolist(),
        "mask": mask.tolist(),
        "edges": edges.tolist(),
        "metrics": analyze_basic(enh)
    }


# =====================================================================
# QUALITY METRICS
# =====================================================================
def compute_quality(original, enhanced):
    mse = float(np.mean((original - enhanced) ** 2))
    psnr = float(20 * np.log10(255 / np.sqrt(mse))) if mse > 0 else 100
    ssim_val = float(ssim(original, enhanced, data_range=255))

    return {"psnr": psnr, "mse": mse, "ssim": ssim_val}


def compute_bone_density(img):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    high = cv2.filter2D(img.astype(np.float32), -1, kernel)
    _, bright = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    return float(np.mean(np.abs(high)[bright > 0]) / 255.0)


def compute_intensity_histograms(img):
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    cdf = np.cumsum(hist) / np.sum(hist) * 100

    return {
        "histogram": hist.tolist(),
        "cumulative": cdf.tolist()
    }


# =====================================================================
# WATERSHED SEGMENTATION
# =====================================================================
def watershed_segmentation(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 2)
    sure_bg = cv2.dilate(opening, kernel, 3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = watershed(dist, markers)

    mask = np.zeros_like(img)
    mask[markers > 1] = 255

    stats = []
    for label in range(2, markers.max() + 1):
        area = int(np.sum(markers == label))
        stats.append({
            "label": label,
            "area": area,
            "percentage": round(area / img.size * 100, 3)
        })

    return {
        "mask": mask.tolist(),
        "markers": markers.tolist(),
        "regions": len(stats),
        "region_stats": stats
    }


# =====================================================================
# SCALABLE PIPELINE GENERATION SYSTEM
# =====================================================================
def compute_noise_level(original, processed):
    """Compute noise level as standard deviation of difference"""
    return float(np.std(original.astype(float) - processed.astype(float)))


def generate_all_pipelines(img: np.ndarray) -> Dict[str, Dict]:
    """
    Generate all possible pipeline combinations (Enhancement → Noise Filter)
    Returns: Dictionary of pipeline_id -> pipeline_info
    """
    pipelines = {}
    
    for enh_key, enh_cfg in ENHANCEMENT_METHODS.items():
        for noise_key, noise_cfg in NOISE_METHODS.items():
            # Create pipeline ID
            pid = f"{enh_key}_then_{noise_key}"
            
            try:
                # Apply enhancement → noise reduction
                enhanced = enh_cfg["function"](img.copy())
                output = noise_cfg["function"](enhanced)
                
                # Ensure output is same shape as input
                if output.shape != img.shape:
                    output = cv2.resize(output, (img.shape[1], img.shape[0]))
                
                # Compute metrics
                contrast = float(np.std(output))
                
                # Compute PSNR
                mse = float(np.mean((img.astype(float) - output.astype(float)) ** 2))
                psnr = float(20 * np.log10(255 / np.sqrt(mse))) if mse > 0 else 100.0
                
                # Compute SSIM
                data_range = float(img.max() - img.min())
                if data_range == 0:
                    data_range = 255.0
                ssim_val = float(ssim(img, output, data_range=data_range))
                
                # Compute noise level
                noise_level = compute_noise_level(img, output)
                
                pipelines[pid] = {
                    "id": pid,
                    "name": f"{enh_cfg['name']} → {noise_cfg['name']}",
                    "enhancement": enh_cfg["name"],
                    "noise_filter": noise_cfg["name"],
                    "enhancement_key": enh_key,
                    "noise_key": noise_key,
                    "enhancement_desc": enh_cfg["description"],
                    "noise_desc": noise_cfg["description"],
                    "enhancement_best_for": enh_cfg.get("best_for", ""),
                    "noise_best_for": noise_cfg.get("best_for", ""),
                    "output": output.tolist(),  # Store as list for JSON serialization
                    "output_array": output,  # Keep numpy array for display
                    "steps": [
                        f"Applied {enh_cfg['name']}: {enh_cfg['description']}",
                        f"Applied {noise_cfg['name']}: {noise_cfg['description']}",
                        "Computed quality metrics"
                    ],
                    "contrast": contrast,
                    "psnr": psnr,
                    "ssim": ssim_val,
                    "noise_level": noise_level,
                    "mse": mse,
                    "composite_score": compute_composite_score(contrast, psnr, ssim_val, noise_level)
                }
                
            except Exception as e:
                print(f"Error in pipeline {pid}: {e}")
                continue
    
    return pipelines


def compute_composite_score(contrast: float, psnr: float, ssim_val: float, noise_level: float) -> float:
    """
    Compute a composite score that balances multiple metrics
    Higher is better
    """
    # Normalize metrics (weights can be adjusted)
    norm_contrast = contrast / 100.0 if contrast > 100 else contrast / 50.0
    norm_psnr = psnr / 100.0 if psnr > 100 else psnr / 50.0
    
    # Weighted sum: contrast is most important, then PSNR, SSIM, low noise
    score = (
        0.4 * norm_contrast +          # Contrast weight
        0.3 * norm_psnr +              # PSNR weight  
        0.2 * ssim_val +               # SSIM weight (already 0-1)
        0.1 * (1.0 - noise_level/255.0)  # Low noise is good
    )
    
    return float(score)


def rank_pipelines(pipelines: Dict[str, Dict]) -> Tuple[str, Dict, List[Tuple[str, float]]]:
    """
    Rank pipelines by composite score and return best one
    Returns: (best_id, best_pipeline, rankings_list)
    """
    if not pipelines:
        return None, None, []
    
    # Create rankings based on composite score
    rankings = []
    for pid, pipeline in pipelines.items():
        rankings.append((pid, pipeline["composite_score"]))
    
    # Sort by composite score (descending)
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    # Get best pipeline
    best_id = rankings[0][0] if rankings else None
    best_pipeline = pipelines.get(best_id) if best_id else None
    
    # Generate explanation for best pipeline
    if best_pipeline:
        best_pipeline["explanation"] = generate_pipeline_explanation(best_pipeline, rankings)
    
    return best_id, best_pipeline, rankings


def generate_pipeline_explanation(pipeline: Dict, rankings: List[Tuple[str, float]]) -> str:
    """Generate human-readable explanation for why this pipeline is best"""
    
    # Get top 3 pipelines for comparison
    top_3 = rankings[:3]
    
    explanation = f"""
**Pipeline {pipeline['name']}** was selected as the best because:

🏆 **Highest Composite Score**: {pipeline['composite_score']:.3f} (vs {top_3[1][1]:.3f} for 2nd, {top_3[2][1]:.3f} for 3rd)

📊 **Performance Metrics:**
- **Contrast**: {pipeline['contrast']:.3f} (detail enhancement)
- **PSNR**: {pipeline['psnr']:.3f} dB (signal quality)
- **SSIM**: {pipeline['ssim']:.3f} (structural similarity)
- **Noise Level**: {pipeline['noise_level']:.3f} (lower is better)

🔧 **Processing Chain:**
1. **{pipeline['enhancement']}**: {pipeline['enhancement_desc']}
2. **{pipeline['noise_filter']}**: {pipeline['noise_desc']}

🎯 **Best For:**
- Enhancement: {pipeline.get('enhancement_best_for', 'General purpose')}
- Noise Reduction: {pipeline.get('noise_best_for', 'General purpose')}

This combination provides optimal balance between detail enhancement and noise reduction for this specific X-ray image.
"""
    
    return explanation


def compare_pipelines(img: np.ndarray) -> Dict:
    """
    Main function to run all pipeline comparisons
    Returns comprehensive comparison results
    """
    # Generate all pipelines
    all_pipelines = generate_all_pipelines(img)
    
    # Rank pipelines
    best_id, best_pipeline, rankings = rank_pipelines(all_pipelines)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_pipelines = {}
    for pid, pipeline in all_pipelines.items():
        # Create a copy without numpy array
        serialized = pipeline.copy()
        if "output_array" in serialized:
            del serialized["output_array"]
        serializable_pipelines[pid] = serialized
    
    # Also serialize best_pipeline
    serializable_best = best_pipeline.copy() if best_pipeline else None
    if serializable_best and "output_array" in serializable_best:
        del serializable_best["output_array"]
    
    return {
        "all_pipelines": serializable_pipelines,
        "best_pipeline_id": best_id,
        "best_pipeline": serializable_best,
        "rankings": rankings,
        "total_pipelines": len(all_pipelines),
        "algorithm_counts": {
            "enhancement_methods": len(ENHANCEMENT_METHODS),
            "noise_methods": len(NOISE_METHODS),
            "total_combinations": len(ENHANCEMENT_METHODS) * len(NOISE_METHODS)
        }
    }


# =====================================================================
# ADVANCED PIPELINE
# =====================================================================
def process_xray_advanced(file_bytes):
    img = load_xray(file_bytes)

    clahe_img = enhance_clahe(img)
    hist_eq = cv2.equalizeHist(img)
    clahe_big = enhance_clahe(img, clip_limit=4.0, tile_grid=(16, 16))

    enhancement_block = {
        "clahe": clahe_img.tolist(),
        "hist_eq": hist_eq.tolist(),
        "adaptive_clahe": clahe_big.tolist()
    }

    gaussian = cv2.GaussianBlur(img, (5, 5), 1)
    median = cv2.medianBlur(img, 5)
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)

    noise_block = {
        "gaussian": gaussian.tolist(),
        "median": median.tolist(),
        "bilateral": bilateral.tolist()
    }

    mask_otsu, edges = segment_otsu(clahe_img)
    water = watershed_segmentation(img)

    quality = compute_quality(img, clahe_img)
    bone = compute_bone_density(img)

    intensity_block = {
        "original": compute_intensity_histograms(img),
        "enhanced": compute_intensity_histograms(clahe_img)
    }

    # Run scalable pipeline comparison
    pipeline_comparison = compare_pipelines(img)

    return {
        "original": img.tolist(),
        "enhanced": clahe_img.tolist(),
        "enhancement": enhancement_block,
        "noise_reduction": noise_block,
        "metrics_quality": quality,
        "bone_density": bone,
        "intensity": intensity_block,
        "segmentation": {
            "otsu": {"mask": mask_otsu.tolist(), "edges": edges.tolist()},
            "watershed": water
        },
        "pipeline_comparison": pipeline_comparison,
        "algorithm_definitions": {
            "enhancement_methods": ENHANCEMENT_METHODS,
            "noise_methods": NOISE_METHODS
        }
    }


# =====================================================================
# FULL HYBRID MODE-B RAG
# =====================================================================
def full_xray_rag_inference(file_bytes):
    # 1. Run Basic & Advanced Analysis
    basic = process_xray(file_bytes)
    adv = process_xray_advanced(file_bytes)

    # 2. Get Original Image (as numpy)
    img_np = np.array(basic["original"], dtype=np.uint8)

    # 3. Mode-B Semantic Analysis
    embedding = embed_image(img_np)
    semantic = analyze_embedding_mode_b(img_np, embedding)

    # 4. Extract Parsing Data for RAG
    # Re-run Best Pipeline to get Image for Heatmap computation
    # (Since 'output' array was removed for serialization)
    best_pipe_info = adv["pipeline_comparison"]["best_pipeline"]
    
    noise_stats = {}
    if best_pipe_info:
        try:
            enh_key = best_pipe_info.get("enhancement_key")
            noise_key = best_pipe_info.get("noise_key")
            
            if enh_key and noise_key:
                # Regenerate Best Output
                enh_func = ENHANCEMENT_METHODS[enh_key]["function"]
                noise_func = NOISE_METHODS[noise_key]["function"]
                
                temp_enh = enh_func(img_np.copy())
                best_output = noise_func(temp_enh)
                
                # Ensure shape match
                if best_output.shape != img_np.shape:
                    best_output = cv2.resize(best_output, (img_np.shape[1], img_np.shape[0]))
                
                # Compute Heatmap Metrics (Stats only, maps not needed for text)
                res = compute_residual_noise(img_np, best_output)
                var = compute_local_variance(best_output)
                grad = compute_gradient_difference(img_np, best_output)
                
                noise_stats = {
                    "residual_mean": res["mean"],
                    "variance_mean": float(np.mean(var)),
                    "gradient_mean": float(np.mean(grad))
                }
        except Exception as e:
            print(f"[WARN] Failed to re-compute heatmaps for best pipeline: {e}")

    # 5. Retrieve Context (Using Metric Summary Query)
    query_summary = f"""
    X-ray Metrics:
    Mean={basic['metrics']['mean_intensity']:.2f}, Std={basic['metrics']['std_intensity']:.2f}
    Contrast={basic['metrics']['contrast']:.2f}, Sharpness={basic['metrics']['sharpness']:.2f}
    Entropy={semantic['entropy']:.3f}, GradVar={semantic['gradient_variance']:.3f}
    Technical explanation of image quality features.
    """
    
    chunks = retrieve_chunks(query_summary)
    reranked = rerank_chunks(query_summary, chunks)
    top_chunks = reranked[:3] if reranked else []
    
    # 6. Build Comprehensive Technical Report Prompt
    # Collect all data
    report_data = {
        "best_pipeline": best_pipe_info,
        "basic_metrics": basic['metrics'],
        "semantic_metrics": semantic,
        "segmentation": {
             "otsu_region_count": len(np.unique(np.array(basic["mask"]))) - 1, # approx
             "watershed_region_count": adv["segmentation"]["watershed"]["regions"] 
        },
        "noise_heatmap_stats": noise_stats,
        "rankings": adv["pipeline_comparison"]["rankings"]
    }
    
    # Build Context Text
    from prompt_builder import build_context_block, build_technical_report_prompt
    rag_context_text = build_context_block(top_chunks)
    
    # Generate Prompt
    final_prompt = build_technical_report_prompt(report_data, rag_context_text)

    # 7. Generate LLM Answer
    final_report = llm.generate(final_prompt)

    return {
        "basic": basic,
        "advanced": adv,
        "semantic": semantic,
        "rag_answer": final_report,
        "noise_stats": noise_stats # Return stats for debugging if needed
    }