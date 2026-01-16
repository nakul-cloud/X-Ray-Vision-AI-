# =====================================================================
# FASTAPI BACKEND FOR X-RAY MEDICAL IMAGING AI
# =====================================================================

import os
import io
import base64
import json
import numpy as np
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim

# Import existing pipeline functions
from xray_rag_full_pipeline import (
    load_xray,
    process_xray,
    process_xray_advanced,
    full_xray_rag_inference,
    enhance_clahe,
    segment_otsu,
    watershed_segmentation,
    compute_intensity_histograms,
    compute_residual_noise,
    compute_local_variance,
    compute_gradient_difference,
    compare_pipelines,
    ENHANCEMENT_METHODS,
    NOISE_METHODS
)
from rag_answer import answer_with_rag

# =====================================================================
# FASTAPI APP INITIALIZATION
# =====================================================================

app = FastAPI(
    title="X-Ray Medical Imaging AI API",
    description="Production-ready API for X-ray analysis with RAG capabilities",
    version="1.0.0"
)

# CORS Configuration - Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def numpy_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64 data URI"""
    if img_array is None:
        return None
    
    # Ensure uint8 type
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    
    # Convert to PIL Image
    if len(img_array.shape) == 2:  # Grayscale
        pil_img = Image.fromarray(img_array, mode='L')
    else:  # RGB
        pil_img = Image.fromarray(img_array, mode='RGB')
    
    # Encode to PNG bytes
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Convert to base64
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def list_to_numpy(img_list) -> np.ndarray:
    """Convert list back to numpy array"""
    return np.array(img_list, dtype=np.uint8)


def create_success_response(data: Any) -> Dict:
    """Create standardized success response"""
    return {
        "success": True,
        "data": data,
        "error": None
    }


def create_error_response(error: str) -> Dict:
    """Create standardized error response"""
    return {
        "success": False,
        "data": None,
        "error": error
    }


# =====================================================================
# API ENDPOINTS
# =====================================================================

def json_friendly(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic): # Handles np.float32, np.int64, etc.
        return obj.item()
    if isinstance(obj, dict):
        return {k: json_friendly(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_friendly(x) for x in obj]
    return obj

def create_success_response(data):
    return {
        "success": True,
        "data": json_friendly(data),
        "error": None
    }
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "X-Ray Medical Imaging AI API",
        "version": "1.0.0"
    }


@app.post("/upload-xray")
async def upload_xray(file: UploadFile = File(...)):
    """
    Upload X-ray image and perform basic processing
    Returns: original image, enhanced image, and basic metrics
    """
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # Process with basic pipeline
        result = process_xray(file_bytes)
        
        # Convert images to base64
        original_img = list_to_numpy(result["original"])
        enhanced_img = list_to_numpy(result["enhanced"])
        
        response_data = {
            "images": {
                "original": numpy_to_base64(original_img),
                "enhanced": numpy_to_base64(enhanced_img)
            },
            "metrics": result["metrics"],
            "filename": file.filename
        }
        
        return create_success_response(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enhancement/{method}")
async def apply_enhancement(method: str, file: UploadFile = File(...)):
    """
    Apply specific enhancement method to X-ray
    Methods: clahe, hist_eq, adaptive_clahe
    """
    try:
        if method not in ENHANCEMENT_METHODS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid method. Choose from: {list(ENHANCEMENT_METHODS.keys())}"
            )
        
        # Read and load image
        file_bytes = await file.read()
        img = load_xray(file_bytes)
        
        # Apply enhancement
        enhanced = ENHANCEMENT_METHODS[method]["function"](img.copy())
        
        response_data = {
            "images": {
                "original": numpy_to_base64(img),
                "enhanced": numpy_to_base64(enhanced)
            },
            "method": {
                "name": ENHANCEMENT_METHODS[method]["name"],
                "description": ENHANCEMENT_METHODS[method]["description"],
                "best_for": ENHANCEMENT_METHODS[method]["best_for"],
                "parameters": ENHANCEMENT_METHODS[method]["parameters"]
            }
        }
        
        return create_success_response(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/noise/{method}")
async def apply_noise_reduction(method: str, file: UploadFile = File(...)):
    """
    Apply noise reduction filter to X-ray
    Methods: gaussian, median, bilateral
    """
    try:
        if method not in NOISE_METHODS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid method. Choose from: {list(NOISE_METHODS.keys())}"
            )
        
        # Read and load image
        file_bytes = await file.read()
        img = load_xray(file_bytes)
        
        # Apply noise reduction
        filtered = NOISE_METHODS[method]["function"](img.copy())
        
        # 1. Compute maps
        residual_data = compute_residual_noise(img, filtered)
        local_variance_map = compute_local_variance(filtered)  # Variance of RESULT
        gradient_diff_map = compute_gradient_difference(img, filtered)
        
        # 2. Compute Metrics
        # Residual stats from detailed analysis
        residual_noise_level = residual_data["mean"]
        
        # Variance Level: Mean of local variance
        variance_level = float(np.mean(local_variance_map))
        
        # Edge Preservation: SSIM between edges
        # OPTIMIZATION: Resize for SSIM speed (use max dim 512)
        h, w = img.shape
        scale = min(1.0, 512 / max(h, w))
        if scale < 1.0:
            img_small = cv2.resize(img, (0,0), fx=scale, fy=scale)
            filt_small = cv2.resize(filtered, (0,0), fx=scale, fy=scale)
            edges_orig = cv2.Canny(img_small, 100, 200)
            edges_filt = cv2.Canny(filt_small, 100, 200)
        else:
            edges_orig = cv2.Canny(img, 100, 200)
            edges_filt = cv2.Canny(filtered, 100, 200)
            
        edge_preservation = float(ssim(edges_orig, edges_filt))

        # OPTIMIZATION: Downsample arrays for visualization to reduce JSON size
        # Transferring 4K arrays via JSON takes seconds. 
        # Visually, 600px is enough for the UI cards.
        def get_display_array(arr, max_dim=600):
            h, w = arr.shape
            if max(h, w) > max_dim:
                s = max_dim / max(h, w)
                return cv2.resize(arr, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA).tolist()
            return arr.tolist()

        response_data = {
            # Return resized arrays for UI speed
            "filtered_image": get_display_array(filtered),
            
            # Primary residual map
            "residual_map": get_display_array(residual_data["normalized_map"]),
            
            # Detailed Residuals
            "residual_normalized": get_display_array(residual_data["normalized_map"]),
            "residual_amplified": get_display_array(residual_data["amplified_map"]),
            
            # Stats (Computed on FULL RESOLUTION image for accuracy)
            "residual_std": residual_data["std"],
            "max_diff": residual_data["max"],
            "mean_diff": residual_data["mean"],
            
            "local_variance_map": get_display_array(local_variance_map),
            "gradient_diff_map": get_display_array(gradient_diff_map),
            
            "metrics": {
                "residual_noise_level": residual_noise_level,
                "variance_level": variance_level,
                "edge_preservation": edge_preservation,
                "residual_std": residual_data["std"],
                "residual_max": residual_data["max"]
            }
        }
        
        return create_success_response(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segmentation")
async def get_segmentation(file: UploadFile = File(...)):
    """
    Perform Otsu and Watershed segmentation
    Returns: masks, edges, and region statistics
    """
    try:
        # Read and load image
        file_bytes = await file.read()
        img = load_xray(file_bytes)
        
        # Enhance first for better segmentation
        enhanced = enhance_clahe(img)
        
        # Otsu segmentation
        otsu_mask, edges = segment_otsu(enhanced)
        
        # Watershed segmentation
        watershed_result = watershed_segmentation(img)
        watershed_mask = list_to_numpy(watershed_result["mask"])
        
        response_data = {
            "images": {
                "original": numpy_to_base64(img),
                "otsu_mask": numpy_to_base64(otsu_mask),
                "edges": numpy_to_base64(edges),
                "watershed_mask": numpy_to_base64(watershed_mask)
            },
            "watershed": {
                "regions": watershed_result["regions"],
                "region_stats": watershed_result["region_stats"]
            }
        }
        
        return create_success_response(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))







@app.post("/advanced")
async def get_advanced_analysis(file: UploadFile = File(...)):
    """
    Get advanced analysis with all enhancement/noise/segmentation stages
    Returns: Complete pixel arrays for frontend histogram computation
    """
    try:
        # Read and load image
        file_bytes = await file.read()
        result = process_xray_advanced(file_bytes)
        
        return create_success_response(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipelines")
async def compare_all_pipelines(file: UploadFile = File(...)):
    """
    Run complete pipeline comparison (all enhancement × noise combinations)
    Returns: all pipelines, best pipeline, rankings, and charts data
    """
    try:
        # Read and load image
        file_bytes = await file.read()
        img = load_xray(file_bytes)
        
        # Run pipeline comparison
        comparison = compare_pipelines(img)
        
        # Convert pipeline output images to base64
        pipelines_with_images = {}
        for pid, pipeline in comparison["all_pipelines"].items():
            pipeline_copy = pipeline.copy()
            # Convert output list to numpy then to base64
            output_img = list_to_numpy(pipeline["output"])
            pipeline_copy["output_image"] = numpy_to_base64(output_img)
            # Remove the large output list
            del pipeline_copy["output"]
            pipelines_with_images[pid] = pipeline_copy
        
        # Also convert best pipeline image
        best_pipeline = comparison["best_pipeline"].copy() if comparison["best_pipeline"] else None
        if best_pipeline:
            best_output = list_to_numpy(best_pipeline["output"])
            best_pipeline["output_image"] = numpy_to_base64(best_output)
            del best_pipeline["output"]
        
        response_data = {
            "images": {
                "original": numpy_to_base64(img)
            },
            "total_pipelines": comparison["total_pipelines"],
            "best_pipeline_id": comparison["best_pipeline_id"],
            "best_pipeline": best_pipeline,
            "all_pipelines": pipelines_with_images,
            "rankings": comparison["rankings"],
            "algorithm_counts": comparison["algorithm_counts"]
        }
        
        return create_success_response(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/analyze")
async def analyze_with_rag(file: UploadFile = File(...)):
    """
    Perform full RAG inference with Mode-B semantic features
    Returns: complete analysis with AI-generated explanation
    """
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # Run full RAG pipeline
        result = full_xray_rag_inference(file_bytes)
        
        # Convert images to base64
        original_img = list_to_numpy(result["basic"]["original"])
        enhanced_img = list_to_numpy(result["basic"]["enhanced"])
        
        response_data = {
            "images": {
                "original": numpy_to_base64(original_img),
                "enhanced": numpy_to_base64(enhanced_img)
            },
            "metrics": result["basic"]["metrics"],
            "semantic_features": result["semantic"],
            "rag_explanation": result["rag_answer"],
            "advanced_analysis": {
                "bone_density": result["advanced"]["bone_density"],
                "quality_metrics": result["advanced"]["metrics_quality"]
            }
        }
        
        return create_success_response(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/followup")
async def rag_followup_question(question: str = Form(...)):
    """
    Ask follow-up question using RAG
    Returns: AI-generated answer based on medical knowledge base
    """
    try:
        # Use RAG to answer question
        answer = answer_with_rag(question, top_k=5, debug=False)
        
        response_data = {
            "question": question,
            "answer": answer
        }
        
        return create_success_response(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export/report")
async def export_report(
    format: str = Form(...),
    data: str = Form(...)
):
    """
    Generate downloadable report in specified format
    Formats: json, csv, txt
    """
    try:
        # Parse data JSON
        report_data = json.loads(data)
        
        if format == "json":
            return JSONResponse(
                content=report_data,
                headers={
                    "Content-Disposition": "attachment; filename=xray_report.json"
                }
            )
        
        elif format == "csv":
            # Generate CSV from metrics
            csv_lines = ["Metric,Value"]
            
            if "metrics" in report_data:
                for key, value in report_data["metrics"].items():
                    csv_lines.append(f"{key},{value}")
            
            csv_content = "\n".join(csv_lines)
            
            return JSONResponse(
                content={"csv": csv_content},
                headers={
                    "Content-Disposition": "attachment; filename=xray_report.csv"
                }
            )
        
        elif format == "txt":
            # Generate text summary
            txt_lines = ["X-RAY ANALYSIS REPORT", "=" * 50, ""]
            
            if "metrics" in report_data:
                txt_lines.append("METRICS:")
                for key, value in report_data["metrics"].items():
                    txt_lines.append(f"  {key}: {value}")
                txt_lines.append("")
            
            if "rag_explanation" in report_data:
                txt_lines.append("AI ANALYSIS:")
                txt_lines.append(report_data["rag_explanation"])
            
            txt_content = "\n".join(txt_lines)
            
            return JSONResponse(
                content={"text": txt_content},
                headers={
                    "Content-Disposition": "attachment; filename=xray_report.txt"
                }
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format. Choose from: json, csv, txt"
            )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# RUN SERVER
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
