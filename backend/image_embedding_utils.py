import torch
from transformers import SiglipModel, SiglipProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

img_model = SiglipModel.from_pretrained(
    "D:/X_Ray/models/medsiglip",
    token="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"
).to(device)

img_processor = SiglipProcessor.from_pretrained(
    "D:/X_Ray/models/medsiglip",
    token="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",
    use_fast=True
)

import cv2
import numpy as np

def embed_image(image):
    """
    Generates embedding for X-Ray image using MedSigLIP.
    ✔ Auto-converts Grayscale -> RGB
    ✔ Handles float32/uint16
    ✔ Resizes huge images (>1024px)
    ✔ Robust error handling
    """
    # Fallback size for MedSigLIP (usually 768)
    # If we don't know, we return empty or try to infer.
    # We'll use 768 as standard SigLIP.
    dim = 768
    
    try:
        # 1. Validation
        if image is None or image.size == 0:
            return [0.0] * dim

        # 2. Conversion to uint8
        if image.dtype != np.uint8:
            # If float (0-1), scale to 255
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                # If uint16 or arbitrary float, normalize
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 3. Grayscale -> RGB (SigLIP expects 3 channels)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
             # RGBA -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # 4. Resize (Prevent OOM)
        h, w = image.shape[:2]
        if max(h, w) > 1024:
            s = 1024 / max(h, w)
            image = cv2.resize(image, (0, 0), fx=s, fy=s)

        # 5. Inference
        inputs = img_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            features = img_model.get_image_features(**inputs)
            
        emb_list = features.cpu().numpy().flatten().tolist()
        return emb_list

    except Exception as e:
        print(f"[ERROR] embed_image failed: {e}")
        return [0.0] * dim

def safe_embed_image(image):
    return embed_image(image)
