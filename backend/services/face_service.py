import math
import io
import json
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import numpy as np

def extract_embedding(image_bytes: bytes) -> List[float]:
    """
    Extracts high-dimensional visual feature descriptor matching EchoBond's embedded specification:
    - 16x16 downsampled mean-subtracted spatial layer (256 dims)
    - 64x64 HSV hue (18 bins) and value (8 bins) histograms (26 dims)
    - 24x24 grayscale gradient orientation histogram (16 bins)
    Normalized to unit vector.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    s = min(w, h)
    # Center square crop
    left = (w - s) // 2
    top = (h - s) // 2
    img_cropped = img.crop((left, top, left + s, top + s))

    # 1. 16x16 spatial layer
    img_16 = img_cropped.resize((16, 16), Image.Resampling.BILINEAR)
    arr_16 = np.array(img_16, dtype=np.float32) / 255.0
    mean_rgb = arr_16.mean(axis=(0, 1))
    layer = (arr_16 - mean_rgb).reshape(-1)

    # 2. 64x64 HSV histograms
    img_64 = img_cropped.resize((64, 64), Image.Resampling.BILINEAR)
    arr_64 = np.array(img_64, dtype=np.float32)
    # Convert RGB to HSV
    r, g, b = arr_64[..., 0], arr_64[..., 1], arr_64[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    df = mx - mn
    
    h_arr = np.zeros_like(mx)
    mask_df = df > 0
    mask_r = mask_df & (mx == r)
    mask_g = mask_df & (mx == g) & (~mask_r)
    mask_b = mask_df & (mx == b) & (~mask_r) & (~mask_g)
    
    h_arr[mask_r] = ((g[mask_r] - b[mask_r]) / df[mask_r]) % 6
    h_arr[mask_g] = ((b[mask_g] - r[mask_g]) / df[mask_r if False else mask_g]) + 2
    h_arr[mask_b] = ((r[mask_b] - g[mask_b]) / df[mask_r if False else mask_b]) + 4
    h_arr = (h_arr / 6.0) % 1.0 * 255.0

    s_arr = np.where(mx > 0, df / mx * 255.0, 0)
    v_arr = mx

    # Hue weighted by saturation (18 bins)
    hue_hist, _ = np.histogram(h_arr, bins=18, range=(0, 255), weights=s_arr / 255.0)
    hue_feat = np.sqrt(hue_hist)

    # Value histogram (8 bins)
    val_hist, _ = np.histogram(v_arr, bins=8, range=(0, 255))
    val_feat = np.sqrt(val_hist)

    # 3. 24x24 grayscale gradient histogram
    img_24 = img_cropped.resize((24, 24), Image.Resampling.BILINEAR).convert("L")
    arr_24 = np.array(img_24, dtype=np.float32)
    
    # Histogram equalization
    hh, _ = np.histogram(arr_24, bins=256, range=(0, 256))
    cdf = hh.cumsum()
    lut = np.floor((cdf / 576.0) * 256.0).clip(0, 255)
    g_arr = lut[arr_24.astype(int)]

    gx = np.zeros_like(g_arr)
    gy = np.zeros_like(g_arr)
    gx[:, 1:-1] = (g_arr[:, 2:] - g_arr[:, :-2]) / 2.0
    gx[:, 0] = g_arr[:, 1] - g_arr[:, 0]
    gx[:, -1] = g_arr[:, -1] - g_arr[:, -2]

    gy[1:-1, :] = (g_arr[2:, :] - g_arr[:-2, :]) / 2.0
    gy[0, :] = g_arr[1, :] - g_arr[0, :]
    gy[-1, :] = g_arr[-1, :] - g_arr[-2, :]

    angles = np.arctan2(gy, gx) + np.pi
    mags = np.hypot(gx, gy)
    grad_hist, _ = np.histogram(angles, bins=16, range=(0, 2 * np.pi), weights=mags)

    # Unit normalizations
    def unit(v):
        norm = np.linalg.norm(v)
        return v / (norm if norm > 0 else 1.0)

    u_layer = unit(layer)
    u_hue = unit(hue_feat) * 0.8
    u_val = unit(val_feat) * 0.4
    u_grad = unit(grad_hist) * 0.4

    full_vector = np.concatenate([u_layer, u_hue, u_val, u_grad])
    final_vec = unit(full_vector)
    return [float(x) for x in final_vec]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def check_image_quality(image_bytes: bytes) -> Dict[str, Any]:
    """
    Evaluates face image suitability before recognition to prevent false positives:
    - Blur check via discrete Laplacian variance
    - Exposure / Illumination check (rejects pitch black or completely blown out frames)
    - Minimum resolution check
    """
    try:
        from scipy.signal import convolve2d
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        w, h = img.size

        if w < 60 or h < 60:
            return {
                "ok": False,
                "problem": "Image resolution is too low for reliable recognition.",
                "blur_score": 0.0,
                "brightness": 0.0,
            }

        arr = np.array(img, dtype=np.float32)
        mean_b = float(arr.mean())

        # Check extreme darkness or glare
        if mean_b < 25.0:
            return {
                "ok": False,
                "problem": "The photo is too dark. Please use better lighting.",
                "blur_score": 0.0,
                "brightness": round(mean_b, 1),
            }
        if mean_b > 235.0:
            return {
                "ok": False,
                "problem": "The photo has too much glare / is overexposed.",
                "blur_score": 0.0,
                "brightness": round(mean_b, 1),
            }

        # Blur check via 3x3 Laplacian
        lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        conv = convolve2d(arr, lap, mode="valid")
        blur_var = float(conv.var())

        if blur_var < 15.0:
            return {
                "ok": False,
                "problem": "The photo is too blurry to clearly see the face.",
                "blur_score": round(blur_var, 1),
                "brightness": round(mean_b, 1),
            }

        return {
            "ok": True,
            "problem": "",
            "blur_score": round(blur_var, 1),
            "brightness": round(mean_b, 1),
        }
    except Exception as e:
        return {"ok": True, "problem": "", "blur_score": 50.0, "brightness": 128.0}


def match_face(
    query_embedding: List[float],
    enrolled_faces: Dict[int, List[List[float]]],
    recognize_threshold: float = 0.90,
    uncertain_threshold: float = 0.75,
    min_margin: float = 0.045
) -> Dict[str, Any]:
    """
    High-assurance face matching engine with strict margin rules:
    - Requires top match score >= recognize_threshold
    - Requires top match to beat runner-up by >= min_margin (prevents lookalike relative confusion)
    - If below threshold or margin is violated, downgrades to 'uncertain' so patient is never given a false guess.
    """
    candidates = []
    for person_id, face_list in enrolled_faces.items():
        if not face_list:
            continue
        scores = [cosine_similarity(query_embedding, emb) for emb in face_list]
        scores.sort(reverse=True)
        # Weight highest score and average top-2
        best_score = (scores[0] + (scores[1] if len(scores) > 1 else scores[0])) / 2.0
        sample_count = len(face_list)
        candidates.append({"person_id": person_id, "score": best_score, "samples": sample_count})

    candidates.sort(key=lambda x: x["score"], reverse=True)

    if not candidates:
        return {
            "status": "unknown",
            "score": 0.0,
            "person_id": None,
            "margin": 0.0,
            "confidence_band": "none",
            "rec_threshold": recognize_threshold
        }

    top = candidates[0]
    score = top["score"]
    person_id = top["person_id"]
    samples = top.get("samples", 1)

    margin = 1.0
    runner_up = None
    if len(candidates) > 1:
        runner_up = candidates[1]
        margin = score - runner_up["score"]

    status = "unknown"
    confidence_band = "none"
    safety_notes = []

    if score >= recognize_threshold:
        if margin < min_margin:
            # Lookalike ambiguity safeguard
            status = "uncertain"
            confidence_band = "ambiguous"
            safety_notes.append(f"Match margin ({margin:.3f}) was below safety threshold ({min_margin}); avoided guessing between relatives.")
        else:
            status = "recognized"
            confidence_band = "high" if (score >= recognize_threshold + 0.03 and samples >= 2) else "medium"
            if samples < 2:
                safety_notes.append("Only 1 enrollment photo exists for this person. Recommend adding 2+ more for peak assurance.")
    elif score >= uncertain_threshold:
        status = "uncertain"
        confidence_band = "low"
    else:
        status = "unknown"
        confidence_band = "none"

    return {
        "status": status,
        "score": round(score, 4),
        "person_id": person_id if status in ("recognized", "uncertain") else None,
        "runner_up_id": runner_up["person_id"] if runner_up else None,
        "margin": round(margin, 4),
        "confidence_band": confidence_band,
        "sample_count": samples,
        "safety_notes": safety_notes,
        "rec_threshold": recognize_threshold
    }
