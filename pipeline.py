"""
pipeline.py — Full VisionSafe inference pipeline
Adapted from the Colab notebook for production use.
"""

import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from scipy import ndimage

# ── Lazy model loading (loaded once, reused across requests) ──────────────────
_model = None
_device = None
_preprocess = None


def get_model():
    global _model, _device, _preprocess
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VisionSafe] Loading DeepLabV3 on {_device}...")
        # Use MobileNetV3 (lighter) for CPU deployment on Render free tier
        _model = models.segmentation.deeplabv3_mobilenet_v3_large(
            weights="DEFAULT"
        ).to(_device).eval()
        _preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        print("[VisionSafe] Model loaded.")
    return _model, _device, _preprocess


# ── Label IDs (torchvision COCO/VOC) ─────────────────────────────────────────
ROAD = 0
PERSON = 15
CAR, BUS, TRUCK, TRAIN, MOTORCYCLE, BICYCLE = 13, 6, 8, 7, 14, 1
VEHICLE_IDS = [CAR, BUS, TRUCK, TRAIN, MOTORCYCLE, BICYCLE]


# ── Helpers ───────────────────────────────────────────────────────────────────
def count_instances(mask):
    _, n = ndimage.label(mask.astype(np.uint8))
    return int(n)


def region_slices(W):
    t = W // 3
    return {"left": (0, t), "center": (t, 2 * t), "right": (2 * t, W)}


def compute_region_stats(seg, road_mask, obstacle_mask):
    H, W = seg.shape
    regions = region_slices(W)
    stats = {}
    for name, (x0, x1) in regions.items():
        r_obst = obstacle_mask[:, x0:x1]
        r_road = road_mask[:, x0:x1]
        region_area = r_obst.size
        obst_density = float(r_obst.sum() / region_area)
        road_free = float(r_road.sum() / region_area)
        prox_norm = 0.0
        if r_obst.sum() > 0:
            ys, _ = np.where(r_obst > 0)
            if ys.size > 0:
                prox_norm = float(np.clip(1.0 - ((H - ys.max()) / H) ** 1.3, 0, 1))
        stats[name] = {
            "obstacle_density": obst_density,
            "free_road_frac": road_free,
            "proximity_score": prox_norm,
        }
    return stats


def compute_direction(stats):
    risk = {
        k: 0.45 * v["obstacle_density"] + 0.35 * v["proximity_score"] + 0.20 * (1 - v["free_road_frac"])
        for k, v in stats.items()
    }
    best = min(risk, key=risk.get)
    c_risk = risk["center"]
    if c_risk <= 0.42:
        return "straight", risk
    elif risk[best] >= 0.65:
        return "stop", risk
    else:
        return best, risk   # "left" or "right"


def fuzzy_safety_score(road_occ, ped_count, veh_count, unknown_area):
    """Weighted fuzzy scoring (no skfuzzy dependency for deployment)."""
    road_score = max(0, 100 - road_occ)
    ped_score  = max(0, 100 - (ped_count * 5))
    veh_score  = max(0, 100 - abs(veh_count - 5) * 10)
    vis_score  = max(0, 100 - (unknown_area * 30))
    score = 0.2 * road_score + 0.4 * ped_score + 0.3 * veh_score + 0.1 * vis_score
    if ped_count > 10 and veh_count > 8:  score -= 25
    elif ped_count > 8 and veh_count > 5: score -= 15
    elif ped_count > 5 and veh_count > 3: score -= 10
    return float(np.clip(score, 0, 100))


def label_from_score(score):
    if score < 40:   return "Danger"
    elif score < 70: return "Caution"
    else:            return "Safe"


def proximity_to_meters(prox_norm):
    return round(12.0 * (1.0 - prox_norm), 1)


def generate_explanation(label, road_occ, ped_count, veh_count, unknown_area):
    parts = []
    if ped_count > 5:   parts.append(f"{ped_count} pedestrians detected")
    if veh_count > 3:   parts.append(f"{veh_count} vehicles nearby")
    if road_occ < 30:   parts.append("low road visibility")
    if unknown_area > 3: parts.append("obstructed areas in scene")
    if not parts:       parts.append("scene appears open and clear")
    return ", ".join(parts)


def build_narration(label, direction, distance_m, explanation):
    direction_map = {
        "straight": "Continue straight",
        "left": "Turn left",
        "right": "Turn right",
        "stop": "Stop and wait",
    }
    move = direction_map.get(direction, "Proceed with caution")
    if label == "Danger":
        return f"Danger ahead. {explanation}. Stop immediately."
    elif label == "Caution":
        return f"Caution. {explanation}. {move}, approximately {distance_m} meters."
    else:
        return f"Path looks safe. {move}, approximately {distance_m} meters ahead."


def draw_overlay(img_rgb, label, direction, distance_m, confidence):
    vis = img_rgb.copy()
    h, w = vis.shape[:2]

    # Semi-transparent top bar
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (10, 10, 10), -1)
    vis = cv2.addWeighted(overlay, 0.65, vis, 0.35, 0)

    color_map = {"Safe": (80, 220, 100), "Caution": (255, 210, 50), "Danger": (220, 60, 60)}
    color = color_map.get(label, (200, 200, 200))

    # BGR for OpenCV
    bgr = (color[2], color[1], color[0])

    cv2.putText(vis, f"{label}", (30, 52),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, bgr, 3, cv2.LINE_AA)
    cv2.putText(vis, f"{direction.upper()}  |  {distance_m}m  |  {confidence:.0f}% conf",
                (30, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bgr, 2, cv2.LINE_AA)

    # Direction arrow
    cx = w // 2
    arrow_color = bgr
    if direction == "left":
        cv2.arrowedLine(vis, (cx, h - 60), (cx - 160, h // 2 + 40), arrow_color, 8, tipLength=0.3)
    elif direction == "right":
        cv2.arrowedLine(vis, (cx, h - 60), (cx + 160, h // 2 + 40), arrow_color, 8, tipLength=0.3)
    elif direction == "straight":
        cv2.arrowedLine(vis, (cx, h - 60), (cx, h // 2 + 20), arrow_color, 8, tipLength=0.3)
    else:
        cv2.putText(vis, "STOP", (cx - 70, h // 2 + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 0, 220), 6, cv2.LINE_AA)

    # Region risk dividers
    t1, t2 = w // 3, 2 * w // 3
    cv2.line(vis, (t1, 100), (t1, h), (180, 180, 180), 1)
    cv2.line(vis, (t2, 100), (t2, h), (180, 180, 180), 1)

    return vis


# ── Main pipeline entry point ─────────────────────────────────────────────────
def run_full_pipeline(input_image_path: str, output_image_path: str, output_audio_path: str) -> dict:
    model, device, preprocess = get_model()

    # Read & resize
    bgr = cv2.imread(input_image_path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {input_image_path}")
    H0, W0 = bgr.shape[:2]
    target_w = 640
    target_h = int(H0 * (target_w / W0))
    bgr_r = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)

    # Segmentation
    inp = preprocess(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        seg = model(inp)["out"][0].argmax(0).byte().cpu().numpy()

    total_pixels = seg.size
    road_mask     = (seg == ROAD).astype(np.uint8)
    person_mask   = (seg == PERSON).astype(np.uint8)
    vehicle_mask  = np.isin(seg, VEHICLE_IDS).astype(np.uint8)
    obstacle_mask = np.clip(person_mask + vehicle_mask, 0, 1).astype(np.uint8)
    unknown_mask  = np.isin(seg, [2, 3, 4, 5, 9, 10, 11, 12]).astype(np.uint8)

    road_occ    = float((road_mask.sum() / total_pixels) * 100)
    ped_count   = count_instances(person_mask)
    veh_count   = count_instances(vehicle_mask)
    unknown_pct = float((unknown_mask.sum() / total_pixels) * 100)

    # Fuzzy safety score → label
    score = fuzzy_safety_score(road_occ, ped_count, veh_count, unknown_pct)
    label = label_from_score(score)

    # Spatial direction
    stats = compute_region_stats(seg, road_mask, obstacle_mask)
    direction, risk = compute_direction(stats)

    # Distance estimate
    prox_key = f"{direction}_proximity" if direction in ("left", "center", "right") else "center_proximity"
    region_key = direction if direction in stats else "center"
    prox_norm = stats[region_key]["proximity_score"]
    distance_m = proximity_to_meters(prox_norm)

    # Confidence
    confidence = float(np.clip(100 - (unknown_pct * 0.8 + ped_count * 2 + veh_count * 1.5), 20, 100))

    # Explanation + narration
    explanation   = generate_explanation(label, road_occ, ped_count, veh_count, unknown_pct)
    narration_txt = build_narration(label, direction, distance_m, explanation)

    # Annotated image
    annotated_rgb = draw_overlay(rgb, label, direction, distance_m, confidence)
    cv2.imwrite(output_image_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

    # Audio narration
    try:
        from gtts import gTTS
        tts = gTTS(text=narration_txt, lang="en", slow=False)
        tts.save(output_audio_path)
    except Exception as e:
        print(f"[Audio] gTTS failed: {e}. Skipping audio.")
        output_audio_path = None

    return {
        "label": label,
        "direction": direction,
        "distance_m": distance_m,
        "confidence": round(confidence, 1),
        "score": round(score, 2),
        "explanation": explanation,
        "narration_text": narration_txt,
        "road_occupancy": round(road_occ, 2),
        "pedestrian_count": ped_count,
        "vehicle_count": veh_count,
    }
