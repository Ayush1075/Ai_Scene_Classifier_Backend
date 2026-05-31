"""
pipeline.py — VisionSafe inference pipeline (RAM-optimised for Render free tier)

Memory budget: ~450 MB total
  - deeplabv3_mobilenet_v3_large weights: ~45 MB
  - inference tensors (640px input): ~80 MB
  - opencv + numpy overhead: ~60 MB
  ─────────────────────────────────────────
  Total: ~185 MB  ✓ fits in 512 MB free tier
"""

import os
import gc
import cv2
import numpy as np
import logging

logger = logging.getLogger("visionsafe")

# ── Lazy model singleton ──────────────────────────────────────────────────────
_model      = None
_preprocess = None


def get_model():
    global _model, _preprocess
    if _model is not None:
        return _model, _preprocess

    import torch
    from torchvision import models, transforms

    logger.info("Loading DeepLabV3-MobileNetV3 (CPU)...")
    _model = models.segmentation.deeplabv3_mobilenet_v3_large(
        weights="DEFAULT"
    ).cpu().eval()

    _preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    logger.info("Model ready.")
    return _model, _preprocess


# ── COCO/VOC class IDs ────────────────────────────────────────────────────────
ROAD        = 0
PERSON      = 15
VEHICLE_IDS = [6, 7, 8, 13, 14, 1]   # bus, train, truck, car, motorbike, bicycle


# ── Helpers ───────────────────────────────────────────────────────────────────
def count_blobs(mask):
    from scipy import ndimage
    _, n = ndimage.label(mask.astype(np.uint8))
    return int(n)


def compute_region_stats(road_mask, obstacle_mask):
    H, W  = road_mask.shape
    third = W // 3
    cuts  = {"left": (0, third), "center": (third, 2*third), "right": (2*third, W)}
    stats = {}
    for name, (x0, x1) in cuts.items():
        ro  = obstacle_mask[:, x0:x1]
        rr  = road_mask[:, x0:x1]
        sz  = ro.size
        od  = float(ro.sum() / sz)
        rf  = float(rr.sum() / sz)
        prox = 0.0
        if ro.sum() > 0:
            ys, _ = np.where(ro > 0)
            if ys.size > 0:
                prox = float(np.clip(1.0 - ((H - ys.max()) / H) ** 1.3, 0, 1))
        stats[name] = {"obstacle_density": od, "free_road_frac": rf, "proximity_score": prox}
    return stats


def decide_direction(stats):
    risk = {
        k: 0.45*v["obstacle_density"] + 0.35*v["proximity_score"] + 0.20*(1-v["free_road_frac"])
        for k, v in stats.items()
    }
    best = min(risk, key=risk.get)
    if risk["center"] <= 0.42:
        return "straight", risk
    if risk[best] >= 0.65:
        return "stop", risk
    return best, risk


def safety_score(road_occ, ped, veh, unk):
    s = (0.2 * max(0, 100 - road_occ)
       + 0.4 * max(0, 100 - ped * 5)
       + 0.3 * max(0, 100 - abs(veh - 5) * 10)
       + 0.1 * max(0, 100 - unk * 30))
    if ped > 10 and veh > 8:  s -= 25
    elif ped > 8 and veh > 5: s -= 15
    elif ped > 5 and veh > 3: s -= 10
    return float(np.clip(s, 0, 100))


def label_from(score):
    if score < 40:   return "Danger"
    if score < 70:   return "Caution"
    return "Safe"


def prox_to_meters(p):
    return round(12.0 * (1.0 - p), 1)


def explain(label, road_occ, ped, veh, unk):
    parts = []
    if ped  > 5:  parts.append(f"{ped} pedestrians detected")
    if veh  > 3:  parts.append(f"{veh} vehicles nearby")
    if road_occ < 30: parts.append("low road visibility")
    if unk  > 3:  parts.append("obstructed areas in scene")
    return ", ".join(parts) if parts else "scene appears open and clear"


def narrate(label, direction, dist, expl):
    mv = {"straight": "Continue straight", "left": "Turn left",
          "right": "Turn right", "stop": "Stop and wait"}.get(direction, "Proceed with caution")
    if label == "Danger":
        return f"Danger ahead. {expl}. Stop immediately."
    if label == "Caution":
        return f"Caution. {expl}. {mv}, approximately {dist} meters."
    return f"Path looks safe. {mv}, approximately {dist} meters ahead."


def draw_overlay(img_rgb, label, direction, dist, conf):
    vis = img_rgb.copy()
    h, w = vis.shape[:2]
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (10, 10, 10), -1)
    vis = cv2.addWeighted(overlay, 0.65, vis, 0.35, 0)

    colors = {"Safe": (80,220,100), "Caution": (255,210,50), "Danger": (220,60,60)}
    rgb = colors.get(label, (200,200,200))
    bgr = (rgb[2], rgb[1], rgb[0])

    cv2.putText(vis, label, (30, 52), cv2.FONT_HERSHEY_DUPLEX, 1.6, bgr, 3, cv2.LINE_AA)
    cv2.putText(vis, f"{direction.upper()}  |  {dist}m  |  {conf:.0f}% conf",
                (30, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bgr, 2, cv2.LINE_AA)

    cx = w // 2
    if direction == "left":
        cv2.arrowedLine(vis, (cx, h-60), (cx-160, h//2+40), bgr, 8, tipLength=0.3)
    elif direction == "right":
        cv2.arrowedLine(vis, (cx, h-60), (cx+160, h//2+40), bgr, 8, tipLength=0.3)
    elif direction == "straight":
        cv2.arrowedLine(vis, (cx, h-60), (cx, h//2+20), bgr, 8, tipLength=0.3)
    else:
        cv2.putText(vis, "STOP", (cx-70, h//2+20),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (0,0,220), 6, cv2.LINE_AA)

    for x in [w//3, 2*w//3]:
        cv2.line(vis, (x, 100), (x, h), (180,180,180), 1)

    return vis


# ── Entry point ───────────────────────────────────────────────────────────────
def run_full_pipeline(input_path: str, output_img: str, output_audio: str) -> dict:
    import torch

    model, preprocess = get_model()

    # Read + resize to 480px wide (saves ~40% RAM vs 640)
    bgr = cv2.imread(input_path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {input_path}")

    H0, W0 = bgr.shape[:2]
    tw = 480
    th = int(H0 * tw / W0)
    bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Segmentation (CPU, no grad)
    inp = preprocess(rgb).unsqueeze(0)
    with torch.no_grad():
        seg = model(inp)["out"][0].argmax(0).byte().cpu().numpy()

    # Free tensor memory immediately
    del inp
    gc.collect()

    total  = seg.size
    road_m = (seg == ROAD).astype(np.uint8)
    pers_m = (seg == PERSON).astype(np.uint8)
    veh_m  = np.isin(seg, VEHICLE_IDS).astype(np.uint8)
    obst_m = np.clip(pers_m + veh_m, 0, 1).astype(np.uint8)
    unk_m  = np.isin(seg, [2,3,4,5,9,10,11,12]).astype(np.uint8)

    road_occ = float(road_m.sum() / total * 100)
    ped      = count_blobs(pers_m)
    veh      = count_blobs(veh_m)
    unk_pct  = float(unk_m.sum() / total * 100)

    score     = safety_score(road_occ, ped, veh, unk_pct)
    lbl       = label_from(score)
    stats     = compute_region_stats(road_m, obst_m)
    direction, risk = decide_direction(stats)

    region_key = direction if direction in stats else "center"
    dist       = prox_to_meters(stats[region_key]["proximity_score"])
    conf       = float(np.clip(100 - (unk_pct*0.8 + ped*2 + veh*1.5), 20, 100))
    expl       = explain(lbl, road_occ, ped, veh, unk_pct)
    narr       = narrate(lbl, direction, dist, expl)

    # Annotated image
    ann = draw_overlay(rgb, lbl, direction, dist, conf)
    cv2.imwrite(output_img, cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))

    # Audio (gTTS calls Google's API — lightweight)
    try:
        from gtts import gTTS
        gTTS(text=narr, lang="en", slow=False).save(output_audio)
    except Exception as e:
        logger.warning(f"gTTS failed: {e}")

    return {
        "label": lbl,
        "direction": direction,
        "distance_m": dist,
        "confidence": round(conf, 1),
        "score": round(score, 2),
        "explanation": expl,
        "narration_text": narr,
        "road_occupancy": round(road_occ, 2),
        "pedestrian_count": ped,
        "vehicle_count": veh,
    }
