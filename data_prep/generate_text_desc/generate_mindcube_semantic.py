#!/usr/bin/env python3
"""
MindCube instance-segmentation map generation script.

Uses Grounded-SAM-2 (GroundingDINO + SAM 2.1) to perform open-vocabulary
instance segmentation on each MindCube image, producing HSV pseudo-colour
instance maps in the same format as the SPAR/VSI ScanNet pipeline.

Unlike SPAR/VSI (ScanNet 3D back-projection), MindCube relies on 2D visual
detection.  Segmentation results carry semantic labels (e.g. "chair", "bottle")
for use in text-description injection.

Output:
  <OUTPUT_DIR>/<category>/<object>/<stem>.png  - HSV pseudo-colour instance map
  <OUTPUT_DIR>/<category>/<object>/<stem>.json - inst_id -> label mapping

Prerequisites:
  - Grounded-SAM-2 repository cloned and set up (set GSAM2_DIR)
  - SAM 2.1 and GroundingDINO checkpoints downloaded (set SAM2_CHECKPOINT,
    GDINO_CHECKPOINT, or place them at the defaults below)
  - Run in a conda environment that has SAM2, GroundingDINO, and PyTorch installed

Run (from any working directory; the script changes to GSAM2_DIR automatically):
  <your_conda_env>/bin/python3 generate_mindcube_semantic.py \\
      2>&1 | tee logs/mindcube_semantic_gen.log

Environment variables (all optional; defaults shown below):
  GSAM2_DIR          - root of the Grounded-SAM-2 repository
  MINDCUBE_JSONL     - JSONL annotation file for MindCube
  MINDCUBE_SEMANTIC  - output directory for semantic maps
  SAM2_CHECKPOINT    - path to SAM 2.1 checkpoint (.pt)
  GDINO_CONFIG       - path to GroundingDINO config file (.py)
  GDINO_CHECKPOINT   - path to GroundingDINO checkpoint (.pth)
"""

import json
import sys
import os
import colorsys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Add Grounded-SAM-2 to Python path and change working directory ──
GSAM2_DIR = Path(os.environ.get("GSAM2_DIR", "/PATH/TO/Grounded-SAM-2"))
sys.path.insert(0, str(GSAM2_DIR))
# build_sam2 uses hydra to load configs via relative paths, so cwd must be GSAM2_DIR
os.chdir(str(GSAM2_DIR))

# ==================== Configuration ====================
MINDCUBE_JSONL    = Path(os.environ.get("MINDCUBE_JSONL",    "/PATH/TO/WORKSPACE/data/mindcube_10k_grpo.jsonl"))
OUTPUT_DIR        = Path(os.environ.get("MINDCUBE_SEMANTIC", "/PATH/TO/WORKSPACE/data/mindcube_semantic"))

SAM2_CHECKPOINT   = os.environ.get("SAM2_CHECKPOINT",  str(GSAM2_DIR / "sam2/checkpoints/sam2.1_hiera_large.pt"))
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"   # relative to GSAM2_DIR

GDINO_CONFIG      = os.environ.get("GDINO_CONFIG",
    str(GSAM2_DIR / "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"))
GDINO_CHECKPOINT  = os.environ.get("GDINO_CHECKPOINT",
    str(GSAM2_DIR / "gdino_checkpoints/groundingdino_swinb_cogcoor.pth"))

# Open-vocabulary detection prompt covering common indoor objects.
# GroundingDINO requires: lowercase + each phrase ends with "."
TEXT_PROMPT = (
    "chair. sofa. couch. table. desk. shelf. cabinet. drawer. "
    "bed. lamp. monitor. keyboard. mouse. bottle. cup. mug. "
    "plant. vase. box. bag. backpack. suitcase. book. "
    "door. window. wall. floor. ceiling. object. item."
)

BOX_THRESHOLD  = 0.30   # detection confidence threshold (slightly relaxed to capture more objects)
TEXT_THRESHOLD = 0.25
MIN_MASK_AREA  = 200    # masks smaller than this pixel area are discarded (noise removal)
# =======================================================


# ─────────────────────── HSV palette (must match generate_mindcube_text_desc.py) ───

def _build_palette(n: int = 256) -> np.ndarray:
    palette = np.zeros((n, 3), dtype=np.uint8)
    for i in range(1, n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
        palette[i] = (int(r * 255), int(g * 255), int(b * 255))
    return palette

PALETTE = _build_palette(256)


def masks_to_semantic_rgb(
    masks: np.ndarray,
    labels: list,
    img_h: int,
    img_w: int,
) -> tuple:
    """
    Merge (N, H, W) boolean masks into an HSV pseudo-colour instance map and
    produce the inst_id -> label mapping.

    Occlusion handling: larger objects are drawn first so smaller objects can
    overwrite them.  Labels are reordered in sync with masks to avoid mismatch.

    Returns:
        rgb         : (H, W, 3) uint8 pseudo-colour map
        id_to_label : {inst_id: "chair", ...} dict (1-based; 0 = background)
    """
    label_map = np.zeros((img_h, img_w), dtype=np.int32)

    areas = np.array([m.sum() for m in masks])
    order = np.argsort(areas)[::-1]   # largest first

    id_to_label: dict[int, str] = {}
    for inst_id, idx in enumerate(order, start=1):
        mask = masks[idx].astype(bool)
        label_map[mask] = inst_id % 256
        id_to_label[inst_id] = labels[idx]   # keep labels in sync with drawing order

    return PALETTE[label_map], id_to_label


# ─────────────────────── Model loading ───────────────────────

def load_models():
    """Load SAM2 predictor and GroundingDINO model (called once per run)."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.util.inference import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading SAM2: {SAM2_CHECKPOINT}")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print(f"Loading GroundingDINO: {GDINO_CHECKPOINT}")
    gdino_model = load_model(
        model_config_path=GDINO_CONFIG,
        model_checkpoint_path=GDINO_CHECKPOINT,
        device=device,
    )

    print("Models loaded successfully")
    return sam2_predictor, gdino_model, device


# ─────────────────────── Per-image inference ───────────────────────

def process_image(
    sam2_predictor,
    gdino_model,
    device: str,
    img_path: str,
    out_path: Path,
):
    """
    Run Grounded-SAM-2 on a single image and save:
      {stem}.png  - HSV pseudo-colour instance map (inst_id encoded as colour)
      {stem}.json - inst_id -> label mapping for use by generate_mindcube_text_desc.py
    """
    import torch
    from torchvision.ops import box_convert
    from grounding_dino.groundingdino.util.inference import load_image, predict

    json_path = out_path.with_suffix(".json")
    image_source, image_tensor = load_image(img_path)
    h, w = image_source.shape[:2]

    _empty_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # ── GroundingDINO detection ──
    boxes, confidences, labels = predict(
        model=gdino_model,
        image=image_tensor,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=device,
    )

    if boxes.shape[0] == 0:
        Image.fromarray(_empty_rgb).save(str(out_path))
        json_path.write_text("{}", encoding="utf-8")
        return 0

    # Convert box format: cxcywh -> xyxy in pixel coordinates
    boxes_px    = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes_px, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    labels_list = list(labels)   # GroundingDINO returns list[str]

    # ── SAM2 segmentation ──
    sam2_predictor.set_image(image_source)

    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    # masks: (N, 1, H, W) or (N, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)   # -> (N, H, W)

    # Filter out too-small masks; labels must be filtered in sync to avoid index drift
    valid_mask  = np.array([m.sum() >= MIN_MASK_AREA for m in masks])
    masks       = masks[valid_mask]
    labels_list = [l for l, v in zip(labels_list, valid_mask) if v]

    if masks.shape[0] == 0:
        Image.fromarray(_empty_rgb).save(str(out_path))
        json_path.write_text("{}", encoding="utf-8")
        return 0

    # Generate pseudo-colour instance map and retrieve inst_id -> label mapping
    rgb, id_to_label = masks_to_semantic_rgb(masks, labels_list, h, w)

    Image.fromarray(rgb).save(str(out_path))
    # JSON keys must be strings (JSON does not support integer keys)
    json_path.write_text(
        json.dumps({str(k): v for k, v in id_to_label.items()}, ensure_ascii=False),
        encoding="utf-8",
    )
    return masks.shape[0]


# ─────────────────────── Main pipeline ───────────────────────

def main():
    global MINDCUBE_JSONL
    env_j = os.environ.get("MINDCUBE_JSONL")
    if env_j:
        MINDCUBE_JSONL = Path(env_j)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all unique image paths
    unique_images: dict[str, str] = {}   # out_path_str -> img_path
    with open(MINDCUBE_JSONL, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            for img_path in s["image"]:
                parts    = img_path.split("/")
                obj_key  = parts[-3] + "/" + parts[-2]   # e.g. among/bottle_118
                stem     = Path(img_path).stem            # e.g. front_000
                out_path = OUTPUT_DIR / obj_key / f"{stem}.png"
                unique_images[str(out_path)] = img_path

    # Only process images where both PNG and JSON are missing
    pending = [
        (Path(out_p), img_p)
        for out_p, img_p in unique_images.items()
        if not Path(out_p).exists() or not Path(out_p).with_suffix(".json").exists()
    ]

    print(f"Total unique images: {len(unique_images)}  Pending: {len(pending)}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    if not pending:
        print("All images already processed; nothing to do.")
        return

    # Pre-create output directories
    dirs = set(p.parent for p, _ in pending)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    sam2_predictor, gdino_model, device = load_models()

    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    done, total_masks, errors = 0, 0, 0

    for out_path, img_path in tqdm(pending, desc="Images"):
        try:
            n = process_image(sam2_predictor, gdino_model, device, img_path, out_path)
            total_masks += n
            done += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  WARNING: {img_path}: {e}")

    print(f"\nDone! Generated semantic maps: {done}/{len(pending)}  "
          f"Total instances: {total_masks}  Failed: {errors}")


if __name__ == "__main__":
    main()
