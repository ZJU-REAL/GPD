# generate_text_desc — 3D assets to per-frame descriptions

Stage 1 of the privilege pipeline (see [`../README.md`](../README.md)). These scripts
convert 3D scene assets into per-frame structured JSON (object labels, depth, image-grid
position, coverage, top-down layout), which Stage 2 renders into the teacher's
`<scene_context>` text.

Like the rest of `data_prep/`, this code is for reference: it expects the original
ScanNet / SPAR / MindCube assets, which are not included in this repository. Input and
output paths are read from environment variables defined at the top of each script.

## Scripts

| Script | Source | Method | Environment |
|--------|--------|--------|-------------|
| `generate_vsi_text_desc.py` | VSI | ScanNet mesh + instance back-projection | `numpy`, `Pillow`, `plyfile`, `tqdm` |
| `generate_spar_text_desc.py` | SPAR | ScanNet mesh + instance back-projection | same as above |
| `generate_mindcube_semantic.py` | MindCube | Grounded-SAM-2 instance masks | SAM 2, GroundingDINO, PyTorch |
| `generate_mindcube_depth_bev.py` | MindCube | Depth Anything 3 depth + BEV | DA3, PyTorch, matplotlib |
| `generate_mindcube_text_desc.py` | MindCube | Merge DA3 depth and SAM 2 masks into JSON | `numpy`, `Pillow`, `tqdm` |

For MindCube, `generate_mindcube_semantic.py` and `generate_mindcube_depth_bev.py` run
first; `generate_mindcube_text_desc.py` combines their outputs.

## Output format

### Per-frame JSON (VSI / SPAR)

```json
{
  "scene_id": "scene0000_00",
  "frame_id": "1004",
  "img_h": 480,
  "img_w": 640,
  "objects": [
    {
      "label": "office chair",
      "inst_id": 3,
      "depth_m": 1.85,
      "cam_right_m": -0.23,
      "cam_fwd_m": 1.85,
      "grid_pos": "center",
      "coverage_pct": 12.4
    }
  ]
}
```

A scene-level `_scene.json` additionally stores the room-compass BEV layout (object
centroids with `compass` direction and `dist_m` from the room centroid).

### Per-image JSON (MindCube)

```json
{
  "img_w": 512,
  "img_h": 512,
  "objects": [
    {
      "label": "bottle",
      "inst_id": 1,
      "depth_rel": 0.312,
      "cam_right": 0.48,
      "cam_up": 0.55,
      "grid_pos": "center",
      "coverage_pct": 8.73
    }
  ]
}
```
