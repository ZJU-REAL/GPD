# GPD data prep: 3D privilege construction & routing

This directory contains the offline pipeline we used to turn 3D scene assets into the
teacher-side privilege text in the Mixed-15k training data.

> **Reference only.** This code is released to document how the training data was built.
> The source annotations, 3D assets, and intermediate outputs it consumes are not part of
> this repository, so the scripts cannot be run as-is. The final training parquet they
> produce is released on Hugging Face as [GPD-15k](https://huggingface.co/datasets/xinyili0624/GPD-15k)
> (see [`../data/README.md`](../data/README.md)), and training does not require running anything here.

## Pipeline

1. **Stage 1 — Per-frame 3D descriptions** (`generate_text_desc/`)
   Convert 3D assets into per-frame structured JSON (object labels, depth, image-grid
   position, coverage, top-down layout).
   - VSI / SPAR: back-project ScanNet meshes and instance annotations into each RGB frame.
   - MindCube: estimate depth with Depth Anything 3 and object masks with Grounded-SAM-2.
2. **Stage 2 — Routing & assembly** (files in this directory)
   For each question, select the relevant cues among depth / semantic / BEV, render them
   as a `<scene_context>` block, append the `<reference_answer>`, and write parquet.

| File / Directory | Role |
|------------------|------|
| `generate_text_desc/` | Stage 1 — 3D assets → per-frame JSON |
| `text_desc_utils.py` | Render depth / semantic / BEV JSON into `<scene_context>` text |
| `feature_router.py` | Question-conditioned routing over depth / semantic / BEV |
| `privilege_utils.py` | Assemble teacher privilege for each variant (`pure_grpo`, `answer_only`, `text_routed`) |
| `prepare_mixed_variants.py` | Combine samples, routing, and privilege into train/val parquet |
