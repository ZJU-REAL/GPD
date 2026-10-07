# Data

The Mixed-15k training data (VSI 10k + SPAR 4k + MindCube 1k) is hosted on Hugging Face: [xinyili0624/GPD-15k](https://huggingface.co/datasets/xinyili0624/GPD-15k).

Download it into this directory from the repository root:

```bash
hf download xinyili0624/GPD-15k --repo-type dataset --local-dir data --exclude README.md
cd data && unzip -q frames.zip && rm frames.zip
```

Expected layout:

```
data/
├── mixed_15k_privfix/          # train/val parquet (3 variants) + manifest.json
│   ├── pure_grpo_{train,val}.parquet
│   ├── answer_only_{train,val}.parquet
│   └── text_routed_{train,val}.parquet
└── frames/                     # RGB frames referenced by parquet
    ├── <scene>/frame_XX.jpg                 # VSI (ScanNet)
    ├── spar/<scene>/image_color/<id>.jpg    # SPAR
    └── mindcube/<among|around|rotation>/... # MindCube
```

The parquet column `images` stores paths relative to `data/frames/`. The training scripts read `FRAMES_DIR` (default `<repo>/data/frames`) and `DATA_DIR_PRIVFIX` (default `<repo>/data/mixed_15k_privfix`); override them if you store the data elsewhere.
