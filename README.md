# flood-water-segmentation

One-shot water body segmentation for flood monitoring using Personalized SAM with KMeans prompt selection and adaptive window-based sensitivity analysis. Applied to four UK gauging stations.

This repository contains the code for our paper. The method builds upon [PerSAM](https://github.com/ZrrSkywalker/Personalize-SAM) and extends it with a KMeans-based positive/negative prompt selection strategy for robust water body segmentation in flood monitoring scenarios.

## Dataset

The dataset used in this work is publicly available on Zenodo:

> **Download:** [https://doi.org/10.5281/zenodo.18758987](https://doi.org/10.5281/zenodo.18758987) — `dataset.zip` (316.30 MB, MD5: `2c0b02412e99521518f8624c1f77a929`)

The dataset contains four flood monitoring sites: **Tewkesbury**, **DiglisLock**, **Strensham**, and **Evesham**. Each site includes:

| Folder | Description |
|--------|-------------|
| `Images/water/` | Raw camera images (`.jpg`) used as input to the segmentation model |
| `Annotations/water/` | Ground-truth water body masks (`.png`, water pixels = 128) used as reference masks and evaluation labels |
| `neg_Annotations/refine_mask/` | Refined background region masks (`.png`, background pixels = 128) used for negative prompt extraction |

**Setup after download:**

```bash
# 1. Download dataset.zip from Zenodo and place it in the project root
# 2. Unzip and rename the folder to flood_data (the name expected by the code)
unzip dataset.zip
mv dataset flood_data
```

After setup, the directory structure should be:

```
flood_data/
├── Tewkesbury/
│   ├── Images/water/
│   ├── Annotations/water/
│   └── neg_Annotations/refine_mask/
├── DiglisLock/    (same structure)
├── Strensham/     (same structure)
└── Evesham/       (same structure)
```

---

## Requirements

### Installation

Clone this repository and create a conda environment:

```bash
git clone https://github.com/hupi11/flood-water-segmentation.git
cd flood-water-segmentation

conda create -n flood-water-segmentation python=3.8
conda activate flood-water-segmentation

pip install -r requirements.txt
```

### SAM Checkpoint

Download the SAM ViT-H checkpoint and place it in the project root:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Or manually download from: [https://github.com/facebookresearch/segment-anything#model-checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)

The checkpoint file should be placed at the project root:

```
flood-water-segmentation/
├── sam_vit_h_4b8939.pth   ← place here
├── persam_kmeans.py
└── ...
```

## Repository Structure

```
flood-water-segmentation/
├── persam_kmeans.py                  # Main algorithm
├── show.py                           # Visualization utilities
├── per_segment_anything/             # Modified SAM framework (based on PerSAM)
│   ├── predictor.py                  # Extended SamPredictor with attn_sim & target_embedding
│   ├── build_sam.py                  # SAM model builder
│   ├── modeling/                     # SAM model components
│   └── utils/                        # Preprocessing utilities
├── evaluation/
│   └── evaluate.py                   # IoU & Accuracy evaluation
└── requirements.txt
```

---

## Main Algorithm

### `persam_kmeans.py`

The core segmentation script. Key modifications over the original PerSAM:

- **KMeans++ positive prompt selection**: Target foreground features (extracted from the reference mask) are clustered into *n* groups using KMeans++. For each cluster centroid, the most similar point in the test image is selected as a positive prompt, enabling spatially diverse coverage of the water body.
- **KMeans++ negative prompt selection**: Background features are extracted from an explicit negative annotation mask (`neg_Annotations/`) and similarly clustered. The least similar points in the test image are selected as negative prompts, suppressing false positives.
- **Target-guided attention**: An overall similarity map is computed by averaging per-centroid similarity maps, then used as attention guidance (`attn_sim`) in the SAM cross-attention layers.
- **Cascaded post-refinement**: Two-stage mask refinement using multi-mask output and bounding box prompt.

**Usage:**

```bash
# Run on all images in a dataset
python persam_kmeans.py \
    --data ./flood_data/Tewkesbury \
    --outdir results \
    --ref_idx 000 \
    --num_point 6

# Run on a single image
python persam_kmeans.py \
    --data ./flood_data/Tewkesbury \
    --one_data ./flood_data/Tewkesbury/Images/water/005.jpg \
    --outdir results \
    --ref_idx 000 \
    --num_point 6
```

**Data directory structure:**
```
flood_data/
└── <dataset>/
    ├── Images/
    │   └── water/
    │       ├── 000.jpg   ← reference image
    │       ├── 001.jpg
    │       └── ...
    ├── Annotations/
    │   └── water/
    │       ├── 000.png   ← reference mask (water pixels = 128)
    │       └── ...
    └── neg_Annotations/
        └── refine_mask/
            └── 000.png   ← negative region mask (background pixels = 128)
```

---

## Modified SAM Framework — `per_segment_anything/`

The `per_segment_anything/` module is adapted from the [PerSAM](https://github.com/ZrrSkywalker/Personalize-SAM) framework (Zhang et al., 2023). We modified `predictor.py` and `modeling/` to support additional inputs (`attn_sim` and `target_embedding`) required by our method. For full details of the original architecture, please refer to the [PerSAM repository](https://github.com/ZrrSkywalker/Personalize-SAM).

---

## Evaluation

### `evaluation/evaluate.py`

Compute IoU and pixel Accuracy by comparing predicted masks against ground-truth annotations.

```bash
python evaluation/evaluate.py \
    --pred_path <output_dir> \
    --gt_path ./flood_data/<dataset>/Annotations \
    --ref_idx <ref_frame_index>
```

---

## Citation

If you use this code, please also cite the original PerSAM paper:

```bibtex
@article{zhang2023personalize,
  title={Personalize Segment Anything Model with One Shot},
  author={Zhang, Renrui and Jiang, Zhengkai and Guo, Ziyu and Yan, Shilin and Pan, Junting and Dong, Hao and Qiao, Yu and Li, Chunyuan and others},
  journal={arXiv preprint arXiv:2305.03048},
  year={2023}
}
```
