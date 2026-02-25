# flood-water-segmentation

One-shot water body segmentation for flood monitoring using Personalized SAM with KMeans prompt selection and adaptive window-based sensitivity analysis. Applied to four UK gauging stations.

This repository contains the code for our paper. The method builds upon [PerSAM](https://github.com/ZrrSkywalker/Personalize-SAM) and extends it with a KMeans-based positive/negative prompt selection strategy for robust water body segmentation in flood monitoring scenarios.

## Dataset

The dataset used in this work is publicly available on Zenodo:

> **Download:** [https://zenodo.org/](https://zenodo.org/) — `dataset.zip` (316.30 MB, MD5: `2c0b02412e99521518f8624c1f77a929`)

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

```bash
pip install torch torchvision opencv-python scikit-learn scipy scikit-posthocs pandas matplotlib seaborn pillow tqdm
```

Download the SAM checkpoint (`sam_vit_h_4b8939.pth`) from the [SAM official repository](https://github.com/facebookresearch/segment-anything).

## Repository Structure

```
MyCode/
├── persam_kmeans.py                  # Main algorithm
├── show.py                           # Visualization utilities
├── per_segment_anything/             # Modified SAM framework (based on PerSAM)
│   ├── predictor.py                  # Extended SamPredictor with attn_sim & target_embedding
│   ├── build_sam.py                  # SAM model builder
│   ├── modeling/                     # SAM model components
│   └── utils/                        # Preprocessing utilities
├── evaluation/                       # Evaluation scripts
│   ├── miou_verife.py
│   ├── miou_verife_iou.py
│   ├── miou_ANOVA.py
│   ├── eval_miou.py
│   ├── persam_coords_verify.py
│   └── batch_run_persam_coords.py
└── regression_data/
    └── caculate.py                   # Water area ratio calculation
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
    --data ./flood_data/camera01/01 \
    --outdir results \
    --ref_idx 000 \
    --num_point 6

# Run on a single image
python persam_kmeans.py \
    --data ./flood_data/camera01/01 \
    --one_data ./flood_data/camera01/01/Images/water/005.jpg \
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

Based on the original [PerSAM](https://github.com/ZrrSkywalker/Personalize-SAM) framework, with the following modifications:

| File | Description |
|------|-------------|
| `predictor.py` | Extended `SamPredictor.predict()` to accept `attn_sim` (target-guided attention map) and `target_embedding` (semantic prompt) as additional inputs |
| `modeling/sam.py` | Modified SAM forward pass to inject `attn_sim` into cross-attention layers |
| `modeling/mask_decoder.py` | Modified mask decoder to incorporate target-semantic prompting |
| `build_sam.py` | Model factory for ViT-H and MobileSAM (ViT-T) |
| `modeling/image_encoder.py` | ViT-based image encoder |
| `modeling/prompt_encoder.py` | Point, box and mask prompt encoder |

---

## Evaluation

### `evaluation/miou_verife.py` / `miou_verife_iou.py`

Compute IoU and pixel Accuracy by comparing predicted masks against ground-truth annotations. Results are grouped by reference image index.

```bash
python evaluation/miou_verife.py \
    --pred_path <output_dir_prefix> \
    --gt_path ./flood_data/<dataset>/Annotations \
    --ref_idx <ref_frame_index>
```

### `evaluation/eval_miou.py`

Evaluate segmentation IoU for a single prediction directory. Useful for quick per-run assessment.

```bash
python evaluation/eval_miou.py \
    --pred_path <output_dir> \
    --gt_path ./flood_data/<dataset>/Annotations \
    --ref_idx <ref_frame_index>
```

### `evaluation/miou_ANOVA.py`

Perform Kruskal-Wallis test and Dunn's post-hoc test (Bonferroni correction) to assess whether the number of prompt points significantly affects segmentation IoU.

```bash
python evaluation/miou_ANOVA.py \
    --pred_path <output_dir_prefix_list> \
    --gt_path ./flood_data/<dataset>/Annotations \
    --ref_idx <ref_idx_list>
```

### `evaluation/persam_coords_verify.py` / `batch_run_persam_coords.py`

Verify the quality and distribution of selected prompt coordinates across datasets. `batch_run_persam_coords.py` automates running the verification across multiple datasets and reference frames.

---

## Water Area Ratio Calculation

### `regression_data/caculate.py`

Calculate the water body area ratio (water pixels / total pixels) from segmentation mask PNG files. Used to generate the water ratio feature for regression-based water level estimation.

```bash
# Edit directory_path inside the script before running
python regression_data/caculate.py
```

Output: a CSV file with columns `image_name` and `water_ratio`.

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
