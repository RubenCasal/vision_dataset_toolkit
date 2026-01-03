# Vision Dataset Toolkit

## Overview

`visionDS_toolkit` is a lightweight toolkit for **loading, inspecting and transforming vision datasets**.

It allows you to:

- Load datasets in **YOLO**, **COCO**, or **COCO-JSON** into a common `DatasetIR` interface.
- Manipulate images, annotations and categories programmatically (remap classes, subset, filter…).
- Convert **between formats** while preserving the `train` / `val` / `test` split structure.
- Generate **PDF dataset reports** with statistics and plots (resolution, boxes, segmentation, etc.).
- Create **visual previews** of annotations to support manual, image-by-image curation.
- Prune datasets based on these previews for **manual visual cleaning**.
- Convert **bounding-box-only datasets → instance segmentation** using **Segment Anything (SAM)**.
- Convert **segmentation datasets → pure detection** by dropping masks and keeping only boxes.
- Download SAM checkpoints programmatically into a standard folder layout.

The goal is to provide a practical toolkit to **speed up dataset analysis and modification workflows** for computer vision experiments.

---

## Installation

#### 1. Create conda environment (Python 3.10)

```
conda create -n visionDS_toolkit_env python=3.10
conda activate visionDS_toolkit_env
```
#### 2. Clone repository and install
```
git clone https://github.com/RubenCasal/vision_dataset_toolkit.git
cd vision_dataset_toolkit
pip install -e .
```
---

## Scripts

### `transform_dataset.py`

**Description**  

Convert a dataset between `yolo`, `coco` and `coco_json` formats, preserving the `train` / `val` / `test` split structure.

**Arguments**

- `--source_path` (required): Path to the source dataset root folder.
- `--source_format` (required): Source format: `yolo`, `coco` or `coco_json`.
- `--dest_path` (required): Path where the converted dataset will be saved.
- `--dest_format` (required): Output format: `yolo`, `coco` or `coco_json`.

**Example**
```
  transform_dataset 
    --source_path ./dataset_formater/dataset_yolo11 
    --source_format yolo 
    --dest_path ./dataset_formater/dataset_coco_json 
    --dest_format coco_json
```


---

### `preview_dataset.py`

**Description**  

Generate visual previews of dataset annotations (YOLO / COCO / COCO JSON) by drawing boxes or masks over images for each split, with optional blur and max-image cap.

**Arguments**

- `--dataset_path` (required): Path to the dataset root folder.
- `--dataset_format` (required): Dataset format: `yolo`, `coco` or `coco_json`.
- `--output_root` (optional): Base folder for previews; if empty, uses `<dataset_root>/preview/<split>`.
- `--blur_radius` (optional): Gaussian blur radius applied to annotation overlays.
- `--max_images` (optional): Max images per split to visualize (`-1` = all).

**Example**
```
 preview_dataset 
    --dataset_path ./dataset_formater/dataset_yolo11 
    --dataset_format yolo 
    --output_root ./dataset_formater/preview_yolo11 
    --blur_radius 3.0 
    --max_images 200
```
---

### `match_preview_folder.py`

**Description**  

Prune a dataset by matching it to a preview folder: only images that are still present in the preview are preserved in the final dataset.
This script is meant to be used after a manual visual review of the preview images (you delete bad / ambiguous samples in the preview, and then match_preview_folder propagates those deletions back to the dataset). An interactive confirmation (yes) is required before applying the pruning.

**Arguments**

- `--dataset_path` (required): Path to the original dataset root folder.
- `--dataset_format` (required): Dataset format: `yolo`, `coco` or `coco_json`.
- `--preview_root` (optional): Base preview folder; if empty, uses `<dataset_root>/preview/<split>`.
- `--output_path` (optional): Output path for the pruned dataset; if empty, uses `<dataset_root>_pruned`.

**Example**
```
 match_preview_folder 
    --dataset_path ./dataset_formater/dataset_yolo11 
    --dataset_format yolo 
    --preview_root ./dataset_formater/preview_yolo11 
    --output_path ./dataset_formater/dataset_yolo11_pruned
```
---

### `get_dataset_report.py`

**Description**  

Generate a PDF report with global statistics for a dataset (YOLO / COCO / COCO JSON), including per-split stats, category distribution and resolution/bbox plots.

**Arguments**

- `--dataset_path` (required): Path to the dataset root folder (parent of `train`, `val`, `test`).
- `--dataset_format` (required): Dataset format: `yolo`, `coco` or `coco_json`.
- `--report_filename` (optional): Output PDF filename (default: `report_dataset.pdf`).
- `--plots_dirname` (optional): Subdirectory inside dataset root to store auxiliary plots.

**Example**
```
 get_dataset_report 
    --dataset_path ./dataset_formater/dataset_yolo11 
    --dataset_format yolo 
    --report_filename report_dataset_yolo11.pdf 
    --plots_dirname report_plots
```
---

### `get_subset_dataset.py`

**Description** 

Create a random subset of a dataset (YOLO / COCO / COCO JSON), keeping a given fraction of images per split while preserving format and structure.

**Arguments**

- `--dataset_path` (required): Path to the dataset root folder.
- `--dataset_format` (required): Dataset format: `yolo`, `coco` or `coco_json`.
- `--keep_percentage` (required): Fraction of images to keep in each split, in `[0, 1]`.
- `--output_path` (optional): Output dataset root; if empty, uses `<dataset_root>_pruned`.

**Example**
```
 get_subset_dataset 
    --dataset_path ./dataset_formater/dataset_yolo11 
    --dataset_format yolo 
    --keep_percentage 0.2 
    --output_path ./dataset_formater/dataset_yolo11_subset_20
```
---

### `remap_labels_dataset.py`

**Description**  

Remap, merge or drop classes in a dataset (YOLO / COCO / COCO JSON) according to an `ID_MAP` defined inside the script, and export the result to a chosen format.

**Arguments**

- `--source_path` (required): Path to the source dataset root folder.
- `--source_format` (required): Source format: `yolo`, `coco` or `coco_json`.
- `--dest_path` (required): Path where the remapped dataset will be saved.
- `--dest_format` (optional): Output format: `yolo`, `coco`, `coco_json` or empty string (`""` = same as source).

Note: The class mapping `ID_MAP` (old_id -> new_id or `None` to drop) is edited directly in the script.

#### Class remapping (ID_MAP)
The remapping dictionary is edited directly in
visionDS_toolkit/scripts/remap_labels_dataset.py.
It maps **old category IDs → new category IDs**, or None if a class must be removed.

**Example**
```
ID_MAP: Dict[int, Optional[int]] = {
    0: 0,    # car    -> vehicle (new class 0)
    1: 0,    # truck  -> vehicle (merged into class 0)
    2: 1,    # person -> person (new class 1)
    3: None, # tree   -> removed from the dataset
}
```
- All annotations with category_id == 0 or 1 will become category_id == 0 (“vehicle”).
- All annotations with category_id == 2 will become category_id == 1 (“person”).
- All annotations with category_id == 3 will be dropped.

**Example**
```
  remap_labels_dataset 
    --source_path ./dataset_formater/dataset_yolo11 
    --source_format yolo 
    --dest_path ./dataset_formater/dataset_yolo11_remapped 
    --dest_format yolo
```
---
### `segmentation2detection.py`

**Description**

Convert a segmentation dataset (with polygon masks) into a pure detection dataset by dropping mask polygons and keeping only bounding boxes, preserving splits and target format.

**Arguments**

`--dataset_path` (required): Path to the segmented dataset root folder.

`--dataset_format` (required): Input format: yolo, coco or coco_json.

`--dest_path` (required): Path where the detection-only dataset will be saved.

`--dest_format` (required): Output format: yolo, coco or coco_json.

**Example**
```
segmentation2detection \
  --dataset_path ./dataset_formater/data_segmented \
  --dataset_format coco \
  --dest_path ./dataset_formater/data_detection_only \
  --dest_format yolo
```
---
### `download_sam_checkpoint.py`

**Description**

Download a Segment Anything (SAM) checkpoint (vit_h, vit_l, or vit_b) into dataset_formater/sam_checkpoints (or a custom folder), with a friendly filename (sam_huge.pth, sam_large.pth, sam_base.pth).

**Arguments**

`--model-type` (optional): SAM variant to download: vit_h, vit_l, or vit_b (default: vit_l → sam_large.pth).

`--output-dir` (optional): Custom output directory; if empty, uses <repo_root>/dataset_formater/sam_checkpoints.

**Example**
```
download_sam_checkpoint \
  --model-type vit_l \
  --output-dir ./dataset_formater/sam_checkpoints
```

---
### `transform2segmentation.py`

**Description**  
Convert a detection dataset with bounding boxes (YOLO / COCO / COCO JSON) into an instance segmentation dataset by generating masks per box using Segment Anything (SAM).

**Arguments**

- `--dataset_path` (required): Path to the dataset root folder.
- `--dataset_format` (required): Input format: `yolo`, `coco` or `coco_json`.
- `--dest_path` (required): Path where the segmentation dataset will be saved (for example COCO JSON with `segmentation` filled).
- `--dest_format` (required): Output format: `yolo`, `coco` or `coco_json`.
- `--sam_checkpoint` (required): Path to the SAM `.pth` checkpoint (for example `dataset_formater/sam_checkpoints/sam_vit_l_0b3195.pth`).
- `--sam_model_type` (optional): SAM backbone type: `vit_h`, `vit_l`, or `vit_b` (default: `vit_l`).
- `--sam_device` (optional): Device for inference: `cuda` or `cpu` (default: `cuda`).
- `--score_threshold` (optional): Minimum SAM mask score to accept a prediction.

**Example**
```
  transform2segmentation 
    --dataset_path ./dataset_formater/dataset_yolo11 
    --dataset_format yolo 
    --dest_path ./dataset_formater/dataset_segmented_coco 
    --dest_format coco_json 
    --sam_checkpoint dataset_formater/sam_checkpoints/sam_vit_l_0b3195.pth 
    --sam_model_type vit_l 
  
```
