# Dataset Toolkit

## Overview

`dataset_formater` is designed to help you:

- Load datasets in YOLO, COCO, or COCO-JSON format into a common `DatasetIR` interface.
- Manipulate images, annotations and categories programmatically (remap classes, subset, filter, etc.).
- Export back to standard formats so the processed dataset is ready for training.
- Generate automatic dataset reports (PDF) with statistics and plots.
- Convert bounding-box-only datasets into instance segmentation datasets using Segment Anything (SAM).

The goal is to centralize typical dataset housekeeping tasks in a single, consistent toolkit.

---

## Installation

(To be completed: pip install / editable install instructions.)

---

## Scripts

### `transform_dataset.py`

Description  
Convert a dataset between `yolo`, `coco` and `coco_json` formats, preserving the `train` / `val` / `test` split structure.

Arguments

- `--source_path` (required): Path to the source dataset root folder.
- `--source_format` (required): Source format: `yolo`, `coco` or `coco_json`.
- `--dest_path` (required): Path where the converted dataset will be saved.
- `--dest_format` (required): Output format: `yolo`, `coco` or `coco_json`.

Example
```python
  python -m dataset_formater.scripts.transform_dataset 
    --source_path ./dataset_formater/dataset_yolo11 
    --source_format yolo ^
    --dest_path ./dataset_formater/dataset_coco_json 
    --dest_format coco_json
```


---

### `preview_dataset.py`

Description  
Generate visual previews of dataset annotations (YOLO / COCO / COCO JSON) by drawing boxes or masks over images for each split, with optional blur and max-image cap.

Arguments

- `--dataset_path` (required): Path to the dataset root folder.
- `--dataset_format` (required): Dataset format: `yolo`, `coco` or `coco_json`.
- `--output_root` (optional): Base folder for previews; if empty, uses `<dataset_root>/preview/<split>`.
- `--blur_radius` (optional): Gaussian blur radius applied to annotation overlays.
- `--max_images` (optional): Max images per split to visualize (`-1` = all).

Example
```python
  python -m dataset_formater.scripts.preview_dataset 
    --dataset_path ./dataset_formater/dataset_yolo11 
    --dataset_format yolo 
    --output_root ./dataset_formater/preview_yolo11 
    --blur_radius 3.0 
    --max_images 200
```
---

### `match_preview_folder.py`

Description  
Prune a dataset by matching it to a preview folder: only images that remain in the preview are preserved in the final dataset, after an interactive confirmation (`yes` is required before deletion).

Arguments

- `--dataset_path` (required): Path to the original dataset root folder.
- `--dataset_format` (required): Dataset format: `yolo`, `coco` or `coco_json`.
- `--preview_root` (optional): Base preview folder; if empty, uses `<dataset_root>/preview/<split>`.
- `--output_path` (optional): Output path for the pruned dataset; if empty, uses `<dataset_root>_pruned`.

Example
```python
  python -m dataset_formater.scripts.match_preview_folder 
    --dataset_path ./dataset_formater/dataset_yolo11 
    --dataset_format yolo 
    --preview_root ./dataset_formater/preview_yolo11 
    --output_path ./dataset_formater/dataset_yolo11_pruned
```
---

### `get_dataset_report.py`

Description  
Generate a PDF report with global statistics for a dataset (YOLO / COCO / COCO JSON), including per-split stats, category distribution and resolution/bbox plots.

Arguments

- `--dataset_path` (required): Path to the dataset root folder (parent of `train`, `val`, `test`).
- `--dataset_format` (required): Dataset format: `yolo`, `coco` or `coco_json`.
- `--report_filename` (optional): Output PDF filename (default: `report_dataset.pdf`).
- `--plots_dirname` (optional): Subdirectory inside dataset root to store auxiliary plots.

Example
```python
  python -m dataset_formater.scripts.get_dataset_report 
    --dataset_path ./dataset_formater/dataset_yolo11 
    --dataset_format yolo 
    --report_filename report_dataset_yolo11.pdf 
    --plots_dirname report_plots
```
---

### `get_subset_dataset.py`

Description  
Create a random subset of a dataset (YOLO / COCO / COCO JSON), keeping a given fraction of images per split while preserving format and structure.

Arguments

- `--dataset_path` (required): Path to the dataset root folder.
- `--dataset_format` (required): Dataset format: `yolo`, `coco` or `coco_json`.
- `--keep_percentage` (required): Fraction of images to keep in each split, in `[0, 1]`.
- `--output_path` (optional): Output dataset root; if empty, uses `<dataset_root>_pruned`.

Example
```python
  python -m dataset_formater.scripts.get_subset_dataset 
    --dataset_path ./dataset_formater/dataset_yolo11 
    --dataset_format yolo 
    --keep_percentage 0.2 
    --output_path ./dataset_formater/dataset_yolo11_subset_20
```
---

### `remap_labels_dataset.py`

Description  
Remap, merge or drop classes in a dataset (YOLO / COCO / COCO JSON) according to an `ID_MAP` defined inside the script, and export the result to a chosen format.

Arguments

- `--source_path` (required): Path to the source dataset root folder.
- `--source_format` (required): Source format: `yolo`, `coco` or `coco_json`.
- `--dest_path` (required): Path where the remapped dataset will be saved.
- `--dest_format` (optional): Output format: `yolo`, `coco`, `coco_json` or empty string (`""` = same as source).

Note: The class mapping `ID_MAP` (old_id -> new_id or `None` to drop) is edited directly in the script.

Example
```python
  python -m dataset_formater.scripts.remap_labels_dataset 
    --source_path ./dataset_formater/dataset_yolo11 
    --source_format yolo 
    --dest_path ./dataset_formater/dataset_yolo11_remapped 
    --dest_format yolo
```
---

### `transform2segmentation.py`

Description  
Convert a detection dataset with bounding boxes (YOLO / COCO / COCO JSON) into an instance segmentation dataset by generating masks per box using Segment Anything (SAM).

Arguments

- `--dataset_path` (required): Path to the dataset root folder.
- `--dataset_format` (required): Input format: `yolo`, `coco` or `coco_json`.
- `--dest_path` (required): Path where the segmentation dataset will be saved (for example COCO JSON with `segmentation` filled).
- `--dest_format` (required): Output format: `yolo`, `coco` or `coco_json`.
- `--sam_checkpoint` (required): Path to the SAM `.pth` checkpoint (for example `dataset_formater/sam_checkpoints/sam_vit_l_0b3195.pth`).
- `--sam_model_type` (optional): SAM backbone type: `vit_h`, `vit_l`, or `vit_b` (default: `vit_l`).
- `--sam_device` (optional): Device for inference: `cuda` or `cpu` (default: `cuda`).
- `--score_threshold` (optional): Minimum SAM mask score to accept a prediction.
- `--box_expansion_ratio` (optional): Relative bbox padding before feeding to SAM (for example `0.05`).
- `--overwrite_existing` (optional flag): If set, overwrite annotations that already have segmentation.
- `--no_recompute_bbox` (optional flag): If set, keep original bbox instead of recomputing it from the mask.

Example
```python
  python -m dataset_formater.scripts.transform2segmentation 
    --dataset_path ./dataset_formater/dataset_yolo11 
    --dataset_format yolo 
    --dest_path ./dataset_formater/dataset_segmented_coco 
    --dest_format coco_json 
    --sam_checkpoint dataset_formater/sam_checkpoints/sam_vit_l_0b3195.pth 
    --sam_model_type vit_l 
  
```
