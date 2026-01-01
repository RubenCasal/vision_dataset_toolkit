from __future__ import annotations

import argparse
from pathlib import Path

from dataset_formater.utilities.dataset_folder_interface import DatasetFolder
from dataset_formater.utilities.dump_functions import select_dump_function
from dataset_formater.utilities.sam_segmentation import add_sam_masks_to_dataset

# =========================
# DEFAULT CONSTANTS
# =========================
DATASET_PATH: str = ""  # e.g. "dataset_formater/dataset_yolo11"
DATASET_FORMAT: str = ""  # "yolo" | "coco" | "coco_json"
DEST_PATH: str = ""  # if empty -> "<dataset_root>_sam"
DEST_FORMAT: str = ""  # if empty -> same as DATASET_FORMAT

# SAM (segment_anything) config
SAM_CHECKPOINT: str = "dataset_formater/sam_checkpoints/sam_vit_l_0b3195.pth"
SAM_MODEL_TYPE: str = "vit_l"  # "vit_h" | "vit_l" | "vit_b"
SAM_DEVICE: str = "cuda"  # "cuda" or "cpu"

SCORE_THRESHOLD: float = 0.0
BOX_EXPANSION_RATIO: float = 0.0
OVERWRITE_EXISTING: bool = False
RECOMPUTE_BBOX: bool = True


def run_sam_conversion(
    dataset_path: str,
    dataset_format: str,
    dest_path: str,
    dest_format: str,
    sam_checkpoint: str,
    sam_model_type: str = "vit_h",
    sam_device: str = "cuda",
    score_threshold: float = 0.0,
    box_expansion_ratio: float = 0.0,
    overwrite_existing: bool = False,
    recompute_bbox: bool = True,
) -> None:
    print(f"[SAM] Init dataset SAM conversion for {dataset_format} at {dataset_path}")
    root = Path(dataset_path)

    ds_folder = DatasetFolder(path=str(root), dataset_type=dataset_format)

    # ------------------- TRAIN
    if ds_folder.train is not None and "train" in ds_folder.split_dirs:
        split_dir = ds_folder.split_dirs["train"]
        images_root = (
            split_dir / "images" if (split_dir / "images").exists() else split_dir
        )
        print(f"[SAM] Processing train split at: {images_root}")
        add_sam_masks_to_dataset(
            dataset=ds_folder.train,
            images_root=images_root,
            sam_checkpoint=sam_checkpoint,
            model_type=sam_model_type,
            device=sam_device,
            score_threshold=score_threshold,
            box_expansion_ratio=box_expansion_ratio,
            overwrite_existing=overwrite_existing,
            recompute_bbox_from_mask=recompute_bbox,
            verbose=True,
        )

    # ------------------- VALID / VAL
    if ds_folder.valid is not None:
        src_key = "valid" if "valid" in ds_folder.split_dirs else "val"
        if src_key in ds_folder.split_dirs:
            split_dir = ds_folder.split_dirs[src_key]
            images_root = (
                split_dir / "images" if (split_dir / "images").exists() else split_dir
            )
            print(f"[SAM] Processing {src_key} split at: {images_root}")
            add_sam_masks_to_dataset(
                dataset=ds_folder.valid,
                images_root=images_root,
                sam_checkpoint=sam_checkpoint,
                model_type=sam_model_type,
                device=sam_device,
                score_threshold=score_threshold,
                box_expansion_ratio=box_expansion_ratio,
                overwrite_existing=overwrite_existing,
                recompute_bbox_from_mask=recompute_bbox,
                verbose=True,
            )

    # ------------------- TEST
    if ds_folder.test is not None and "test" in ds_folder.split_dirs:
        split_dir = ds_folder.split_dirs["test"]
        images_root = (
            split_dir / "images" if (split_dir / "images").exists() else split_dir
        )
        print(f"[SAM] Processing test split at: {images_root}")
        add_sam_masks_to_dataset(
            dataset=ds_folder.test,
            images_root=images_root,
            sam_checkpoint=sam_checkpoint,
            model_type=sam_model_type,
            device=sam_device,
            score_threshold=score_threshold,
            box_expansion_ratio=box_expansion_ratio,
            overwrite_existing=overwrite_existing,
            recompute_bbox_from_mask=recompute_bbox,
            verbose=True,
        )

    # Dump converted dataset
    dest_root = Path(dest_path)
    dumper = select_dump_function(dest_format)

    print(f"[SAM] Saving converted dataset to: {dest_root} (format={dest_format})")
    dumper(ds_folder, str(dest_root))
    print("[SAM] Conversion + save completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert a detection dataset (YOLO / COCO / COCO JSON) into a "
            "segmentation dataset using SAM (segment_anything) masks per bounding box."
        )
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DATASET_PATH,
        help="Path to the dataset root folder.",
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        default=DATASET_FORMAT,
        choices=["yolo", "coco", "coco_json"],
        help="Format of the input dataset.",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        default=DEST_PATH,
        help='Path to save the converted dataset. If empty -> "<dataset_root>_sam".',
    )
    parser.add_argument(
        "--dest_format",
        type=str,
        default=DEST_FORMAT,
        choices=["yolo", "coco", "coco_json"],
        help="Format of the output dataset. If empty -> same as dataset_format.",
    )

    # SAM config
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=SAM_CHECKPOINT,
        help="Path to SAM .pth checkpoint (e.g. sam_vit_h_4b8939.pth).",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default=SAM_MODEL_TYPE,
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type.",
    )
    parser.add_argument(
        "--sam_device",
        type=str,
        default=SAM_DEVICE,
        choices=["cuda", "cpu"],
        help="Device to run SAM.",
    )

    parser.add_argument(
        "--score_threshold",
        type=float,
        default=SCORE_THRESHOLD,
        help="Min mask score to accept SAM mask (0.0–1.0).",
    )
    parser.add_argument(
        "--box_expansion_ratio",
        type=float,
        default=BOX_EXPANSION_RATIO,
        help="Relative expansion of each bbox before passing to SAM (e.g. 0.05).",
    )
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        default=OVERWRITE_EXISTING,
        help="If set, overwrite annotations that already have segmentation.",
    )
    parser.add_argument(
        "--no_recompute_bbox",
        action="store_true",
        help="If set, DO NOT recompute bbox from SAM mask (keep original box).",
    )

    args = parser.parse_args()

    if not args.dataset_path:
        raise ValueError(
            "dataset_path is empty. Set DATASET_PATH in the script or pass --dataset_path."
        )
    if not args.dataset_format:
        raise ValueError(
            "dataset_format is empty. Set DATASET_FORMAT in the script or pass "
            "--dataset_format (yolo|coco|coco_json)."
        )
    if not args.sam_checkpoint:
        raise ValueError(
            "sam_checkpoint is empty. Set SAM_CHECKPOINT in the script or pass --sam_checkpoint."
        )

    ds_path = args.dataset_path
    ds_format = args.dataset_format

    if args.dest_path:
        dest_path = args.dest_path
    else:
        root = Path(ds_path)
        dest_path = str(root.parent / f"{root.name}_sam")

    dest_format = args.dest_format if args.dest_format else ds_format

    run_sam_conversion(
        dataset_path=ds_path,
        dataset_format=ds_format,
        dest_path=dest_path,
        dest_format=dest_format,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        sam_device=args.sam_device,
        score_threshold=args.score_threshold,
        box_expansion_ratio=args.box_expansion_ratio,
        overwrite_existing=args.overwrite_existing,
        recompute_bbox=not args.no_recompute_bbox,
    )
