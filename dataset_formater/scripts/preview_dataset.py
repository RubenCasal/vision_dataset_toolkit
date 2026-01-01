import argparse
from pathlib import Path
from typing import Optional

from dataset_formater.utilities.dataset_folder_interface import DatasetFolder
from dataset_formater.utilities.draw_annotations import visualize_dataset_ir

# =========================
# DEFAULT CONSTANTS
# =========================
DATASET_PATH: str = ""  # e.g. "data/source_yolo"
DATASET_FORMAT: str = ""  # "yolo" | "coco" | "coco_json"
OUTPUT_ROOT: str = ""  # e.g. "data/preview_out" (if empty -> auto preview/<split>)
BLUR_RADIUS: float = 3.0
MAX_IMAGES: Optional[int] = None  # e.g. 100 for debug


def run_visualization(
    dataset_path: str,
    dataset_format: str,
    output_root: Optional[str] = None,
    blur_radius: float = 3.0,
    max_images: Optional[int] = None,
) -> None:
    print(f"Init visualization for dataset {dataset_format} at {dataset_path}")

    ds_folder = DatasetFolder(path=dataset_path, dataset_type=dataset_format)

    # train
    if ds_folder.train is not None and "train" in ds_folder.split_dirs:
        split_dir = ds_folder.split_dirs["train"]
        images_root = (
            split_dir / "images" if (split_dir / "images").exists() else split_dir
        )
        split_out = None
        if output_root:
            split_out = Path(output_root) / "train"
        visualize_dataset_ir(
            dataset=ds_folder.train,
            images_root=images_root,
            output_root=split_out,
            blur_radius=blur_radius,
            max_images=max_images,
        )

    # valid / val
    if ds_folder.valid is not None:
        src_key = "valid" if "valid" in ds_folder.split_dirs else "val"
        if src_key in ds_folder.split_dirs:
            split_dir = ds_folder.split_dirs[src_key]
            images_root = (
                split_dir / "images" if (split_dir / "images").exists() else split_dir
            )
            split_out = None
            if output_root:
                split_out = Path(output_root) / "val"
            visualize_dataset_ir(
                dataset=ds_folder.valid,
                images_root=images_root,
                output_root=split_out,
                blur_radius=blur_radius,
                max_images=max_images,
            )

    # test
    if ds_folder.test is not None and "test" in ds_folder.split_dirs:
        split_dir = ds_folder.split_dirs["test"]
        images_root = (
            split_dir / "images" if (split_dir / "images").exists() else split_dir
        )
        split_out = None
        if output_root:
            split_out = Path(output_root) / "test"
        visualize_dataset_ir(
            dataset=ds_folder.test,
            images_root=images_root,
            output_root=split_out,
            blur_radius=blur_radius,
            max_images=max_images,
        )

    print("Visualization completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize dataset annotations (YOLO, COCO, COCO JSON) with blurred overlays."
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
        help="Format of the dataset (yolo, coco, coco_json).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=OUTPUT_ROOT,
        help="Base output folder for previews. "
        "If empty, previews are stored in <dataset_root>/preview/<split>.",
    )
    parser.add_argument(
        "--blur_radius",
        type=float,
        default=BLUR_RADIUS,
        help="Gaussian blur radius applied to annotation overlays.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=MAX_IMAGES if MAX_IMAGES is not None else -1,
        help="Max images per split to visualize (-1 means all).",
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

    max_imgs = (
        None if args.max_images is None or args.max_images < 0 else args.max_images
    )
    out_root = args.output_root if args.output_root else None

    run_visualization(
        dataset_path=args.dataset_path,
        dataset_format=args.dataset_format,
        output_root=out_root,
        blur_radius=args.blur_radius,
        max_images=max_imgs,
    )

if __name__ == "__main__":
    main()
