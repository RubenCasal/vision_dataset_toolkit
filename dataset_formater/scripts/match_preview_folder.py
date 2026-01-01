import argparse
from pathlib import Path
from typing import Optional

from dataset_formater.utilities.dataset_folder_interface import DatasetFolder
from dataset_formater.utilities.match_preview import (
    interactive_prune_datasetfolder_by_preview,
)
from dataset_formater.utilities.dump_functions import (
    select_dump_function,
    dump_yolo_dataset,
    dump_coco_estandar_dataset,
    dump_coco_json_dataset,
)

# =========================
# DEFAULT CONSTANTS
# =========================
DATASET_PATH: str = ""  # e.g. "data/source_yolo"
DATASET_FORMAT: str = ""  # "yolo" | "coco" | "coco_json"
PREVIEW_ROOT: str = ""  # if empty -> <dataset_root>/preview
OUTPUT_PATH: str = ""  # if empty -> "<dataset_root>_pruned"


def run_prune(
    dataset_path: str,
    dataset_format: str,
    preview_root: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    dataset_path = str(Path(dataset_path).resolve())
    print(f"Init pruning for dataset {dataset_format} at {dataset_path}")

    ds_folder = DatasetFolder(path=dataset_path, dataset_type=dataset_format)

    preview_root_path: Optional[Path] = None
    if preview_root:
        preview_root_path = Path(preview_root).resolve()

    pruned_folder = interactive_prune_datasetfolder_by_preview(
        folder=ds_folder,
        dataset_root=dataset_path,
        preview_root=preview_root_path,
    )

    # If user aborted or nothing to prune, interactive_* returns the original object
    if pruned_folder is ds_folder:
        print("[INFO] No pruning applied. Exiting without writing output.")
        return

    if not output_path:
        output_path = f"{dataset_path}_pruned"

    out_path = Path(output_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    dumper = select_dump_function(dataset_format)
    dumper(pruned_folder, str(out_path))

    print(f"[INFO] Pruned dataset written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Prune a dataset (YOLO, COCO, COCO JSON) using the images kept "
            "in the preview folder. A new pruned dataset is written to disk."
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
        help="Format of the dataset (yolo, coco, coco_json).",
    )
    parser.add_argument(
        "--preview_root",
        type=str,
        default=PREVIEW_ROOT,
        help=(
            "Base preview folder. If empty, uses <dataset_root>/preview "
            "with subfolders train/val/test."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=OUTPUT_PATH,
        help=(
            "Output path for the pruned dataset. If empty, uses "
            "<dataset_root>_pruned."
        ),
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

    preview_root_arg = args.preview_root if args.preview_root else None
    output_path_arg = args.output_path if args.output_path else None

    run_prune(
        dataset_path=args.dataset_path,
        dataset_format=args.dataset_format,
        preview_root=preview_root_arg,
        output_path=output_path_arg,
    )
