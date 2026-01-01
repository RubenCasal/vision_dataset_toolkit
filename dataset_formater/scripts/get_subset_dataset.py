import argparse
from pathlib import Path
from typing import Optional

from dataset_formater.utilities.dataset_folder_interface import DatasetFolder
from dataset_formater.utilities.subset_dataset import subset_dataset
from dataset_formater.utilities.dump_functions import (
    select_dump_function,
)

# =========================
# DEFAULT CONSTANTS
# =========================
DATASET_PATH: str = ""  # e.g. "data/source_yolo"
DATASET_FORMAT: str = ""  # "yolo" | "coco" | "coco_json"
KEEP_PERCENTAGE: float = 1.0  # value in [0, 1]
OUTPUT_PATH: str = ""  # if empty -> "<dataset_root>_pruned"


def run_subset(
    dataset_path: str,
    dataset_format: str,
    keep_percentage: float,
    output_path: Optional[str] = None,
) -> None:
    print(
        f"Init subset: dataset={dataset_path}, "
        f"format={dataset_format}, keep={keep_percentage:.3f}"
    )

    # Load original dataset
    ds_folder = DatasetFolder(path=dataset_path, dataset_type=dataset_format)

    # Create in-memory subset (same split_dirs, fewer images/annotations)
    subset_folder = subset_dataset(ds_folder, keep_percentage=keep_percentage)

    # Decide destination path
    src_root = Path(dataset_path)
    if output_path:
        dest_root = Path(output_path)
    else:
        # default: "<dataset_root>_pruned"
        dest_root = src_root.with_name(src_root.name + "_pruned")

    dest_root = dest_root.resolve()
    dest_root.parent.mkdir(parents=True, exist_ok=True)

    # Dump subset in the same format as the source
    dumper = select_dump_function(dataset_format)
    dumper(subset_folder, str(dest_root))

    print(f"Subset dataset saved at: {dest_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create a random subset of a dataset (YOLO, COCO, COCO JSON), "
            "keeping a given percentage of images per split."
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
        "--keep_percentage",
        type=float,
        default=KEEP_PERCENTAGE,
        help="Fraction of images to keep in each split, in [0, 1].",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=OUTPUT_PATH,
        help='Output dataset root. If empty, uses "<dataset_root>_pruned".',
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

    # Basic validation/clamp for keep_percentage
    keep = max(0.0, min(1.0, args.keep_percentage))

    run_subset(
        dataset_path=args.dataset_path,
        dataset_format=args.dataset_format,
        keep_percentage=keep,
        output_path=args.output_path if args.output_path else None,
    )
