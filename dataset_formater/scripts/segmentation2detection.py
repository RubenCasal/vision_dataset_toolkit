# dataset_formater/scripts/segmentation2detection.py

from __future__ import annotations

import argparse
from pathlib import Path

from dataset_formater.utilities.dataset_folder_interface import DatasetFolder
from dataset_formater.utilities.seg_to_bbox import segmented_folder_to_detection
from dataset_formater.utilities.dump_functions import select_dump_function


def run_segmentation_to_detection(
    dataset_path: str,
    dataset_format: str,
    dest_path: str,
    dest_format: str,
) -> None:
    dataset_root = Path(dataset_path).resolve()
    dest_root = Path(dest_path).resolve()

    print(
        f"[SEG->DET] Init conversion: "
        f"dataset={dataset_root}, format={dataset_format} -> dest_format={dest_format}"
    )

    # Load segmented dataset (with masks)
    folder = DatasetFolder(path=str(dataset_root), dataset_type=dataset_format)

    # Convert masks → bboxes in-place
    segmented_folder_to_detection(folder)

    # Dump in desired format
    dumper = select_dump_function(dest_format)
    dumper(folder, str(dest_root))

    print(f"[SEG->DET] Converted detection dataset saved at: {dest_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a segmentation dataset (with masks) into a pure detection "
            "dataset (only bounding boxes)."
        )
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the segmented dataset root folder.",
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        required=True,
        choices=["yolo", "coco", "coco_json"],
        help="Input dataset format.",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        required=True,
        help="Path where the detection dataset will be saved.",
    )
    parser.add_argument(
        "--dest_format",
        type=str,
        required=True,
        choices=["yolo", "coco", "coco_json"],
        help="Output dataset format.",
    )

    args = parser.parse_args()

    run_segmentation_to_detection(
        dataset_path=args.dataset_path,
        dataset_format=args.dataset_format,
        dest_path=args.dest_path,
        dest_format=args.dest_format,
    )


if __name__ == "__main__":
    main()
