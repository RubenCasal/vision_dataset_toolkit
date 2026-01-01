# dataset_formater/scripts/remap_labels_dataset.py

import argparse
from pathlib import Path
from typing import Optional, Dict

from dataset_formater.utilities.dataset_folder_interface import DatasetFolder
from dataset_formater.utilities.dump_functions import (
    select_dump_function,
    dump_yolo_dataset,
    dump_coco_estandar_dataset,
    dump_coco_json_dataset,
)
from dataset_formater.utilities.change_labels import remap_dataset_labels


# =========================
# DEFAULT CONSTANTS
# =========================
SOURCE_PATH: str = ""  # e.g. "dataset_formater/dataset_yolo11"
SOURCE_FORMAT: str = ""  # "yolo" | "coco" | "coco_json"
DEST_PATH: str = ""  # e.g. "dataset_formater/dataset_yolo11_remapped"
DEST_FORMAT: str = ""  # if "", defaults to SOURCE_FORMAT

# Class remapping plan: old_id -> new_id (or None to drop)
ID_MAP: Dict[int, Optional[int]] = {
    0: 1,  # car     -> vehicle
    1: None,  # truck   -> vehicle
    # 2: 0,    # person  -> person (now id 0)
    # 3: 2,    # boat    -> boat
    # 4: None, # tree    -> removed
}


def run_remap_labels(
    source_path: str,
    source_format: str,
    dest_path: str,
    dest_format: Optional[str] = None,
) -> None:
    """
    Load a dataset (DatasetFolder), remap labels for each split using ID_MAP,
    and dump the result in dest_path with the selected format.
    """
    if not ID_MAP:
        raise ValueError(
            "ID_MAP is empty. Define the class remapping in the script before running."
        )

    if dest_format is None or dest_format == "":
        dest_format = source_format

    print(
        f"Init label remapping for dataset {source_format} at {source_path} "
        f"-> dest {dest_format} at {dest_path}"
    )

    ds_folder = DatasetFolder(path=source_path, dataset_type=source_format)

    # Remap each split in memory
    if ds_folder.train is not None:
        print("[INFO] Remapping labels for train split...")
        ds_folder.train = remap_dataset_labels(
            ds_folder.train,
            id_map=ID_MAP,
            drop_empty_images=True,
            verbose=True,
        )

    if ds_folder.valid is not None:
        print("[INFO] Remapping labels for val/valid split...")
        ds_folder.valid = remap_dataset_labels(
            ds_folder.valid,
            id_map=ID_MAP,
            drop_empty_images=True,
            verbose=True,
        )

    if ds_folder.test is not None:
        print("[INFO] Remapping labels for test split...")
        ds_folder.test = remap_dataset_labels(
            ds_folder.test,
            id_map=ID_MAP,
            drop_empty_images=True,
            verbose=True,
        )

    # Dump with the chosen format
    dumper = select_dump_function(dest_format)
    dumper(ds_folder, dest_path)

    print(f"[DONE] Remapped dataset saved in: {Path(dest_path).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Remap dataset labels (merge/remove classes) and export "
            "to YOLO / COCO / COCO JSON."
        )
    )

    parser.add_argument(
        "--source_path",
        type=str,
        default=SOURCE_PATH,
        help="Path to the source dataset root folder.",
    )
    parser.add_argument(
        "--source_format",
        type=str,
        default=SOURCE_FORMAT,
        choices=["yolo", "coco", "coco_json"],
        help="Format of the source dataset.",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        default=DEST_PATH,
        help="Path to save the remapped dataset.",
    )
    parser.add_argument(
        "--dest_format",
        type=str,
        default=DEST_FORMAT,
        choices=["yolo", "coco", "coco_json", ""],
        help="Format of the destination dataset ('' = same as source).",
    )

    args = parser.parse_args()

    # Use constants if CLI args are empty
    source_path = args.source_path
    source_format = args.source_format
    dest_path = args.dest_path
    dest_format = args.dest_format or args.source_format

    if not source_path:
        raise ValueError(
            "source_path is empty. Set SOURCE_PATH in the script or pass --source_path."
        )
    if not dest_path:
        raise ValueError(
            "dest_path is empty. Set DEST_PATH in the script or pass --dest_path."
        )
    if not source_format:
        raise ValueError(
            "source_format is empty. Set SOURCE_FORMAT in the script or pass "
            "--source_format (yolo|coco|coco_json)."
        )

    run_remap_labels(
        source_path=source_path,
        source_format=source_format,
        dest_path=dest_path,
        dest_format=dest_format,
    )
