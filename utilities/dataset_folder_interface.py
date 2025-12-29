from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

from dataset_formater.utilities.load_functions import (
    load_yolo_dataset,
    load_coco_estandar_dataset,
    load_coco_json_dataset,
)
from dataset_formater.utilities.dataset_interface import DatasetIR


def get_loader(dataset_type: str):
    if dataset_type == "yolo":
        return load_yolo_dataset
    elif dataset_type == "coco":
        return load_coco_estandar_dataset
    elif dataset_type == "coco_json":
        return load_coco_json_dataset
    else:
        raise ValueError(f"Unsupported dataset_type={dataset_type!r}")


class DatasetFolder:
    """
    Simple container for train/valid/test splits loaded into DatasetIR.

    - Looks for:
        root/train
        root/test
        root/valid  OR  root/val  (in that priority)
    - If neither 'valid' nor 'val' exists, no valid split is loaded and no warning
      is printed for that case.
    """

    def __init__(self, path: str, dataset_type: str) -> None:
        base = Path(path)
        loader = get_loader(dataset_type=dataset_type)

        self.train: Optional[DatasetIR] = None
        self.valid: Optional[DatasetIR] = None
        self.test: Optional[DatasetIR] = None

        # Keeps the real directory path for each split key:
        #   "train", "test", "valid" or "val"
        self.split_dirs: Dict[str, Path] = {}

        # -----------------------------
        # TRAIN
        # -----------------------------
        train_dir = base / "train"
        if train_dir.exists():
            try:
                ds = loader(str(train_dir))
                self.train = ds
                self.split_dirs["train"] = train_dir
                print(f"Loaded train dataset from {train_dir}")
            except FileNotFoundError as e:
                print(f"[WARN] Error loading split 'train' at {train_dir}: {e}")
        else:
            print(f"[WARN] Split 'train' does not exist in {train_dir}")

        # -----------------------------
        # TEST
        # -----------------------------
        test_dir = base / "test"
        if test_dir.exists():
            try:
                ds = loader(str(test_dir))
                self.test = ds
                self.split_dirs["test"] = test_dir
                print(f"Loaded test dataset from {test_dir}")
            except FileNotFoundError as e:
                print(f"[WARN] Error loading split 'test' at {test_dir}: {e}")
        else:
            print(f"[WARN] Split 'test' does not exist in {test_dir}")

        # -----------------------------
        # VALIDATION: valid or val
        # -----------------------------
        valid_dir: Optional[Path] = None
        valid_key: Optional[str] = None

        cand_valid = base / "valid"
        cand_val = base / "val"

        if cand_valid.exists():
            valid_dir = cand_valid
            valid_key = "valid"
        elif cand_val.exists():
            valid_dir = cand_val
            valid_key = "val"

        if valid_dir is not None and valid_key is not None:
            try:
                ds = loader(str(valid_dir))
                self.valid = ds
                self.split_dirs[valid_key] = valid_dir
                print(f"Loaded valid dataset from {valid_dir}")
            except FileNotFoundError as e:
                print(f"[WARN] Error loading split '{valid_key}' at {valid_dir}: {e}")
        # Si no hay ni 'valid' ni 'val', no se imprime warning: simplemente no hay split de validación.
