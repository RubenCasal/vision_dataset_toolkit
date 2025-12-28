from dataset_formater.utilities.load_functions import (
    load_yolo_dataset,
    load_coco_estandar_dataset,
    load_coco_json_dataset,
)
from pathlib import Path
from typing import Optional
from dataset_formater.utilities.dataset_interface import DatasetIR
from dataclasses import dataclass


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

    def __init__(
        self, path: str, dataset_type: str, folders=("train", "test", "valid")
    ):

        base = Path(path)
        loader = get_loader(dataset_type=dataset_type)

        self.train: Optional[DatasetIR] = None
        self.test: Optional[DatasetIR] = None
        self.valid: Optional[DatasetIR] = None

        self.split_dirs: dict[str, Path] = {}

        for folder in folders:
            split_dir = base / folder

            if not split_dir.exists():
                print(f"[WARN] Split '{folder}' no existe en {split_dir}")
                continue
            try:
                ds = loader(str(split_dir))
            except FileNotFoundError as e:
                print(f"[WARN] Error cargando split '{folder}' en {split_dir}: {e}")
                continue

            self.split_dirs[folder] = split_dir

            if folder == "train":
                self.train = ds
                print(f"Loaded train dataset from {split_dir}")
            elif folder in ("val", "valid"):
                self.valid = ds
                print(f"Loaded valid dataset from {split_dir}")
            elif folder == "test":
                self.test = ds
                print(f"Loaded test dataset from {split_dir}")
