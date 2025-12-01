from load_datasets import load_yolo_dataset
from pathlib import Path
from dataset_interface import DatasetIR
from dataclasses import dataclass


def get_loader(dataset_type: str):
    if dataset_type == "yolo":
        return load_yolo_dataset
    elif dataset_type == "coco":
        pass
    else:
        raise ValueError(f"Unsupported dataset_type={dataset_type!r}")


class DatasetFolder:

    def __init__(
        self, path: str, dataset_type: str, folders=["train", "test", "valid"]
    ):

        base = Path(path)
        loader = get_loader(dataset_type=dataset_type)

        self.train = None
        self.test = None
        self.valid = None

        for folder in folders:
            split_dir = base / folder

            if not split_dir.exists():
                print(f"[WARN] Split '{folder}' no existe en {split_dir}")
                continue
            try:
                ds = loader(str(split_dir))
            except FileNotFoundError:
                print(f"[WARN] Error cargando split '{folder}' en {split_dir}: {e}")
                continue

            if folder == "train":
                self.train = ds
                print(f"Loaded train dataset from {split_dir}")
            elif folder in ("val", "valid"):
                self.valid = ds
                print(f"Loaded valid dataset from {split_dir}")
            elif folder == "test":
                self.test = ds
                print(f"Loaded test dataset from {split_dir}")
