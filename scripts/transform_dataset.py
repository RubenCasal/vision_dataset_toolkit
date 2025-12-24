import argparse
from utilities.dataset_folder_interface import DatasetFolder
from utilities.dump_functions import (
    dump_yolo_dataset,
    dump_coco_estandar_dataset,
    dump_coco_json_dataset,
)


def select_dump_function(format: str):
    if format == "yolo":
        return dump_yolo_dataset
    elif format == "coco":
        return dump_coco_estandar_dataset
    elif format == "coco_json":
        return dump_coco_json_dataset
    else:
        raise ValueError(f"Unsupported dest_format={format!r}")


# FORMATS SUPPORTED: coco, coco_json, yolo
def transform_dataset(
    source_path: str,
    source_format: str,
    dest_path: str,
    dest_format: str,
):
    print(f"Init transformation from {source_format} to {dest_format}")

    dataset = DatasetFolder(path=source_path, dataset_type=source_format)
    dumper = select_dump_function(format=dest_format)
    dumper(dataset, dest_path)
    print(f"Transformation completed. Dataset saved in {dest_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Transform dataset between different formats (YOLO, COCO, COCO JSON)."
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path to the source dataset folder.",
    )
    parser.add_argument(
        "--source_format",
        type=str,
        required=True,
        choices=["yolo", "coco", "coco_json"],
        help="Format of the source dataset.",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        required=True,
        help="Path to save the transformed dataset.",
    )
    parser.add_argument(
        "--dest_format",
        type=str,
        required=True,
        choices=["yolo", "coco", "coco_json"],
        help="Format of the destination dataset.",
    )

    args = parser.parse_args()

    transform_dataset(
        source_path=args.source_path,
        source_format=args.source_format,
        dest_path=args.dest_path,
        dest_format=args.dest_format,
    )
