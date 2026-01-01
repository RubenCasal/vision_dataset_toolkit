from pathlib import Path
from typing import Dict, Optional
import shutil
from dataset_formater.utilities.dataset_folder_interface import DatasetFolder
from dataset_formater.utilities.dataset_interface import (
    DatasetIR,
    Image,
    Annotation,
    BBox,
    Category,
)
import json


def select_dump_function(fmt: str):
    if fmt == "yolo":
        return dump_yolo_dataset
    elif fmt == "coco":
        return dump_coco_estandar_dataset
    elif fmt == "coco_json":
        return dump_coco_json_dataset
    else:
        raise ValueError(f"Unsupported dest_format={fmt!r}")


######### YOLO DUMPING FUNCTIONS #########
def dump_yolo_split(
    dataset: DatasetIR,
    out_split_root: str,
    images_source_dir: Path,
):
    out_root_path = Path(out_split_root)
    images_dir = out_root_path / "images"
    labels_dir = out_root_path / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cat_id_to_yolo: Dict[int, int] = {
        cat.id: idx
        for idx, cat in enumerate(sorted(dataset.categories, key=lambda c: c.id))
    }
    anns_by_image: Dict[int, list[Annotation]] = {}
    for ann in dataset.annotations:
        anns_by_image.setdefault(ann.image_id, []).append(ann)

    for image in dataset.images:
        w_img, h_img = image.width, image.height
        anns = anns_by_image.get(image.id, [])

        lines: list[str] = []

        for ann in anns:
            b = ann.bbox

            cx = (b.x + b.width / 2.0) / w_img
            cy = (b.y + b.height / 2.0) / h_img
            nw = b.width / w_img
            nh = b.height / h_img

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            cls_id = cat_id_to_yolo[ann.category_id]
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = labels_dir / (Path(image.file_name).stem + ".txt")
        label_path.write_text("\n".join(lines), encoding="utf-8")

        src_img = images_source_dir / image.file_name
        if src_img.exists():
            shutil.copy2(src_img, images_dir / image.file_name)


def dump_yolo_dataset(
    dataset: DatasetFolder,
    output_path: str,
):
    out_base = Path(output_path)
    if dataset.train is not None and "train" in dataset.split_dirs:
        src_images_dir = dataset.split_dirs["train"] / "images"
        dump_yolo_split(
            dataset.train,
            out_split_root=out_base / "train",
            images_source_dir=src_images_dir,
        )

    if dataset.valid is not None:
        src_key = "valid" if "valid" in dataset.split_dirs else "val"
        if src_key in dataset.split_dirs:
            src_images_dir = dataset.split_dirs[src_key] / "images"
            dump_yolo_split(
                dataset.valid,
                out_split_root=out_base / "val",
                images_source_dir=src_images_dir,
            )

    if dataset.test is not None and "test" in dataset.split_dirs:
        src_images_dir = dataset.split_dirs["test"] / "images"
        dump_yolo_split(
            dataset.test,
            out_split_root=out_base / "test",
            images_source_dir=src_images_dir,
        )


######### COCO ESTÁNDAR DUMPING FUNCTIONS #########


def _dataset_ir_to_coco_dict(dataset: DatasetIR) -> dict:
    """
    Helper común para construir el JSON COCO a partir de DatasetIR.
    Lo reaprovechamos tanto en Roboflow-style como en estándar.
    """
    images_json = [
        {
            "id": int(im.id),
            "file_name": im.file_name,
            "width": int(im.width),
            "height": int(im.height),
        }
        for im in dataset.images
    ]

    categories_json = [
        {
            "id": int(cat.id),
            "name": cat.name,
            "supercategory": cat.supercategory or "",
        }
        for cat in dataset.categories
    ]

    anns_by_image: Dict[int, list[Annotation]] = {}
    for ann in dataset.annotations:
        anns_by_image.setdefault(ann.image_id, []).append(ann)

    annotations_json = []
    for anns in anns_by_image.values():
        for ann in anns:
            b = ann.bbox
            bbox = [float(b.x), float(b.y), float(b.width), float(b.height)]
            area = (
                float(ann.area) if ann.area is not None else float(b.width * b.height)
            )

            seg = ann.segmentation
            if seg is not None:
                if seg and isinstance(seg[0], (int, float)):
                    segmentation = [seg]  # [[x1, y1, ...]]
                else:
                    segmentation = seg
            else:
                segmentation = None

            annotations_json.append(
                {
                    "id": int(ann.id),
                    "image_id": int(ann.image_id),
                    "category_id": int(ann.category_id),
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": int(ann.iscrowd),
                    "segmentation": segmentation,
                    "keypoints": ann.keypoints,
                    "num_keypoints": ann.num_keypoints,
                }
            )

    return {
        "images": images_json,
        "annotations": annotations_json,
        "categories": categories_json,
    }


def dump_coco_estandar_split(
    dataset: DatasetIR,
    out_images_dir: str | Path,
    images_source_dir: Path,
    ann_path: str | Path,
) -> None:
    """
    COCO estándar para un split:

      out_root/
        train/images/...
        annotations/instances_train.json

    Aquí solo gestionamos UN split:
      - Copiamos imágenes a out_images_dir
      - Escribimos ann_path (instances_*.json)
    """
    out_images_dir = Path(out_images_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)

    ann_path = Path(ann_path)
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Copiar imágenes
    for im in dataset.images:
        src_img = images_source_dir / im.file_name
        dst_img = out_images_dir / im.file_name

        if src_img.exists():
            if src_img.resolve() != dst_img.resolve():
                shutil.copy2(src_img, dst_img)

    # 2) Construir JSON COCO
    coco_dict = _dataset_ir_to_coco_dict(dataset)

    ann_path.write_text(json.dumps(coco_dict, indent=2), encoding="utf-8")


def dump_coco_estandar_dataset(
    dataset: DatasetFolder,
    output_path: str | Path,
    coco_prefix: str = "instances",
) -> None:
    """
    Crea un dataset COCO estándar con layout:

      output_path/
        train/images/...
        val/images/...
        test/images/...
        annotations/
          instances_train.json
          instances_val.json
          instances_test.json

    Asume que las imágenes fuente están en:
      split_dir / "images"
    (coincide con tu loader coco_estandar y con YOLO → IR → COCO estándar)
    """
    out_base = Path(output_path)
    out_base.mkdir(parents=True, exist_ok=True)

    ann_dir = out_base / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # TRAIN
    if dataset.train is not None and "train" in dataset.split_dirs:
        src_images_dir = dataset.split_dirs["train"] / "images"
        out_images_dir = out_base / "train" / "images"
        ann_path = ann_dir / f"{coco_prefix}_train.json"

        dump_coco_estandar_split(
            dataset.train,
            out_images_dir=out_images_dir,
            images_source_dir=src_images_dir,
            ann_path=ann_path,
        )

    # VALID / VAL → salida siempre "val"
    if dataset.valid is not None:
        src_key = "valid" if "valid" in dataset.split_dirs else "val"
        if src_key in dataset.split_dirs:
            src_images_dir = dataset.split_dirs[src_key] / "images"
            out_images_dir = out_base / "val" / "images"
            ann_path = ann_dir / f"{coco_prefix}_val.json"

            dump_coco_estandar_split(
                dataset.valid,
                out_images_dir=out_images_dir,
                images_source_dir=src_images_dir,
                ann_path=ann_path,
            )

    # TEST
    if dataset.test is not None and "test" in dataset.split_dirs:
        src_images_dir = dataset.split_dirs["test"] / "images"
        out_images_dir = out_base / "test" / "images"
        ann_path = ann_dir / f"{coco_prefix}_test.json"

        dump_coco_estandar_split(
            dataset.test,
            out_images_dir=out_images_dir,
            images_source_dir=src_images_dir,
            ann_path=ann_path,
        )


######### COCO JSON (tipo Roboflow) DUMPING FUNCTIONS #########


def dump_coco_json_split(
    dataset: DatasetIR,
    out_split_root: str | Path,
    images_source_dir: Path,
    coco_filename: str = "_annotations.coco.json",
) -> None:

    out_split_root = Path(out_split_root)
    out_split_root.mkdir(parents=True, exist_ok=True)

    images_source_dir = Path(images_source_dir)
    if (
        not any(images_source_dir.glob("*.*"))
        and (images_source_dir / "images").exists()
    ):
        images_source_dir = images_source_dir / "images"

    for im in dataset.images:
        src_img = images_source_dir / im.file_name
        dst_img = out_split_root / im.file_name

        if src_img.exists():
            if src_img.resolve() != dst_img.resolve():
                shutil.copy2(src_img, dst_img)

    coco_dict = _dataset_ir_to_coco_dict(dataset)

    ann_path = out_split_root / coco_filename
    ann_path.write_text(json.dumps(coco_dict, indent=2), encoding="utf-8")


def dump_coco_json_dataset(
    dataset: DatasetFolder,
    output_path: str | Path,
    coco_filename: str = "_annotations.coco.json",
) -> None:

    out_base = Path(output_path)
    out_base.mkdir(parents=True, exist_ok=True)

    # TRAIN
    if dataset.train is not None and "train" in dataset.split_dirs:
        split_dir = dataset.split_dirs["train"]
        # puede ser split_dir o split_dir/images
        images_source_dir = split_dir
        if (split_dir / "images").exists():
            images_source_dir = split_dir / "images"

        dump_coco_json_split(
            dataset.train,
            out_split_root=out_base / "train",
            images_source_dir=images_source_dir,
            coco_filename=coco_filename,
        )

    # VALID / VAL → salida siempre "val"
    if dataset.valid is not None:
        src_key = "valid" if "valid" in dataset.split_dirs else "val"
        if src_key in dataset.split_dirs:
            split_dir = dataset.split_dirs[src_key]
            images_source_dir = split_dir
            if (split_dir / "images").exists():
                images_source_dir = split_dir / "images"

            dump_coco_json_split(
                dataset.valid,
                out_split_root=out_base / "val",
                images_source_dir=images_source_dir,
                coco_filename=coco_filename,
            )

    # TEST
    if dataset.test is not None and "test" in dataset.split_dirs:
        split_dir = dataset.split_dirs["test"]
        images_source_dir = split_dir
        if (split_dir / "images").exists():
            images_source_dir = split_dir / "images"

        dump_coco_json_split(
            dataset.test,
            out_split_root=out_base / "test",
            images_source_dir=images_source_dir,
            coco_filename=coco_filename,
        )
