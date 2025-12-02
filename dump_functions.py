from pathlib import Path
from typing import Dict, Optional
import shutil
from dataset_folder_interface import DatasetFolder
from dataset_interface import DatasetIR, Image, Annotation, BBox, Category


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
