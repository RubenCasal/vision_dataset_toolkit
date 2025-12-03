from dataset_interface import DatasetIR, Image, Annotation, Category, BBox
from pathlib import Path
from typing import List
from PIL import Image as PILImage
import json


def load_yolo_dataset(root: str) -> DatasetIR:
    root_path = Path(root)
    images_dir = root_path / "images"
    labels_dir = root_path / "labels"

    images: List[Image] = []
    annotations: List[Annotation] = []
    categories_map: dict[int, Category] = {}

    img_id = 1
    ann_id = 1

    for img_path in sorted(images_dir.glob("*.*")):
        with PILImage.open(img_path) as im:
            w, h = im.size

        image = Image(
            id=img_id,
            file_name=img_path.name,
            width=w,
            height=h,
        )
        images.append(image)

        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                x = (cx - bw / 2.0) * w
                y = (cy - bh / 2.0) * h
                abs_w = bw * w
                abs_h = bh * h

                if class_id not in categories_map:
                    categories_map[class_id] = Category(
                        id=class_id,
                        name=f"class_{class_id}",
                    )

                annotations.append(
                    Annotation(
                        id=ann_id,
                        image_id=img_id,
                        category_id=class_id,
                        bbox=BBox(x=x, y=y, width=abs_w, height=abs_h),
                        area=abs_w * abs_h,
                    )
                )
                ann_id += 1

        img_id += 1

    return DatasetIR(
        images=images,
        annotations=annotations,
        categories=list(categories_map.values()),
    )


def load_coco_dataset(split_root: str) -> DatasetIR:

    split_dir = Path(split_root)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO annotations JSON not found at: {ann_path}")

    print(f"[INFO] Using COCO annotations: {ann_path}")

    data = json.loads(ann_path.read_text(encoding="utf-8"))

    images: List[Image] = [
        Image(
            id=int(im["id"]),
            file_name=im["file_name"],
            width=int(im["width"]),
            height=int(im["height"]),
        )
        for im in data["images"]
    ]

    categories: List[Category] = [
        Category(
            id=int(c["id"]),
            name=c["name"],
            supercategory=c.get("supercategory"),
        )
        for c in data["categories"]
    ]

    annotations: List[Annotation] = []
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        area = a.get("area", w * h)

        seg = a.get("segmentation")
        flat_seg = None
        if isinstance(seg, list) and seg:
            if isinstance(seg[0], list):
                flat_seg = seg[0]
            else:
                flat_seg = seg

        annotations.append(
            Annotation(
                id=int(a["id"]),
                image_id=int(a["image_id"]),
                category_id=int(a["category_id"]),
                bbox=BBox(x=float(x), y=float(y), width=float(w), height=float(h)),
                area=float(area),
                iscrowd=int(a.get("iscrowd", 0)),
                segmentation=flat_seg,
                keypoints=a.get("keypoints"),
                num_keypoints=a.get("num_keypoints"),
            )
        )

    return DatasetIR(
        images=images,
        annotations=annotations,
        categories=categories,
    )
