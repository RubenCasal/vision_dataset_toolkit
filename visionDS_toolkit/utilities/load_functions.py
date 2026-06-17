from visionDS_toolkit.utilities.dataset_interface import (
    DatasetIR,
    Image,
    Annotation,
    Category,
    BBox,
)
from pathlib import Path
from typing import List
from PIL import Image as PILImage
import json


def load_yolo_dataset(root: str, max_images: int | None = None) -> DatasetIR:
    root_path = Path(root)
    images_dir = root_path / "images"
    labels_dir = root_path / "labels"

    images: List[Image] = []
    annotations: List[Annotation] = []
    categories_map: dict[int, Category] = {}

    img_id = 1
    ann_id = 1
    processed = 0

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
            txt = label_path.read_text(encoding="utf-8").strip()
            if txt:
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                      
                        continue

                    class_id = int(parts[0])

                    if class_id not in categories_map:
                        categories_map[class_id] = Category(
                            id=class_id,
                            name=f"class_{class_id}",
                        )

                    nums = list(map(float, parts[1:]))

                    # ---------------------------
                    # YOLO DET: cls cx cy w h
                    # ---------------------------
                    if len(nums) == 4:
                        cx, cy, bw, bh = nums

                        x = (cx - bw / 2.0) * w
                        y = (cy - bh / 2.0) * h
                        abs_w = bw * w
                        abs_h = bh * h

                       
                        x = max(0.0, min(x, w - 1.0))
                        y = max(0.0, min(y, h - 1.0))
                        abs_w = max(0.0, min(abs_w, w - x))
                        abs_h = max(0.0, min(abs_h, h - y))

                        annotations.append(
                            Annotation(
                                id=ann_id,
                                image_id=img_id,
                                category_id=class_id,
                                bbox=BBox(x=x, y=y, width=abs_w, height=abs_h),
                                area=abs_w * abs_h,
                                segmentation=None,
                            )
                        )
                        ann_id += 1
                        continue

                    # ---------------------------------------------
                    # YOLO SEG: cls x1 y1 x2 y2 ... 
                    # ---------------------------------------------
                    if len(nums) >= 6 and (len(nums) % 2 == 0):
                        poly_px: list[float] = []
                        for i in range(0, len(nums), 2):
                            xn = nums[i]
                            yn = nums[i + 1]

                            # clamp [0,1]
                            xn = max(0.0, min(1.0, xn))
                            yn = max(0.0, min(1.0, yn))

                            poly_px.extend([xn * w, yn * h])

                        xs = poly_px[0::2]
                        ys = poly_px[1::2]
                        if not xs or not ys:
                            continue

                        x0 = max(0.0, min(xs))
                        y0 = max(0.0, min(ys))
                        x1 = min(w - 1.0, max(xs))
                        y1 = min(h - 1.0, max(ys))

                        bw = max(0.0, x1 - x0)
                        bh = max(0.0, y1 - y0)
                        area = bw * bh 

                        annotations.append(
                            Annotation(
                                id=ann_id,
                                image_id=img_id,
                                category_id=class_id,
                                bbox=BBox(x=x0, y=y0, width=bw, height=bh),
                                area=area,
                                segmentation=poly_px,
                            )
                        )
                        ann_id += 1
                        continue

                    # Si llega aquí: formato desconocido -> skip
                    continue

        img_id += 1
        processed += 1
        if max_images is not None and processed >= max_images:
            break

    return DatasetIR(
        images=images,
        annotations=annotations,
        categories=list(categories_map.values()),
    )

def load_coco_estandar_dataset(split_root: str, max_images: int | None = None) -> DatasetIR:

    split_dir = Path(split_root)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    split_name = split_dir.name 
    base = split_dir.parent  
    ann_dir = base / "annotations"

    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations folder not found: {ann_dir}")

    candidate_names = [
        f"instances_{split_name}.json",
        f"instances_{split_name}2017.json",
        f"{split_name}.json",
    ]

    ann_path = None
    for name in candidate_names:
        p = ann_dir / name
        if p.exists() and p.is_file():
            ann_path = p
            break

    if ann_path is None:
        raise FileNotFoundError(
            f"No COCO JSON found for split '{split_name}' in {ann_dir}"
        )

    print(f"[INFO] Using COCO estándar annotations for '{split_name}': {ann_path}")

    data = json.loads(ann_path.read_text(encoding="utf-8"))

    images_data = data["images"]
    if max_images is not None:
        images_data = images_data[:max_images]

    images: List[Image] = [
        Image(
            id=int(im["id"]),
            file_name=im["file_name"],
            width=int(im["width"]),
            height=int(im["height"]),
        )
        for im in images_data
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
    # if images were limited, filter annotations to those images
    kept_image_ids = {im.id for im in images}
    for a in data["annotations"]:
        if kept_image_ids and int(a["image_id"]) not in kept_image_ids:
            continue
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


def load_coco_json_dataset(split_root: str, max_images: int | None = None) -> DatasetIR:

    split_dir = Path(split_root)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    candidate_names = [
        "_annotations.coco.json",
        "__annotations.coco.json",
        "annotations.coco.json",
    ]

    ann_path = None
    for name in candidate_names:
        p = split_dir / name
        if p.exists() and p.is_file():
            ann_path = p
            break

    if ann_path is None:
        raise FileNotFoundError(
            f"RF-DETR COCO JSON not found in split dir: {split_dir}"
        )

    print(f"[INFO] Using RF-DETR COCO annotations at: {ann_path}")

    data = json.loads(ann_path.read_text(encoding="utf-8"))

    images_data = data["images"]
    if max_images is not None:
        images_data = images_data[:max_images]

    images: List[Image] = [
        Image(
            id=int(im["id"]),
            file_name=im["file_name"],
            width=int(im["width"]),
            height=int(im["height"]),
        )
        for im in images_data
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
    kept_image_ids = {im.id for im in images}
    for a in data["annotations"]:
        if kept_image_ids and int(a["image_id"]) not in kept_image_ids:
            continue
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
