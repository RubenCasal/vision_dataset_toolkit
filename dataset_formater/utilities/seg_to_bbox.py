# dataset_formater/utilities/seg_to_bbox.py

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from dataset_formater.utilities.dataset_interface import (
    DatasetIR,
    Image,
    Annotation,
    BBox,
)
from dataset_formater.utilities.dataset_folder_interface import DatasetFolder


def _poly_to_bbox(
    poly_flat: List[float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float] | None:
    """
    poly_flat: [x0, y0, x1, y1, ...] in image coords.
    Returns (x, y, w, h) or None if invalid.
    """
    if len(poly_flat) < 6:
        return None

    xs = np.array(poly_flat[0::2], dtype=float)
    ys = np.array(poly_flat[1::2], dtype=float)

    if xs.size == 0 or ys.size == 0:
        return None

    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())

    # Clamp to image
    x_min = max(0.0, min(x_min, img_w - 1.0))
    x_max = max(0.0, min(x_max, img_w - 1.0))
    y_min = max(0.0, min(y_min, img_h - 1.0))
    y_max = max(0.0, min(y_max, img_h - 1.0))

    w = x_max - x_min
    h = y_max - y_min
    if w <= 0.0 or h <= 0.0:
        return None

    return x_min, y_min, w, h


def _normalize_segmentation(seg) -> List[float] | None:
    """
    Normalize `segmentation` to a single flat polygon:
      - [x0, y0, x1, y1, ...] -> itself
      - [[...], [...]] -> concatenated
    """
    if seg is None:
        return None
    if isinstance(seg, list) and len(seg) == 0:
        return None

    # Case 1: flat list
    if isinstance(seg, list) and all(isinstance(v, (int, float)) for v in seg):
        return [float(v) for v in seg]

    # Case 2: list of lists
    if isinstance(seg, list) and isinstance(seg[0], list):
        flat: List[float] = []
        for poly in seg:
            if not poly:
                continue
            for v in poly:
                if isinstance(v, (int, float)):
                    flat.append(float(v))
        if len(flat) < 6:
            return None
        return flat

    return None


def _segmented_split_to_detection_inplace(
    ds: DatasetIR,
    split_name: str = "",
) -> None:
    """
    In-place conversion for one split:
      - For each annotation with `segmentation`, compute bbox from mask polygon.
      - Overwrite `ann.bbox` and `ann.area`.
      - Remove `ann.segmentation` (set to None).
      - Annotations without valid segmentation are left unchanged.
    """
    if ds is None:
        return

    img_by_id: Dict[int, Image] = {im.id: im for im in ds.images}

    total = len(ds.annotations)
    updated = 0
    skipped_no_seg = 0
    skipped_invalid = 0

    for ann in ds.annotations:
        im = img_by_id.get(ann.image_id)
        if im is None or im.width <= 0 or im.height <= 0:
            continue

        seg = _normalize_segmentation(ann.segmentation)
        if seg is None:
            skipped_no_seg += 1
            continue

        bbox_vals = _poly_to_bbox(seg, img_w=im.width, img_h=im.height)
        if bbox_vals is None:
            skipped_invalid += 1
            continue

        x, y, w, h = bbox_vals
        ann.bbox = BBox(x=float(x), y=float(y), width=float(w), height=float(h))
        ann.area = float(w * h)
        ann.segmentation = None  # remove mask → pure detection

        updated += 1

    name = f" ({split_name})" if split_name else ""
    print(
        f"[SEG->DET]{name} annotations: total={total}, "
        f"updated_bbox={updated}, "
        f"skipped_no_seg={skipped_no_seg}, "
        f"skipped_invalid_poly={skipped_invalid}"
    )


def segmented_folder_to_detection(folder: DatasetFolder) -> DatasetFolder:
    """
    Convert a segmented DatasetFolder into a detection dataset in-place:
      - train / valid / test are updated.
      - Returns the same DatasetFolder instance.
    """

    if folder.train is not None:
        _segmented_split_to_detection_inplace(folder.train, split_name="train")

    if folder.valid is not None:
        _segmented_split_to_detection_inplace(folder.valid, split_name="valid")

    if folder.test is not None:
        _segmented_split_to_detection_inplace(folder.test, split_name="test")

    return folder
