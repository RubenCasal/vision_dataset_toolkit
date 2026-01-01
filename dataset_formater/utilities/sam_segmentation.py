from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

from dataset_formater.utilities.dataset_interface import (
    DatasetIR,
    Image,
    Annotation,
    BBox,
)


def add_sam_masks_to_dataset(
    dataset: DatasetIR,
    images_root: str | Path,
    sam_checkpoint: str | Path,
    model_type: str = "vit_h",  # "vit_h" | "vit_l" | "vit_b"
    device: str = "cuda",
    score_threshold: float = 0.0,
    box_expansion_ratio: float = 0.0,
    overwrite_existing: bool = False,
    recompute_bbox_from_mask: bool = True,
    verbose: bool = True,
) -> DatasetIR:
    """
    Usa Meta SAM (segment_anything) para añadir máscaras de instancia a un DatasetIR
    de detección (solo bounding boxes).

    Para cada anotación:
      - Usa su bbox como box-prompt de SAM.
      - Selecciona la mejor máscara de SAM (multimask_output).
      - Guarda un polígono COCO (flatten) en `ann.segmentation`.
      - Opcionalmente recalcula `ann.bbox` a partir de la máscara.
      - Actualiza `ann.area` con el número de píxeles del objeto.

    Modifica `dataset` IN-PLACE y también lo devuelve.
    """
    images_root = Path(images_root)
    sam_checkpoint = Path(sam_checkpoint)

    if verbose:
        print("[SAM] Loading SAM (segment_anything)...")
        print(f"      checkpoint: {sam_checkpoint}")
        print(f"      model_type: {model_type}")
        print(f"      device:     {device}")

    if not sam_checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at: {sam_checkpoint}\n"
            "Descárgalo desde el repo oficial de SAM y coloca el .pth en esa ruta."
        )

    sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Índice image_id -> lista de anotaciones
    anns_by_image: Dict[int, List[Annotation]] = {}
    for ann in dataset.annotations:
        anns_by_image.setdefault(ann.image_id, []).append(ann)

    total_anns = len(dataset.annotations)
    processed_anns = 0
    with_masks = 0
    skipped_existing = 0
    low_score_skipped = 0
    failed_masks = 0
    missing_images = 0

    for im in dataset.images:
        img_path = images_root / im.file_name
        if not img_path.exists():
            missing_images += 1
            if verbose:
                print(f"[WARN][SAM] Image not found, skipping: {img_path}")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            missing_images += 1
            if verbose:
                print(f"[WARN][SAM] Cannot read image, skipping: {img_path}")
            continue

        img_h, img_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            predictor.set_image(img_rgb)
        except Exception as e:
            missing_images += 1
            if verbose:
                print(f"[WARN][SAM] set_image failed for {img_path}: {e}")
            continue

        anns = anns_by_image.get(im.id, [])
        if not anns:
            continue

        for ann in anns:
            processed_anns += 1

            # Si ya tiene máscara y no queremos sobreescribir, saltamos
            if (not overwrite_existing) and (ann.segmentation is not None):
                skipped_existing += 1
                continue

            b = ann.bbox
            x0 = float(b.x)
            y0 = float(b.y)
            x1 = float(b.x + b.width)
            y1 = float(b.y + b.height)

            # Expansión opcional
            if box_expansion_ratio > 0.0:
                pad_x = box_expansion_ratio * (x1 - x0)
                pad_y = box_expansion_ratio * (y1 - y0)
                x0 -= pad_x
                y0 -= pad_y
                x1 += pad_x
                y1 += pad_y

            # Clip al tamaño de la imagen
            x0 = max(0.0, min(x0, img_w - 1.0))
            y0 = max(0.0, min(y0, img_h - 1.0))
            x1 = max(0.0, min(x1, img_w - 1.0))
            y1 = max(0.0, min(y1, img_h - 1.0))

            if x1 <= x0 or y1 <= y0:
                failed_masks += 1
                continue

            box = np.array([x0, y0, x1, y1], dtype=np.float32)

            try:
                masks, scores, _ = predictor.predict(
                    box=box[None, :],
                    multimask_output=True,
                )
            except Exception as e:
                failed_masks += 1
                if verbose:
                    print(f"[WARN][SAM] Error predicting mask for {img_path}: {e}")
                continue

            if masks is None or len(masks) == 0:
                failed_masks += 1
                continue

            scores = np.array(scores, dtype=np.float32)
            best_idx = int(scores.argmax())
            best_score = float(scores[best_idx])

            if best_score < score_threshold:
                low_score_skipped += 1
                continue

            mask = masks[best_idx].astype(np.uint8)  # [H, W]
            if mask.ndim != 2 or mask.sum() == 0:
                failed_masks += 1
                continue

            ys, xs = np.where(mask > 0)
            if ys.size == 0 or xs.size == 0:
                failed_masks += 1
                continue

            area = float(mask.sum())

            y_min = int(ys.min())
            y_max = int(ys.max())
            x_min = int(xs.min())
            x_max = int(xs.max())

            # Contornos -> polígono COCO
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                failed_masks += 1
                continue

            contour = max(contours, key=cv2.contourArea)
            contour = contour.squeeze(1)

            if contour.ndim != 2 or contour.shape[0] < 3:
                failed_masks += 1
                continue

            poly_flat: List[float] = []
            for px, py in contour:
                poly_flat.extend([float(px), float(py)])

            if len(poly_flat) < 6:
                failed_masks += 1
                continue

            # Escribimos máscara y área en la anotación
            ann.segmentation = poly_flat
            ann.area = area

            # Recalcular bbox desde la máscara si se pide
            if recompute_bbox_from_mask:
                new_w = float(x_max - x_min + 1)
                new_h = float(y_max - y_min + 1)
                ann.bbox = BBox(
                    x=float(x_min),
                    y=float(y_min),
                    width=new_w,
                    height=new_h,
                )

            with_masks += 1

            if verbose and processed_anns % 100 == 0:
                print(
                    f"[SAM] Processed {processed_anns}/{total_anns} anns, "
                    f"with_masks={with_masks}, skipped_existing={skipped_existing}, "
                    f"low_score={low_score_skipped}, failed={failed_masks}, "
                    f"missing_imgs={missing_images}",
                )

    if verbose:
        print("========== [SAM conversion summary] ==========")
        print(f"Total annotations:          {total_anns}")
        print(f"Processed annotations:      {processed_anns}")
        print(f"Annotations with new masks: {with_masks}")
        print(f"Skipped (existing segm.):   {skipped_existing}")
        print(f"Skipped (low score):        {low_score_skipped}")
        print(f"Failed / invalid masks:     {failed_masks}")
        print(f"Images missing/unreadable:  {missing_images}")
        print("=============================================")

    return dataset
