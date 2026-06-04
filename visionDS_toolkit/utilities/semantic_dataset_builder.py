from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from visionDS_toolkit.utilities.sam3_inference import load_sam3_inferencer
from visionDS_toolkit.utilities.semantic_mask_utils import (
    apply_binary_mask_to_label_map,
    blend_image_with_mask,
    build_empty_mask,
    colorize_label_map,
    discover_images,
    save_color_mask,
    save_index_mask,
    save_preview,
)

BACKGROUND_ID = 0


def create_semantic_dataset(
    source_path: str,
    dest_path: str,
    prompt_to_semantic_id: dict[str, int],
    semantic_id_to_color: dict[int, tuple[int, int, int]],
    sam3_checkpoint: str,
    sam3_model_type: str,
    device: str = "cuda",
    mask_threshold: float = 0.0,
    confidence_threshold: float = 0.5,
    prompt_to_confidence: dict[str, float] | None = None,
    preview_alpha: float = 0.5,
) -> None:
    src = Path(source_path).resolve()
    dst = Path(dest_path).resolve()

    images_dir = dst / "images"
    labels_dir = dst / "labels"
    labels_color_dir = dst / "labels_color"
    preview_dir = dst / "preview"
    for d in (images_dir, labels_dir, labels_color_dir, preview_dir):
        d.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(src)
    total = len(image_paths)
    print(f"[INFO] Found {total} image(s) in: {src}")

    print("[INFO] Loading SAM3 model...")
    inferencer = load_sam3_inferencer(sam3_checkpoint, sam3_model_type, device)
    print("[INFO] Model loaded.")

    ok = 0
    failed_read = 0
    failed_inference = 0

    for idx, img_path in enumerate(image_paths, 1):
        stem = img_path.stem

        shutil.copy2(img_path, images_dir / img_path.name)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Cannot read image, skipping: {img_path.name}")
            failed_read += 1
            continue

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        label_map = build_empty_mask(w, h, background_id=BACKGROUND_ID)

        inference_failed = False
        for prompt, semantic_id in prompt_to_semantic_id.items():
            conf = (prompt_to_confidence or {}).get(prompt, confidence_threshold)
            try:
                masks = inferencer.predict_prompt_masks(img_rgb, prompt, mask_threshold, conf)
            except Exception as exc:
                print(
                    f"[WARN] Inference failed for '{img_path.name}' "
                    f"prompt='{prompt}': {exc}"
                )
                inference_failed = True
                break

            for mask in masks:
                if mask.shape != (h, w):
                    continue
                apply_binary_mask_to_label_map(
                    label_map, mask, semantic_id, BACKGROUND_ID
                )

        if inference_failed:
            failed_inference += 1
            continue

        color_mask = colorize_label_map(label_map, semantic_id_to_color)
        preview = blend_image_with_mask(img_rgb, color_mask, alpha=preview_alpha)

        save_index_mask(labels_dir / f"{stem}.png", label_map)
        save_color_mask(labels_color_dir / f"{stem}.png", color_mask)
        save_preview(preview_dir / f"{stem}.png", preview)

        ok += 1
        print(f"[{idx}/{total}] {img_path.name}")

    print("========== [Semantic dataset summary] ==========")
    print(f"Total images found:     {total}")
    print(f"Successfully processed: {ok}")
    print(f"Failed to read:         {failed_read}")
    print(f"Failed inference:       {failed_inference}")
    print(f"Masks generated:        {ok}")
    print(f"Destination:            {dst}")
    print("=================================================")
