from pathlib import Path
from typing import Dict, Tuple, Optional
from dataset_formater.utilities.dataset_interface import DatasetIR, Annotation
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import random
import colorsys


def get_color_for_category(
    cat_id: int, cache: Dict[int, Tuple[int, int, int]]
) -> Tuple[int, int, int]:
    if cat_id in cache:
        return cache[cat_id]

    random.seed(cat_id)
    h = random.random()
    s = 0.6 + 0.4 * random.random()
    v = 0.7 + 0.3 * random.random()

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    color = (int(r * 255), int(g * 255), int(b * 255))
    cache[cat_id] = color
    return color


def draw_annotation(
    draw: ImageDraw.ImageDraw,
    ann: Annotation,
    color_rgb: Tuple[int, int, int],
    fill_alpha: int = 80,
    outline_alpha: int = 220,
) -> None:
    r, g, b = color_rgb

    if ann.segmentation:
        seg = ann.segmentation
        if len(seg) >= 6:
            points = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
            draw.polygon(
                points,
                fill=(r, g, b, fill_alpha),
                outline=(r, g, b, outline_alpha),
            )
            return

    b_box = ann.bbox
    x1 = b_box.x
    y1 = b_box.y
    x2 = b_box.x + b_box.width
    y2 = b_box.y + b_box.height

    draw.rectangle(
        [x1, y1, x2, y2],
        fill=(r, g, b, fill_alpha),
        outline=(r, g, b, outline_alpha),
        width=2,
    )


def visualize_dataset_ir(
    dataset: DatasetIR,
    images_root: str | Path,
    output_root: Optional[str | Path] = None,
    blur_radius: float = 3.0,
    max_images: Optional[int] = None,
) -> None:
    images_root = Path(images_root)

    if output_root is None or str(output_root) == "":
        if images_root.name == "images":
            split_dir = images_root.parent
        else:
            split_dir = images_root

        split_name = split_dir.name
        dataset_root = split_dir.parent
        output_root = dataset_root / "preview" / split_name
    else:
        output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    anns_by_image: Dict[int, list[Annotation]] = {}
    for ann in dataset.annotations:
        anns_by_image.setdefault(ann.image_id, []).append(ann)

    color_cache: Dict[int, Tuple[int, int, int]] = {}
    cat_id_to_name: Dict[int, str] = {c.id: c.name for c in dataset.categories}

    font = ImageFont.load_default()

    processed = 0
    for img in dataset.images:
        if max_images is not None and processed >= max_images:
            break

        img_path = images_root / img.file_name
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            continue

        with Image.open(img_path) as base_img:
            base = base_img.convert("RGB")
            w, h = base.size

            overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay, mode="RGBA")

            anns = anns_by_image.get(img.id, [])
            for ann in anns:
                color = get_color_for_category(ann.category_id, color_cache)
                draw_annotation(draw, ann, color_rgb=color)

            blurred_overlay = overlay.filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            )
            composed = Image.alpha_composite(base.convert("RGBA"), blurred_overlay)

            draw_sharp = ImageDraw.Draw(composed, mode="RGBA")
            for ann in anns:
                color = get_color_for_category(ann.category_id, color_cache)
                r, g, b = color
                b_box = ann.bbox
                x1 = b_box.x
                y1 = b_box.y
                x2 = b_box.x + b_box.width
                y2 = b_box.y + b_box.height

                # Sharp bbox
                draw_sharp.rectangle(
                    [x1, y1, x2, y2],
                    outline=(r, g, b, 255),
                    width=2,
                )

                # Label text
                label = cat_id_to_name.get(ann.category_id, str(ann.category_id))

                # Compute text size (Pillow >= 8: textbbox; fallback to textsize if exists)
                if hasattr(draw_sharp, "textbbox"):
                    bbox = draw_sharp.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                else:
                    # Older Pillow
                    text_w, text_h = font.getsize(label)

                text_x = x1
                text_y = max(0, y1 - text_h - 2)

                # Background box for text
                draw_sharp.rectangle(
                    [
                        text_x,
                        text_y,
                        text_x + text_w + 4,
                        text_y + text_h + 2,
                    ],
                    fill=(0, 0, 0, 160),
                )

                # Text (white)
                draw_sharp.text(
                    (text_x + 2, text_y + 1),
                    label,
                    font=font,
                    fill=(255, 255, 255, 255),
                )

            out_path = output_root / img.file_name
            composed.convert("RGB").save(out_path)

        processed += 1
        print(f"[INFO] Saved visualization: {out_path}")
