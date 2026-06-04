from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def build_empty_mask(width: int, height: int, background_id: int = 0) -> np.ndarray:
    return np.full((height, width), background_id, dtype=np.uint8)


def apply_binary_mask_to_label_map(
    label_map: np.ndarray,
    binary_mask: np.ndarray,
    semantic_id: int,
    background_id: int = 0,
) -> None:
    """Write semantic_id into label_map only where binary_mask is True and pixel is background.

    Modifies label_map in-place. Never overwrites already-assigned pixels.
    """
    write_region = binary_mask.astype(bool) & (label_map == background_id)
    label_map[write_region] = semantic_id


def colorize_label_map(
    label_map: np.ndarray,
    semantic_id_to_color: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    h, w = label_map.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for semantic_id, (r, g, b) in semantic_id_to_color.items():
        region = label_map == semantic_id
        color_mask[region, 0] = r
        color_mask[region, 1] = g
        color_mask[region, 2] = b
    return color_mask


def save_index_mask(path: str | Path, label_map: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label_map.astype(np.uint8), mode="L").save(str(path))


def save_color_mask(path: str | Path, color_mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color_mask.astype(np.uint8), mode="RGB").save(str(path))


def blend_image_with_mask(
    image_rgb: np.ndarray,
    color_mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Alpha-blend color_mask over image_rgb. alpha controls mask opacity (0=invisible, 1=opaque)."""
    blended = image_rgb * (1.0 - alpha) + color_mask * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_preview(path: str | Path, preview: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview.astype(np.uint8), mode="RGB").save(str(path))


def discover_images(source_path: Path) -> list[Path]:
    images: list[Path] = []
    for p in source_path.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(p)
    return sorted(images)


def validate_palette_coverage(
    prompt_to_semantic_id: dict[str, int],
    semantic_id_to_color: dict[int, tuple[int, int, int]],
    background_id: int = 0,
) -> None:
    if background_id not in semantic_id_to_color:
        raise ValueError(
            f"Background ID {background_id} must have an entry in semantic_id_to_color."
        )
    used_ids = set(prompt_to_semantic_id.values()) | {background_id}
    missing = used_ids - set(semantic_id_to_color.keys())
    if missing:
        raise ValueError(
            f"Semantic IDs {sorted(missing)} appear in prompt_to_semantic_id "
            f"but have no color entry in semantic_id_to_color."
        )
