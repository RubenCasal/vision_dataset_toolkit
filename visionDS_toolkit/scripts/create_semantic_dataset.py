from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

from visionDS_toolkit.utilities.semantic_mask_utils import (
    discover_images,
    validate_palette_coverage,
)
from visionDS_toolkit.utilities.semantic_dataset_builder import create_semantic_dataset

# =========================
# DEFAULT CONSTANTS
# =========================
SOURCE_PATH: str = ""
DEST_PATH: str = ""
SAM3_CHECKPOINT: str = ""
SAM3_MODEL_TYPE: str = "sam3"
DEVICE: str = "cuda"
MASK_THRESHOLD: float = 0.0
CONFIDENCE_THRESHOLD: float = 0.5

BACKGROUND_ID: int = 0

PROMPT_TO_SEMANTIC_ID: Dict[str, int] = {
    "building":   1,
    "vegetation": 2,
    "tree":       2,
    "road":       3,
    "pavement":   3,
}

# Per-prompt confidence overrides. Any prompt not listed here falls back to
# CONFIDENCE_THRESHOLD. Set to {} to use the global threshold for all prompts.
PROMPT_TO_CONFIDENCE: Dict[str, float] = {
    "building":   0.5,
    "vegetation": 0.5,
    "tree":       0.5,
    "road":       0.5,
    "pavement":   0.5,
}

SEMANTIC_ID_TO_COLOR: Dict[int, Tuple[int, int, int]] = {
    0: (243,  69, 141),
    1: ( 35, 198, 233),
    2: ( 93, 220,  11),
    3: (227, 192,  29),
}


def _validate(args: argparse.Namespace) -> None:
    src = Path(args.source_path)
    dst = Path(args.dest_path)

    if not src.exists():
        raise ValueError(f"source_path does not exist: {src}")
    if not src.is_dir():
        raise ValueError(f"source_path is not a directory: {src}")
    if src.resolve() == dst.resolve():
        raise ValueError("source_path and dest_path must be different directories.")

    images = discover_images(src)
    if not images:
        raise ValueError(f"No image files found in: {src}")
    print(f"[INFO] Discovered {len(images)} image(s).")

    ckpt = Path(args.sam3_checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {ckpt}\n"
            "Download it with:  download_sam3 --model-type large"
        )

    validate_palette_coverage(PROMPT_TO_SEMANTIC_ID, SEMANTIC_ID_TO_COLOR, BACKGROUND_ID)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a folder of images into a semantic segmentation dataset "
            "using SAM3 prompt-driven segmentation.\n\n"
            "Output structure:\n"
            "  <dest_path>/images/         (copies of source images)\n"
            "  <dest_path>/labels/         (single-channel uint8 ID masks)\n"
            "  <dest_path>/labels_color/   (RGB visualization masks)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--source_path",
        type=str,
        default=SOURCE_PATH,
        help="Path to the folder of source images.",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        default=DEST_PATH,
        help="Path to save the output dataset.",
    )
    parser.add_argument(
        "--sam3_checkpoint",
        type=str,
        default=SAM3_CHECKPOINT,
        help="Path to the SAM3 .pt checkpoint file.",
    )
    parser.add_argument(
        "--sam3_model_type",
        type=str,
        default=SAM3_MODEL_TYPE,
        choices=["sam3", "sam3.1"],
        help="SAM3 release to use: 'sam3' or 'sam3.1' (default: sam3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        choices=["cuda", "cpu"],
        help="Device to run SAM3 on (default: cuda).",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=MASK_THRESHOLD,
        help="Minimum SAM3 stability score to accept a mask, in [0.0, 1.0] (default: 0.0).",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=(
            "Global SAM3 detection confidence threshold, in [0.0, 1.0] (default: 0.5). "
            "Used for prompts not listed in PROMPT_TO_CONFIDENCE."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing destination directory.",
    )

    args = parser.parse_args()

    if not args.source_path:
        parser.error("--source_path is required (or set SOURCE_PATH in the script).")
    if not args.dest_path:
        parser.error("--dest_path is required (or set DEST_PATH in the script).")
    if not args.sam3_checkpoint:
        parser.error(
            "--sam3_checkpoint is required (or set SAM3_CHECKPOINT in the script)."
        )

    dst = Path(args.dest_path)
    if dst.exists() and not args.overwrite:
        print(
            f"[ERROR] dest_path already exists: {dst}\n"
            "        Use --overwrite to allow writing into an existing directory."
        )
        sys.exit(1)

    try:
        _validate(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    create_semantic_dataset(
        source_path=args.source_path,
        dest_path=args.dest_path,
        prompt_to_semantic_id=PROMPT_TO_SEMANTIC_ID,
        semantic_id_to_color=SEMANTIC_ID_TO_COLOR,
        sam3_checkpoint=args.sam3_checkpoint,
        sam3_model_type=args.sam3_model_type,
        device=args.device,
        mask_threshold=args.mask_threshold,
        confidence_threshold=args.confidence_threshold,
        prompt_to_confidence=PROMPT_TO_CONFIDENCE or None,
    )


if __name__ == "__main__":
    main()
