from __future__ import annotations

from typing import Any

import torch
import numpy as np
from PIL import Image as PILImage

# Supported model versions
_SUPPORTED_MODELS = ("sam3", "sam3.1")


class Sam3Inferencer:
    """Wraps the SAM3 image processor behind a simple predict-per-prompt interface.

    Each call to predict_prompt_masks runs a fresh text-prompted inference via
    Sam3Processor.set_text_prompt(), which natively supports semantic text prompts
    (e.g. "building", "vegetation").  No per-image caching is needed because SAM3
    keeps the image embedding inside the inference_state object.
    """

    def __init__(self, checkpoint: str, device: str) -> None:
        self._device = device
        self._processor = _build_processor(checkpoint, device)

    def predict_prompt_masks(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        mask_threshold: float = 0.0,
        confidence_threshold: float = 0.5,
    ) -> list[np.ndarray]:
        pil_image = PILImage.fromarray(image_rgb)
        self._processor.confidence_threshold = confidence_threshold
        device_type = self._device.split(":")[0]  # "cuda" or "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            inference_state = self._processor.set_image(pil_image)
            output = self._processor.set_text_prompt(state=inference_state, prompt=prompt)
        return _extract_masks(output, mask_threshold)


def load_sam3_inferencer(
    checkpoint: str,
    model_type: str = "sam3",
    device: str = "cuda",
) -> Sam3Inferencer:
    if model_type not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose one of: {list(_SUPPORTED_MODELS)}"
        )
    return Sam3Inferencer(checkpoint, device)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_processor(checkpoint: str, device: str) -> Any:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' requested but CUDA is not available in this environment.\n"
            "Fix: reinstall PyTorch matching your CUDA driver version, or pass --device cpu."
        )

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise ImportError(
            "sam3 is not installed. Install it with:\n"
            "  pip install 'sam3 @ git+https://github.com/facebookresearch/sam3.git'\n"
            f"Original error: {exc}"
        ) from exc

    # SAM3's PositionEmbeddingSine hardcodes device="cuda" during position encoding
    # precompute. Patch it to skip precompute on non-CUDA devices; encodings are then
    # computed on-the-fly during inference.
    if device != "cuda":
        from sam3.model import position_encoding as _pe_mod
        _orig_init = _pe_mod.PositionEmbeddingSine.__init__
        def _init_no_precompute(self, *args, **kwargs):
            kwargs["precompute_resolution"] = None
            _orig_init(self, *args, **kwargs)
        _pe_mod.PositionEmbeddingSine.__init__ = _init_no_precompute

    model = build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=False, device=device)
    model.to(device)
    model.eval()
    return Sam3Processor(model, device=device)


def _extract_masks(
    output: dict,
    mask_threshold: float,
) -> list[np.ndarray]:
    raw_masks = output.get("masks", [])
    scores = output.get("scores", [])

    # Pad scores with 1.0 if not provided for every mask
    if len(scores) < len(raw_masks):
        scores = list(scores) + [1.0] * (len(raw_masks) - len(scores))

    masks: list[np.ndarray] = []
    for mask, score in zip(raw_masks, scores):
        if float(score) < mask_threshold:
            continue
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu()
        m = np.asarray(mask, dtype=np.uint8)
        if m.ndim == 3:
            m = m[0]  # (1, H, W) -> (H, W)
        if m.sum() > 0:
            masks.append(m)
    return masks
