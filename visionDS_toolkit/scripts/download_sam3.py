from __future__ import annotations

import argparse
from pathlib import Path

# HuggingFace repo IDs for each SAM3 release (gated — requires accepted access request)
SAM3_REPOS = {
    "sam3":   "facebook/sam3",
    "sam3.1": "facebook/sam3.1",
}

SAM3_FILENAMES = {
    "sam3":   "sam3.pt",
    "sam3.1": "sam3.1.pt",
}


def download_checkpoint(repo_id: str, filename: str, out_dir: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is not installed. Install it with:\n"
            "  pip install huggingface_hub\n"
            f"Original error: {exc}"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / filename

    if dst.exists():
        print(f"[INFO] Checkpoint already exists at: {dst}")
        return

    print(f"[INFO] Downloading {filename} from {repo_id} ...")
    print(
        "[INFO] Note: SAM3 checkpoints are gated on HuggingFace.\n"
        "       If this fails, visit the model page, accept the license,\n"
        f"       then run:  huggingface-cli login"
    )

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(out_dir),
    )
    print(f"[INFO] Download completed. Saved at: {local_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download a SAM3 checkpoint from HuggingFace into "
            "visionDS_toolkit_models/sam3/ (or a custom folder).\n\n"
            "SAM3 weights are gated — you must accept the license at\n"
            "https://huggingface.co/facebook/sam3 and run\n"
            "  huggingface-cli login\n"
            "before downloading."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=list(SAM3_REPOS.keys()),
        default="sam3",
        help="SAM3 release to download: 'sam3' or 'sam3.1' (default: sam3).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Optional custom output directory. "
            "If empty, uses <repo_root>/visionDS_toolkit_models/sam3/."
        ),
    )

    args = parser.parse_args()

    model_type = args.model_type
    repo_id = SAM3_REPOS[model_type]
    filename = SAM3_FILENAMES[model_type]

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        this_file = Path(__file__).resolve()
        repo_root = this_file.parents[2]
        out_dir = repo_root / "visionDS_toolkit_models" / "sam3"

    download_checkpoint(repo_id, filename, out_dir)


if __name__ == "__main__":
    main()
