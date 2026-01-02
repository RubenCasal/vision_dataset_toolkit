# dataset_formater/scripts/download_sam_checkpoint.py

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import requests

# Official SAM URLs (Meta)
SAM_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

# Friendly filenames to store locally
SAM_FILENAMES = {
    "vit_h": "sam_huge.pth",
    "vit_l": "sam_large.pth",
    "vit_b": "sam_base.pth",
}


def download_file(url: str, dst: Path, chunk_size: int = 8192) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    done = int(50 * downloaded / total)
                    sys.stdout.write(
                        f"\rDownloading [{dst.name}]: "
                        f"[{'=' * done}{' ' * (50 - done)}] "
                        f"{downloaded / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MB"
                    )
                    sys.stdout.flush()
    print()  # newline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a SAM checkpoint into dataset_formater/sam_checkpoints."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["vit_h", "vit_l", "vit_b"],
        default="vit_l",
        help="SAM variant to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Optional custom output dir. "
            "If empty, uses <repo_root>/dataset_formater/sam_checkpoints."
        ),
    )

    args = parser.parse_args()

    model_type = args.model_type
    url = SAM_URLS[model_type]
    filename = SAM_FILENAMES[model_type]

    # Default path: <repo_root>/dataset_formater/sam_checkpoints
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        # this file is at dataset_formater/scripts/
        this_file = Path(__file__).resolve()
        repo_root = this_file.parents[2]  # .../<repo_root>/
        out_dir = repo_root / "dataset_formater" / "sam_checkpoints"

    dst = out_dir / filename

    if dst.exists():
        print(f"[INFO] Checkpoint already exists at: {dst}")
        return

    print(f"[INFO] Downloading SAM {model_type} checkpoint to: {dst}")
    print(f"[INFO] URL: {url}")

    download_file(url, dst)

    print("[INFO] Download completed.")


if __name__ == "__main__":
    main()
