import argparse
from pathlib import Path
from typing import Optional

from dataset_formater.utilities.dataset_folder_interface import DatasetFolder
from dataset_formater.utilities.report_dataset import generate_dataset_report_pdf

# =========================
# DEFAULT CONSTANTS
# =========================
DATASET_PATH: str = ""  # e.g. "data/source_yolo"
DATASET_FORMAT: str = ""  # "yolo" | "coco" | "coco_json"
REPORT_FILENAME: str = "report_dataset.pdf"
PLOTS_DIRNAME: str = "report_plots"


def run_report(
    dataset_path: str,
    dataset_format: str,
    report_filename: str = REPORT_FILENAME,
    plots_dirname: str = PLOTS_DIRNAME,
) -> None:
    """
    Build DatasetFolder and generate a PDF report for the whole dataset.
    """
    dataset_root = Path(dataset_path)
    print(f"Init dataset report for {dataset_format} at {dataset_root}")

    ds_folder = DatasetFolder(path=str(dataset_root), dataset_type=dataset_format)

    generate_dataset_report_pdf(
        folder=ds_folder,
        dataset_root=dataset_root,
        dataset_format=dataset_format,
        output_filename=report_filename,
        plots_dirname=plots_dirname,
    )

    print(f"Report generated at: {dataset_root / report_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a PDF report for a dataset (YOLO, COCO, COCO JSON)."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DATASET_PATH,
        help="Path to the dataset root folder (parent of train/val/test).",
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        default=DATASET_FORMAT,
        choices=["yolo", "coco", "coco_json"],
        help="Format of the dataset (yolo, coco, coco_json).",
    )
    parser.add_argument(
        "--report_filename",
        type=str,
        default=REPORT_FILENAME,
        help="Name of the output PDF file (default: report_dataset.pdf).",
    )
    parser.add_argument(
        "--plots_dirname",
        type=str,
        default=PLOTS_DIRNAME,
        help="Directory (inside dataset root) where plots will be stored.",
    )

    args = parser.parse_args()

    if not args.dataset_path:
        raise ValueError(
            "dataset_path is empty. Set DATASET_PATH in the script or pass --dataset_path."
        )
    if not args.dataset_format:
        raise ValueError(
            "dataset_format is empty. Set DATASET_FORMAT in the script or pass "
            "--dataset_format (yolo|coco|coco_json)."
        )

    run_report(
        dataset_path=args.dataset_path,
        dataset_format=args.dataset_format,
        report_filename=args.report_filename,
        plots_dirname=args.plots_dirname,
    )

if __name__ == "__main__":
    main()
