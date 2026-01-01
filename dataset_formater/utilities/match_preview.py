# dataset_formater/utilities/prune_by_preview.py

from pathlib import Path
from typing import Set, Dict, Tuple, Optional, Sequence
from .dataset_interface import DatasetIR, Image, Annotation, Category
from .dataset_folder_interface import DatasetFolder


def _collect_preview_filenames(preview_dir: str | Path) -> Set[str]:
    """Collect file names present in the preview folder."""
    p = Path(preview_dir)
    if not p.exists():
        raise FileNotFoundError(f"Preview directory not found: {p}")

    return {f.name for f in p.iterdir() if f.is_file()}


def prune_dataset_ir_by_preview(
    dataset: DatasetIR,
    preview_dir: str | Path,
) -> DatasetIR:
    """
    Return a new DatasetIR filtered by the images present in preview_dir.

    Only images whose file_name exists in preview_dir will be kept,
    along with their annotations and the categories that are actually used.
    """
    keep_names = _collect_preview_filenames(preview_dir)

    if not keep_names:
        return DatasetIR(images=[], annotations=[], categories=[])

    kept_images: list[Image] = [
        im for im in dataset.images if im.file_name in keep_names
    ]
    kept_image_ids = {im.id for im in kept_images}

    kept_annotations: list[Annotation] = [
        ann for ann in dataset.annotations if ann.image_id in kept_image_ids
    ]

    used_cat_ids = {ann.category_id for ann in kept_annotations}
    kept_categories: list[Category] = [
        cat for cat in dataset.categories if cat.id in used_cat_ids
    ]

    return DatasetIR(
        images=kept_images,
        annotations=kept_annotations,
        categories=kept_categories,
    )


def interactive_prune_datasetfolder_by_preview(
    folder: DatasetFolder,
    dataset_root: str | Path,
    preview_root: Optional[str | Path] = None,
) -> DatasetFolder:
    """
    Use preview/<split> to prune a DatasetFolder in memory.

    - Prints a summary of how many images would be removed per split.
    - Asks the user for confirmation in English.
    - If the user does NOT type 'yes', the original folder is returned.
    - If confirmed, returns a new DatasetFolder with pruned splits.

    This function does NOT touch the filesystem. The caller is responsible
    for writing the pruned dataset to disk using the appropriate dumper.
    """
    dataset_root = Path(dataset_root)

    if preview_root is None:
        preview_root = dataset_root / "preview"
    else:
        preview_root = Path(preview_root)

    def _find_preview_dir(
        base_preview_root: Path,
        candidates: Sequence[str],
    ) -> Optional[Path]:
        """
        Try multiple candidate names for the preview split folder
        (e.g. ['val', 'valid']) and return the first existing one.
        """
        for name in candidates:
            candidate = base_preview_root / name
            if candidate.exists():
                return candidate
        return None

    def _prune_split(
        ds_split: Optional[DatasetIR],
        preview_candidates: Sequence[str],
    ) -> Tuple[Optional[DatasetIR], int]:
        if ds_split is None:
            return None, 0

        split_preview_dir = _find_preview_dir(preview_root, preview_candidates)
        if split_preview_dir is None:
            print(
                f"[WARN] Preview directory not found for split candidates "
                f"{preview_candidates} under {preview_root}. Split will not be pruned."
            )
            return ds_split, 0

        pruned = prune_dataset_ir_by_preview(ds_split, split_preview_dir)
        removed = len(ds_split.images) - len(pruned.images)
        return pruned, removed

    # train → preview/train
    pruned_train, removed_train = _prune_split(folder.train, ["train"])

    # valid/val → probar preview/val y preview/valid
    pruned_valid, removed_valid = _prune_split(folder.valid, ["val", "valid"])

    # test → preview/test
    pruned_test, removed_test = _prune_split(folder.test, ["test"])

    total_removed = removed_train + removed_valid + removed_test

    print(f"Dataset: {dataset_root}")
    print("Images to remove by split:")
    print(f"  train: {removed_train}")
    print(f"  val:   {removed_valid}")
    print(f"  test:  {removed_test}")
    print(f"Total images to remove: {total_removed}")

    if total_removed <= 0:
        print("[INFO] Nothing to prune. Returning original dataset.")
        return folder

    answer = (
        input(
            "Are you sure you want to prune these images from the dataset? "
            "Type 'yes' to confirm: "
        )
        .strip()
        .lower()
    )

    if answer != "yes":
        print("[INFO] Pruning aborted by user. Returning original dataset.")
        return folder

    print("[INFO] Pruning confirmed. Building pruned DatasetFolder in memory...")

    pruned_folder = DatasetFolder.__new__(DatasetFolder)
    pruned_folder.split_dirs = folder.split_dirs
    pruned_folder.train = pruned_train
    pruned_folder.valid = pruned_valid
    pruned_folder.test = pruned_test

    print(
        "[INFO] Pruned DatasetFolder created. You can now dump it to disk "
        "using your preferred format (YOLO / COCO / COCO JSON)."
    )

    return pruned_folder
