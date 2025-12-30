import random
from typing import Optional, Dict, List
from dataset_formater.utilities.dataset_interface import (
    DatasetIR,
    Image,
    Annotation,
    Category,
)
from dataset_formater.utilities.dataset_folder_interface import DatasetFolder


def _subset_split(dataset_split: DatasetIR, keep_percentage: float) -> DatasetIR:
    if dataset_split is None:
        return None

    keep = max(0.0, min(1.0, keep_percentage))
    images = dataset_split.images
    n_total = len(images)

    if n_total == 0 or keep == 0.0:
        return dataset_split

    n_keep = int(round(n_total * keep))
    n_keep = max(1, min(n_total, n_keep))

    selected_images = random.sample(images, n_keep)

    selected_images = sorted(selected_images, key=lambda im: im.id)

    old_to_new_img_id: Dict[int, int] = {}
    new_images: List[Image] = []
    for new_id, im in enumerate(selected_images, start=1):
        old_to_new_img_id[im.id] = new_id
        new_images.append(
            Image(
                id=new_id,
                file_name=im.file_name,
                width=im.width,
                height=im.height,
            )
        )
    selected_ids = set(old_to_new_img_id.keys())
    new_annotations: List[Annotation] = []
    new_ann_id = 1

    for ann in dataset_split.annotations:
        if ann.image_id not in selected_ids:
            continue
        new_annotations.append(
            Annotation(
                id=new_ann_id,
                image_id=old_to_new_img_id[ann.image_id],
                category_id=ann.category_id,
                bbox=ann.bbox,
                iscrowd=ann.iscrowd,
                segmentation=ann.segmentation,
                area=ann.area,
                keypoints=ann.keypoints,
                num_keypoints=ann.num_keypoints,
            )
        )
        new_ann_id += 1

    # Mantener solo categorías usadas
    used_cat_ids = {a.category_id for a in new_annotations}
    new_categories = [
        Category(
            id=cat.id,
            name=cat.name,
            supercategory=cat.supercategory,
        )
        for cat in dataset_split.categories
        if cat.id in used_cat_ids
    ]

    return DatasetIR(
        images=new_images,
        annotations=new_annotations,
        categories=new_categories,
    )


def subset_dataset(dataset: DatasetFolder, keep_percentage: float) -> DatasetFolder:

    print(f"Subsetting dataset to keep {keep_percentage*100:.2f}% of images...")

    if dataset.train is not None:
        before = len(dataset.train.images)
        dataset.train = _subset_split(dataset.train, keep_percentage)
        after = len(dataset.train.images) if dataset.train is not None else 0
        print(f"  Train: {before} -> {after} images")

    if dataset.valid is not None:
        before = len(dataset.valid.images)
        dataset.valid = _subset_split(dataset.valid, keep_percentage)
        after = len(dataset.valid.images) if dataset.valid is not None else 0
        print(f"  Val:   {before} -> {after} images")

    if dataset.test is not None:
        before = len(dataset.test.images)
        dataset.test = _subset_split(dataset.test, keep_percentage)
        after = len(dataset.test.images) if dataset.test is not None else 0
        print(f"  Test:  {before} -> {after} images")

    return dataset
