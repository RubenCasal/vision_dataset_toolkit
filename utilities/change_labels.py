from __future__ import annotations

from typing import Dict, Optional, List

from dataset_formater.utilities.dataset_interface import (
    DatasetIR,
    Category,
    Annotation,
    Image,
    BBox,
)


def remap_dataset_labels(
    dataset: DatasetIR,
    id_map: Dict[int, Optional[int]],
    drop_empty_images: bool = True,
    verbose: bool = True,
) -> DatasetIR:
    """
    Remap category_ids in a DatasetIR according to id_map.

    - id_map[old_id] = new_id  -> reassign category
    - id_map[old_id] = None    -> drop that annotation
    - old_ids not in id_map    -> keep as-is

    If drop_empty_images is True:
      * Only drop images that originally had annotations and,
        after remapping, end up with zero annotations.
      * Images that were originally background (no annotations) are preserved.
    """

    # Original image_ids that had at least one annotation
    orig_image_ids_with_anns = {a.image_id for a in dataset.annotations}

    # Build helper: old_id -> Category
    old_cats_by_id: Dict[int, Category] = {c.id: c for c in dataset.categories}

    # New categories (merge support)
    new_categories_by_id: Dict[int, Category] = {}
    for old_id, new_id in id_map.items():
        if new_id is None:
            continue
        if new_id in new_categories_by_id:
            continue
        old_cat = old_cats_by_id.get(old_id)
        if old_cat is None:
            continue
        new_categories_by_id[new_id] = Category(
            id=new_id,
            name=old_cat.name,
            supercategory=old_cat.supercategory,
        )

    # Preserve categories not mentioned in id_map
    for cat in dataset.categories:
        if cat.id not in id_map and cat.id not in new_categories_by_id:
            new_categories_by_id[cat.id] = Category(
                id=cat.id,
                name=cat.name,
                supercategory=cat.supercategory,
            )

    # Remap annotations
    new_annotations: List[Annotation] = []
    new_ann_id = 1
    dropped_boxes = 0

    for ann in dataset.annotations:
        old_cid = ann.category_id
        target = id_map.get(old_cid, old_cid)  # not in map -> keep same id

        if target is None:
            dropped_boxes += 1
            continue

        new_annotations.append(
            Annotation(
                id=new_ann_id,
                image_id=ann.image_id,
                category_id=target,
                bbox=ann.bbox,
                iscrowd=ann.iscrowd,
                segmentation=ann.segmentation,
                area=ann.area,
                keypoints=ann.keypoints,
                num_keypoints=ann.num_keypoints,
            )
        )
        new_ann_id += 1

    # Image ids that have annotations after remap
    new_image_ids_with_anns = {a.image_id for a in new_annotations}

    # Drop images only if:
    #   - they had anns originally
    #   - and now have 0 anns
    if drop_empty_images:
        new_images: List[Image] = []
        dropped_images_due_to_remap = 0

        for im in dataset.images:
            if im.id in orig_image_ids_with_anns:
                # image had annotations originally
                if im.id in new_image_ids_with_anns:
                    new_images.append(im)  # still has anns -> keep
                else:
                    # all anns were removed by remap -> drop
                    dropped_images_due_to_remap += 1
            else:
                # background image (never had anns) -> keep
                new_images.append(im)
    else:
        new_images = list(dataset.images)
        dropped_images_due_to_remap = 0

    # Keep only categories that are actually used
    used_cat_ids = {a.category_id for a in new_annotations}
    final_categories = [
        c
        for cid, c in sorted(new_categories_by_id.items(), key=lambda kv: kv[0])
        if cid in used_cat_ids
    ]

    result = DatasetIR(
        images=new_images,
        annotations=new_annotations,
        categories=final_categories,
    )

    if verbose:
        print(
            "[RemapLabels] Categories:",
            len(dataset.categories),
            "->",
            len(result.categories),
        )
        print(
            "[RemapLabels] Annotations:",
            len(dataset.annotations),
            "->",
            len(result.annotations),
        )
        print(
            "[RemapLabels] Images:",
            len(dataset.images),
            "->",
            len(result.images),
            f"(dropped due to remap: {dropped_images_due_to_remap})",
        )
        print("[RemapLabels] Dropped boxes:", dropped_boxes)

    return result
