from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2

from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)

from dataset_formater.utilities.dataset_interface import (
    DatasetIR,
    Image,
    Annotation,
    Category,
)
from dataset_formater.utilities.dataset_folder_interface import DatasetFolder


# ---------------------------------------------------------------------
#   STATS HELPERS
# ---------------------------------------------------------------------
def _compute_split_stats(ds: DatasetIR) -> Dict:
    """Compute basic stats for one split."""
    num_images = len(ds.images)
    num_annotations = len(ds.annotations)

    img_by_id: Dict[int, Image] = {im.id: im for im in ds.images}

    anns_by_image: Dict[int, List[Annotation]] = {}
    for ann in ds.annotations:
        anns_by_image.setdefault(ann.image_id, []).append(ann)

    anns_per_image: List[int] = [len(anns_by_image.get(im.id, [])) for im in ds.images]

    cat_ann_count: Dict[int, int] = {}
    cat_image_ids: Dict[int, set[int]] = {}
    for ann in ds.annotations:
        cid = ann.category_id
        cat_ann_count[cid] = cat_ann_count.get(cid, 0) + 1
        cat_image_ids.setdefault(cid, set()).add(ann.image_id)

    resolutions: List[Tuple[int, int]] = [(im.width, im.height) for im in ds.images]

    bbox_area_ratios: List[float] = []
    bbox_centers: List[Tuple[float, float]] = []

    # segmentation-related stats (per-annotation, no rasterization)
    mask_area_rel_list: List[float] = []        # mask_area / img_area
    mask_vertices_list: List[int] = []          # nº vértices del polígono
    mask_bbox_ratio_list: List[float] = []      # mask_area / bbox_area

    num_seg = 0

    for ann in ds.annotations:
        im = img_by_id.get(ann.image_id)
        if im is None or im.width <= 0 or im.height <= 0:
            continue

        b = ann.bbox
        img_area = im.width * im.height
        box_area = max(0.0, b.width) * max(0.0, b.height)
        if img_area > 0:
            bbox_area_ratios.append(box_area / img_area)

        cx = (b.x + b.width / 2.0) / im.width
        cy = (b.y + b.height / 2.0) / im.height
        bbox_centers.append((cx, cy))

        # --- segmentation-related stats ---
        if ann.segmentation is not None:
            num_seg += 1
            ann_area = getattr(ann, "area", None)
            if ann_area is not None and img_area > 0:
                mask_area_rel_list.append(float(ann_area) / float(img_area))

            if ann.segmentation:
                verts = len(ann.segmentation) // 2
                if verts > 0:
                    mask_vertices_list.append(verts)

            if ann_area is not None and box_area > 0:
                mask_bbox_ratio_list.append(float(ann_area) / float(box_area))

    return {
        "num_images": num_images,
        "num_annotations": num_annotations,
        "anns_per_image": anns_per_image,
        "cat_ann_count": cat_ann_count,
        "cat_image_ids": cat_image_ids,
        "resolutions": resolutions,
        "bbox_area_ratios": bbox_area_ratios,
        "bbox_centers": bbox_centers,
        "num_seg_annotations": num_seg,
        "mask_area_rel_list": mask_area_rel_list,
        "mask_vertices_list": mask_vertices_list,
        "mask_bbox_ratio_list": mask_bbox_ratio_list,
    }


def _aggregate_global_stats(
    splits_stats: Dict[str, Dict],
    folder: DatasetFolder,
) -> Dict:
    """Aggregate stats across all splits."""
    total_images = sum(s["num_images"] for s in splits_stats.values())
    total_annotations = sum(s["num_annotations"] for s in splits_stats.values())

    all_categories: Dict[int, Category] = {}
    for ds in (folder.train, folder.valid, folder.test):
        if ds is None:
            continue
        for cat in ds.categories:
            if cat.id not in all_categories:
                all_categories[cat.id] = cat

    global_cat_ann_count: Dict[int, int] = {}
    global_cat_image_ids: Dict[int, set[int]] = {}
    for stats in splits_stats.values():
        cat_ann_count = stats["cat_ann_count"]
        cat_image_ids = stats["cat_image_ids"]
        for cid, count in cat_ann_count.items():
            global_cat_ann_count[cid] = global_cat_ann_count.get(cid, 0) + count
        for cid, img_ids in cat_image_ids.items():
            global_cat_image_ids.setdefault(cid, set()).update(img_ids)

    all_resolutions: List[Tuple[int, int]] = []
    all_bbox_area_ratios: List[float] = []
    all_bbox_centers: List[Tuple[float, float]] = []
    total_seg_annotations = 0

    # global segmentation-annotation stats (no raster)
    all_mask_area_rel: List[float] = []
    all_mask_vertices: List[int] = []
    all_mask_bbox_ratio: List[float] = []

    for stats in splits_stats.values():
        all_resolutions.extend(stats["resolutions"])
        all_bbox_area_ratios.extend(stats["bbox_area_ratios"])
        all_bbox_centers.extend(stats["bbox_centers"])
        total_seg_annotations += stats["num_seg_annotations"]

        all_mask_area_rel.extend(stats["mask_area_rel_list"])
        all_mask_vertices.extend(stats["mask_vertices_list"])
        all_mask_bbox_ratio.extend(stats["mask_bbox_ratio_list"])

    return {
        "total_images": total_images,
        "total_annotations": total_annotations,
        "all_categories": all_categories,
        "global_cat_ann_count": global_cat_ann_count,
        "global_cat_image_ids": global_cat_image_ids,
        "all_resolutions": all_resolutions,
        "all_bbox_area_ratios": all_bbox_area_ratios,
        "all_bbox_centers": all_bbox_centers,
        "total_seg_annotations": total_seg_annotations,
        "all_mask_area_rel": all_mask_area_rel,
        "all_mask_vertices": all_mask_vertices,
        "all_mask_bbox_ratio": all_mask_bbox_ratio,
    }


def _compute_segmentation_pixel_stats(
    folder: DatasetFolder,
    grid_size: int = 50,
) -> Dict:
    """
    Global pixel-level segmentation stats:
      - total labeled pixels vs total image pixels (coverage)
      - coverage per image
      - overlap pixels (pixels belonging to >=2 instances)
      - number of images with masks / with overlaps
      - heatmap of labeled pixel density in normalized coordinates
    """
    total_image_pixels = 0
    total_labeled_pixels = 0
    overlap_pixels = 0
    coverage_per_image: List[float] = []
    images_with_masks = 0
    images_with_overlap = 0

    heatmap = np.zeros((grid_size, grid_size), dtype=np.float64)

    for ds in (folder.train, folder.valid, folder.test):
        if ds is None:
            continue

        anns_by_image: Dict[int, List[Annotation]] = {}
        for ann in ds.annotations:
            if ann.segmentation is not None and ann.segmentation:
                anns_by_image.setdefault(ann.image_id, []).append(ann)

        for im in ds.images:
            seg_anns = anns_by_image.get(im.id, [])
            if not seg_anns:
                continue

            h, w = im.height, im.width
            if h <= 0 or w <= 0:
                continue

            mask_union = np.zeros((h, w), dtype=np.uint8)
            mask_count = np.zeros((h, w), dtype=np.uint16)

            for ann in seg_anns:
                poly_flat = ann.segmentation
                if not poly_flat or len(poly_flat) < 6:
                    continue
                coords = np.array(poly_flat, dtype=np.float32).reshape(-1, 2)
                coords_int = np.round(coords).astype(np.int32)

                m = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(m, [coords_int], 1)

                mask_union |= m
                mask_count = mask_count + m

            labeled = int(mask_union.sum())
            if labeled == 0:
                continue

            images_with_masks += 1
            img_pixels = h * w
            total_image_pixels += img_pixels
            total_labeled_pixels += labeled

            coverage_per_image.append(labeled / float(img_pixels))

            ov_mask = mask_count >= 2
            ov_pixels = int(ov_mask.sum())
            overlap_pixels += ov_pixels
            if ov_pixels > 0:
                images_with_overlap += 1

            # accumulate heatmap in normalized space
            mask_union_f = mask_union.astype(np.float32)
            small = cv2.resize(
                mask_union_f,
                (grid_size, grid_size),
                interpolation=cv2.INTER_AREA,
            )
            heatmap += small

    return {
        "total_image_pixels": total_image_pixels,
        "total_labeled_pixels": total_labeled_pixels,
        "coverage_per_image": coverage_per_image,
        "overlap_pixels": overlap_pixels,
        "images_with_masks": images_with_masks,
        "images_with_overlap": images_with_overlap,
        "heatmap": heatmap,
    }


# ---------------------------------------------------------------------
#   PLOTS HELPERS
# ---------------------------------------------------------------------
def _plot_resolution_heatmap(
    resolutions: List[Tuple[int, int]],
    out_path: Path,
) -> None:
    if not resolutions:
        return
    widths = np.array([w for (w, _) in resolutions], dtype=float)
    heights = np.array([h for (_, h) in resolutions], dtype=float)

    bins_w = 20
    bins_h = 20
    H, xedges, yedges = np.histogram2d(widths, heights, bins=[bins_w, bins_h])

    plt.figure()
    plt.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.title("Resolution 2D histogram")
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_bbox_size_hist(
    bbox_area_ratios: List[float],
    out_path: Path,
) -> None:
    if not bbox_area_ratios:
        return

    ratios = np.clip(np.array(bbox_area_ratios, dtype=float), 0.0, 1.0)

    plt.figure()
    plt.hist(ratios, bins=30)
    plt.xlabel("BBox area / image area")
    plt.ylabel("Count")
    plt.title("Bounding box relative area histogram")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_bbox_centers_heatmap(
    centers: List[Tuple[float, float]],
    out_path: Path,
    grid_size: int = 50,
) -> None:
    if not centers:
        return

    heat = np.zeros((grid_size, grid_size), dtype=float)
    for cx, cy in centers:
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            continue
        ix = min(grid_size - 1, int(cx * grid_size))
        iy = min(grid_size - 1, int(cy * grid_size))
        heat[iy, ix] += 1.0

    if heat.max() > 0:
        heat = heat / heat.max()

    plt.figure()
    plt.imshow(heat, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
    plt.xlabel("Normalized X")
    plt.ylabel("Normalized Y")
    plt.title("Bounding box center heatmap")
    plt.colorbar(label="Relative density")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_mask_pixel_heatmap(
    heatmap: np.ndarray,
    out_path: Path,
) -> None:
    if heatmap is None or heatmap.size == 0 or np.all(heatmap == 0):
        return

    plt.figure()
    plt.imshow(
        heatmap,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="equal",
    )
    plt.xlabel("Normalized X")
    plt.ylabel("Normalized Y")
    plt.title("Labeled pixel density heatmap")
    plt.colorbar(label="Relative density (aggregated)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_mask_coverage_hist(
    coverage_per_image: List[float],
    out_path: Path,
) -> None:
    if not coverage_per_image:
        return

    vals = np.clip(np.array(coverage_per_image, dtype=float), 0.0, 1.0)

    plt.figure()
    plt.hist(vals, bins=30)
    plt.xlabel("Labeled pixels / image pixels")
    plt.ylabel("Images")
    plt.title("Per-image mask coverage")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------
#   PDF REPORT
# ---------------------------------------------------------------------
def generate_dataset_report_pdf(
    folder: DatasetFolder,
    dataset_root: str | Path,
    dataset_format: str,
    output_filename: str = "report_dataset.pdf",
    plots_dirname: str = "report_plots",
) -> None:
    """
    Generate a nicely formatted PDF report for a dataset:
      - overview (images, annotations, categories, segmentation presence)
      - split-level stats
      - category table with percentages
      - resolution stats + 2D histogram
      - bbox stats + hist + heatmap
      - segmentation stats (if masks exist):
          * pixel coverage
          * mask size distribution
          * polygon complexity (vertices)
          * mask/bbox ratio
          * overlaps and occlusions
          * spatial distribution (heatmap of labeled pixels)
      - split-wise annotation stats
    """
    dataset_root = Path(dataset_root)
    dataset_name = dataset_root.name

    report_path = dataset_root / output_filename
    plots_dir = dataset_root / plots_dirname
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Collect splits
    splits: Dict[str, DatasetIR] = {}
    if folder.train is not None:
        splits["train"] = folder.train
    if folder.valid is not None:
        splits["val"] = folder.valid
    if folder.test is not None:
        splits["test"] = folder.test

    if not splits:
        raise ValueError("No splits found in DatasetFolder. Nothing to report.")

    # Per-split stats
    splits_stats: Dict[str, Dict] = {
        split_name: _compute_split_stats(ds) for split_name, ds in splits.items()
    }

    # Global aggregation
    global_stats = _aggregate_global_stats(splits_stats, folder)

    total_images = global_stats["total_images"]
    total_annotations = global_stats["total_annotations"]
    all_categories: Dict[int, Category] = global_stats["all_categories"]
    global_cat_ann_count: Dict[int, int] = global_stats["global_cat_ann_count"]
    global_cat_image_ids: Dict[int, set[int]] = global_stats["global_cat_image_ids"]
    all_resolutions: List[Tuple[int, int]] = global_stats["all_resolutions"]
    all_bbox_area_ratios: List[float] = global_stats["all_bbox_area_ratios"]
    all_bbox_centers: List[Tuple[float, float]] = global_stats["all_bbox_centers"]
    total_seg_annotations: int = global_stats["total_seg_annotations"]

    all_mask_area_rel: List[float] = global_stats["all_mask_area_rel"]
    all_mask_vertices: List[int] = global_stats["all_mask_vertices"]
    all_mask_bbox_ratio: List[float] = global_stats["all_mask_bbox_ratio"]

    has_segmentation = total_seg_annotations > 0
    num_categories = len(all_categories)

    # Generate base plots
    res_plot_path = plots_dir / "resolutions_heatmap.png"
    bbox_hist_path = plots_dir / "bbox_size_hist.png"
    bbox_heatmap_path = plots_dir / "bbox_centers_heatmap.png"

    _plot_resolution_heatmap(all_resolutions, res_plot_path)
    _plot_bbox_size_hist(all_bbox_area_ratios, bbox_hist_path)
    _plot_bbox_centers_heatmap(all_bbox_centers, bbox_heatmap_path)

    # Segmentation pixel-level plots (only if masks exist)
    seg_pixel_stats = None
    mask_heatmap_path = plots_dir / "mask_pixels_heatmap.png"
    mask_coverage_hist_path = plots_dir / "mask_coverage_hist.png"

    if has_segmentation:
        seg_pixel_stats = _compute_segmentation_pixel_stats(folder)
        _plot_mask_pixel_heatmap(seg_pixel_stats["heatmap"], mask_heatmap_path)
        _plot_mask_coverage_hist(
            seg_pixel_stats["coverage_per_image"],
            mask_coverage_hist_path,
        )

    # -----------------------------------------------------------------
    #   ReportLab styles & doc
    # -----------------------------------------------------------------
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    h2 = ParagraphStyle(
        "Heading2",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=6,
    )
    h3 = ParagraphStyle(
        "Heading3",
        parent=styles["Heading3"],
        fontSize=12,
        leading=16,
        spaceBefore=8,
        spaceAfter=4,
    )
    normal = styles["Normal"]
    small = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
    )

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    story: List = []

    # -----------------------------------------------------------------
    #   0. Title
    # -----------------------------------------------------------------
    story.append(Paragraph(f"Dataset report: {dataset_name}", title_style))
    story.append(
        Paragraph(
            f"Path: <b>{dataset_root.resolve()}</b><br/>"
            f"Format: <b>{dataset_format}</b><br/>"
            f"Generated at: {datetime.datetime.now().isoformat(timespec='seconds')}",
            normal,
        )
    )
    story.append(Spacer(1, 18))

    # -----------------------------------------------------------------
    #   1. Overview
    # -----------------------------------------------------------------
    story.append(Paragraph("1. Overview", h2))

    overview_lines = [
        f"Total images: <b>{total_images}</b>",
        f"Total annotations (bounding boxes): <b>{total_annotations}</b>",
        f"Total categories: <b>{num_categories}</b>",
        f"Contains segmentation masks: "
        + ("<b>YES</b>" if has_segmentation else "<b>NO</b>"),
    ]
    for txt in overview_lines:
        story.append(Paragraph("• " + txt, normal))
    story.append(Spacer(1, 12))

    # 1.1 Split table
    story.append(Paragraph("1.1 Images and annotations per split", h3))
    split_data = [["Split", "Images", "Annotations", "Mean boxes / image"]]
    for split_name, stats in splits_stats.items():
        n_img = stats["num_images"]
        n_ann = stats["num_annotations"]
        mean_boxes = n_ann / n_img if n_img > 0 else 0.0
        split_data.append([split_name, str(n_img), str(n_ann), f"{mean_boxes:.3f}"])

    t = Table(split_data, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(t)

    global_mean_boxes = total_annotations / total_images if total_images > 0 else 0.0
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            f"Global mean boxes per image: <b>{global_mean_boxes:.3f}</b>",
            normal,
        )
    )
    story.append(Spacer(1, 12))

    # -----------------------------------------------------------------
    #   2. Categories
    # -----------------------------------------------------------------
    story.append(Paragraph("2. Categories", h2))

    if num_categories == 0:
        story.append(Paragraph("No categories found.", normal))
    else:
        data = [
            [
                "Category ID",
                "Name",
                "Boxes",
                "Images",
                "Boxes %",
                "Images %",
            ]
        ]
        for cid in sorted(all_categories.keys()):
            cat = all_categories[cid]
            box_count = global_cat_ann_count.get(cid, 0)
            img_count = len(global_cat_image_ids.get(cid, set()))
            box_pct = (
                100.0 * box_count / total_annotations if total_annotations > 0 else 0.0
            )
            img_pct = 100.0 * img_count / total_images if total_images > 0 else 0.0
            data.append(
                [
                    str(cid),
                    cat.name,
                    str(box_count),
                    str(img_count),
                    f"{box_pct:5.2f}%",
                    f"{img_pct:5.2f}%",
                ]
            )

        t_cat = Table(data, hAlign="LEFT")
        t_cat.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(t_cat)
    story.append(Spacer(1, 12))

    # -----------------------------------------------------------------
    #   3. Image resolution statistics
    # -----------------------------------------------------------------
    story.append(Paragraph("3. Image resolution statistics", h2))

    if not all_resolutions:
        story.append(Paragraph("No resolution data available.", normal))
    else:
        widths = [w for (w, _) in all_resolutions]
        heights = [h for (_, h) in all_resolutions]
        min_w, max_w = min(widths), max(widths)
        min_h, max_h = min(heights), max(heights)
        mean_w = sum(widths) / len(widths)
        mean_h = sum(heights) / len(heights)

        story.append(Paragraph(f"Min resolution: <b>{min_w} × {min_h}</b> px", normal))
        story.append(Paragraph(f"Max resolution: <b>{max_w} × {max_h}</b> px", normal))
        story.append(
            Paragraph(f"Mean resolution: <b>{mean_w:.1f} × {mean_h:.1f}</b> px", normal)
        )
        story.append(Spacer(1, 8))

        if res_plot_path.exists():
            story.append(Paragraph("3.1 Resolution 2D histogram", h3))
            story.append(Spacer(1, 4))
            story.append(RLImage(str(res_plot_path), width=360, height=270))
        else:
            story.append(Paragraph("Resolution plot not available.", normal))
    story.append(Spacer(1, 12))

    # -----------------------------------------------------------------
    #   4. Bounding box statistics
    # -----------------------------------------------------------------
    story.append(Paragraph("4. Bounding box statistics", h2))

    if total_annotations == 0:
        story.append(Paragraph("No bounding boxes available.", normal))
    else:
        all_anns_per_image: List[int] = []
        for stats in splits_stats.values():
            all_anns_per_image.extend(stats["anns_per_image"])

        if all_anns_per_image:
            min_boxes = min(all_anns_per_image)
            max_boxes = max(all_anns_per_image)
            mean_boxes = sum(all_anns_per_image) / len(all_anns_per_image)
        else:
            min_boxes = max_boxes = mean_boxes = 0

        story.append(
            Paragraph(
                f"Boxes per image (global): "
                f"min=<b>{min_boxes}</b>, max=<b>{max_boxes}</b>, "
                f"mean=<b>{mean_boxes:.3f}</b>",
                normal,
            )
        )

        zero_box_images = sum(1 for v in all_anns_per_image if v == 0)
        zero_box_pct = (
            100.0 * zero_box_images / total_images if total_images > 0 else 0.0
        )
        story.append(
            Paragraph(
                f"Images with 0 boxes: <b>{zero_box_images}</b> "
                f"({zero_box_pct:.2f}%)",
                normal,
            )
        )

        if all_bbox_area_ratios:
            ratios = all_bbox_area_ratios
            min_r = min(ratios)
            max_r = max(ratios)
            mean_r = sum(ratios) / len(ratios)
            small_boxes = sum(1 for r in ratios if r < 0.01)
            large_boxes = sum(1 for r in ratios if r > 0.5)

            story.append(
                Paragraph(
                    "BBox relative area (box_area / img_area): "
                    f"min=<b>{min_r:.5f}</b>, max=<b>{max_r:.5f}</b>, "
                    f"mean=<b>{mean_r:.5f}</b>",
                    normal,
                )
            )
            story.append(
                Paragraph(
                    f"Small boxes (&lt;1% of image area): "
                    f"<b>{small_boxes}</b> "
                    f"({100.0 * small_boxes / total_annotations:.2f}%)",
                    small,
                )
            )
            story.append(
                Paragraph(
                    f"Large boxes (&gt;50% of image area): "
                    f"<b>{large_boxes}</b> "
                    f"({100.0 * large_boxes / total_annotations:.2f}%)",
                    small,
                )
            )

        story.append(Spacer(1, 8))

        # 4.1 size histogram
        if bbox_hist_path.exists():
            story.append(Paragraph("4.1 BBox size histogram (relative area)", h3))
            story.append(Spacer(1, 4))
            story.append(RLImage(str(bbox_hist_path), width=360, height=270))
            story.append(Spacer(1, 8))

        # 4.2 center heatmap
        if bbox_heatmap_path.exists():
            story.append(Paragraph("4.2 BBox center heatmap", h3))
            story.append(Spacer(1, 4))
            story.append(RLImage(str(bbox_heatmap_path), width=360, height=270))

    story.append(Spacer(1, 12))

    # -----------------------------------------------------------------
    #   5. Segmentation statistics (only if there are masks)
    # -----------------------------------------------------------------
    if has_segmentation and total_annotations > 0:
        story.append(Paragraph("5. Segmentation statistics", h2))

        
        # 5.2 Pixel coverage (mask coverage)
        if seg_pixel_stats is not None and seg_pixel_stats["total_image_pixels"] > 0:
            total_img_px = seg_pixel_stats["total_image_pixels"]
            total_lbl_px = seg_pixel_stats["total_labeled_pixels"]
            cov_global = total_lbl_px / float(total_img_px)

            cov_vals = seg_pixel_stats["coverage_per_image"]
            if cov_vals:
                cov_arr = np.array(cov_vals, dtype=float)
                cov_min = cov_arr.min()
                cov_max = cov_arr.max()
                cov_mean = cov_arr.mean()
                cov_med = float(np.median(cov_arr))
            else:
                cov_min = cov_max = cov_mean = cov_med = 0.0

            story.append(
                Paragraph(
                    "Global pixel coverage (union of all masks): "
                    f"<b>{100.0 * cov_global:.2f}%</b> of all image pixels.",
                    normal,
                )
            )
            story.append(
                Paragraph(
                    "Per-image mask coverage (labeled pixels / image pixels): "
                    f"min=<b>{100.0 * cov_min:.2f}%</b>, "
                    f"max=<b>{100.0 * cov_max:.2f}%</b>, "
                    f"mean=<b>{100.0 * cov_mean:.2f}%</b>, "
                    f"median=<b>{100.0 * cov_med:.2f}%</b>.",
                    small,
                )
            )
            story.append(Spacer(1, 6))

            if mask_coverage_hist_path.exists():
                story.append(Paragraph("5.2.1 Coverage histogram", h3))
                story.append(Spacer(1, 4))
                story.append(RLImage(str(mask_coverage_hist_path), width=360, height=270))
                story.append(Spacer(1, 8))

        # 5.3 Mask size distribution (instance size by mask)
        if all_mask_area_rel:
            arr = np.clip(np.array(all_mask_area_rel, dtype=float), 0.0, 1.0)
            m_min = arr.min()
            m_max = arr.max()
            m_mean = arr.mean()
            m_med = float(np.median(arr))
            small_frac = 100.0 * np.mean(arr < 0.001)  # <0.1% img
            large_frac = 100.0 * np.mean(arr > 0.5)    # >50% img

            story.append(Paragraph("5.3 Mask instance size (relative area)", h3))
            story.append(
                Paragraph(
                    "Mask area / image area: "
                    f"min=<b>{100.0 * m_min:.4f}%</b>, "
                    f"max=<b>{100.0 * m_max:.4f}%</b>, "
                    f"mean=<b>{100.0 * m_mean:.4f}%</b>, "
                    f"median=<b>{100.0 * m_med:.4f}%</b>.",
                    normal,
                )
            )
            story.append(
                Paragraph(
                    f"Masks &lt;0.1% of image area: <b>{small_frac:.2f}%</b> of instances.",
                    small,
                )
            )
            story.append(
                Paragraph(
                    f"Masks &gt;50% of image area: <b>{large_frac:.2f}%</b> of instances.",
                    small,
                )
            )
            story.append(Spacer(1, 8))

        # 5.4 Polygon complexity + mask/bbox tightness
        if all_mask_vertices:
            v_arr = np.array(all_mask_vertices, dtype=float)
            v_min = v_arr.min()
            v_max = v_arr.max()
            v_mean = v_arr.mean()
            v_med = float(np.median(v_arr))

            story.append(Paragraph("5.4 Mask polygon complexity", h3))
            story.append(
                Paragraph(
                    "Vertices per mask polygon: "
                    f"min=<b>{v_min:.0f}</b>, "
                    f"max=<b>{v_max:.0f}</b>, "
                    f"mean=<b>{v_mean:.2f}</b>, "
                    f"median=<b>{v_med:.0f}</b>.",
                    normal,
                )
            )
            story.append(Spacer(1, 4))

        if all_mask_bbox_ratio:
            r_arr = np.array(all_mask_bbox_ratio, dtype=float)
            r_min = r_arr.min()
            r_max = r_arr.max()
            r_mean = r_arr.mean()
            r_med = float(np.median(r_arr))

            story.append(Paragraph("5.5 Mask / bbox area ratio (tightness)", h3))
            story.append(
                Paragraph(
                    "mask_area / bbox_area: "
                    f"min=<b>{r_min:.4f}</b>, "
                    f"max=<b>{r_max:.4f}</b>, "
                    f"mean=<b>{r_mean:.4f}</b>, "
                    f"median=<b>{r_med:.4f}</b>.",
                    normal,
                )
            )
            story.append(Spacer(1, 8))

        # 5.6 Overlaps and occlusions
        if seg_pixel_stats is not None and seg_pixel_stats["total_labeled_pixels"] > 0:
            ov_px = seg_pixel_stats["overlap_pixels"]
            lbl_px = seg_pixel_stats["total_labeled_pixels"]
            ov_ratio = ov_px / float(lbl_px)

            img_with_masks = seg_pixel_stats["images_with_masks"]
            img_with_ov = seg_pixel_stats["images_with_overlap"]
            img_ov_pct = (
                100.0 * img_with_ov / img_with_masks if img_with_masks > 0 else 0.0
            )

            story.append(Paragraph("5.6 Overlaps and occlusions", h3))
            story.append(
                Paragraph(
                    "Pixels belonging to &ge;2 masks: "
                    f"<b>{ov_px}</b> "
                    f"({100.0 * ov_ratio:.2f}% of labeled pixels).",
                    normal,
                )
            )
            story.append(
                Paragraph(
                    "Images with at least one overlapping region: "
                    f"<b>{img_with_ov}</b> / <b>{img_with_masks}</b> "
                    f"({img_ov_pct:.2f}%).",
                    small,
                )
            )
            story.append(Spacer(1, 8))

        # 5.7 Spatial distribution of labeled pixels
        if mask_heatmap_path.exists():
            story.append(Paragraph("5.7 Spatial distribution of labeled pixels", h3))
            story.append(Spacer(1, 4))
            story.append(RLImage(str(mask_heatmap_path), width=360, height=270))
            story.append(Spacer(1, 12))

    # -----------------------------------------------------------------
    #   6. Split-wise annotation statistics
    # -----------------------------------------------------------------
    story.append(Paragraph("6. Split-wise annotation statistics", h2))

    idx = 1
    for split_name in ["train", "val", "test"]:
        if split_name not in splits_stats:
            continue
        stats = splits_stats[split_name]
        n_img = stats["num_images"]
        n_ann = stats["num_annotations"]
        anns_per_img = stats["anns_per_image"]
        if anns_per_img:
            min_b = min(anns_per_img)
            max_b = max(anns_per_img)
            mean_b = sum(anns_per_img) / len(anns_per_img)
        else:
            min_b = max_b = mean_b = 0

        story.append(Paragraph(f"6.{idx} {split_name} split", h3))
        idx += 1
        story.append(
            Paragraph(
                f"Images: <b>{n_img}</b>, "
                f"Annotations: <b>{n_ann}</b>, "
                f"Boxes per image: "
                f"min=<b>{min_b}</b>, max=<b>{max_b}</b>, "
                f"mean=<b>{mean_b:.3f}</b>",
                normal,
            )
        )
        story.append(Spacer(1, 6))

    # Build PDF
    doc.build(story)
