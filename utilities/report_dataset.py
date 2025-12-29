from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import datetime
import numpy as np
import matplotlib.pyplot as plt

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

    num_seg = sum(1 for ann in ds.annotations if ann.segmentation is not None)

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

    for stats in splits_stats.values():
        all_resolutions.extend(stats["resolutions"])
        all_bbox_area_ratios.extend(stats["bbox_area_ratios"])
        all_bbox_centers.extend(stats["bbox_centers"])
        total_seg_annotations += stats["num_seg_annotations"]

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
      - split-wise annotation stats
      - segmentation stats section only if the dataset has masks
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

    has_segmentation = total_seg_annotations > 0
    num_categories = len(all_categories)

    # Generate plots
    res_plot_path = plots_dir / "resolutions_heatmap.png"
    bbox_hist_path = plots_dir / "bbox_size_hist.png"
    bbox_heatmap_path = plots_dir / "bbox_centers_heatmap.png"

    _plot_resolution_heatmap(all_resolutions, res_plot_path)
    _plot_bbox_size_hist(all_bbox_area_ratios, bbox_hist_path)
    _plot_bbox_centers_heatmap(all_bbox_centers, bbox_heatmap_path)

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
    #   3. Resolution statistics
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

        seg_pct = 100.0 * total_seg_annotations / total_annotations
        story.append(
            Paragraph(
                f"Annotations with segmentation: "
                f"<b>{total_seg_annotations}</b> "
                f"({seg_pct:.2f}% of all annotations)",
                normal,
            )
        )

        # Split-wise segmentation ratio
        rows = [["Split", "Segmented annotations", "Segmentation %"]]
        for split_name, stats in splits_stats.items():
            n_ann = stats["num_annotations"]
            n_seg = stats["num_seg_annotations"]
            pct = 100.0 * n_seg / n_ann if n_ann > 0 else 0.0
            rows.append([split_name, str(n_seg), f"{pct:.2f}%"])

        t_seg = Table(rows, hAlign="LEFT")
        t_seg.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]
            )
        )
        story.append(Spacer(1, 6))
        story.append(t_seg)
        story.append(Spacer(1, 12))

    # -----------------------------------------------------------------
    #   6. Split-wise annotation stats
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
