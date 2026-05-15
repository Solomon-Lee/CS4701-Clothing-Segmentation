"""
CS 4701 — Clothing Segmentation Comparative Study
Matthew Lew's contributions consolidated into one module.

Three sections:
  1. Dataset preparation (DeepFashion2 -> COCO + polygon repair + splits)
  2. Visualization pipeline (side-by-side qualitative grids)
  3. Class-imbalance analysis (instance counts vs per-category metrics)

Designed to run inside the team's Colab notebook with paths under /content/,
but the functions are importable and the paths are configurable.

Run order:
  - Section 1 first (produces *_repaired.json files used by all models)
  - Sections 2 and 3 after Mask2Former / SAM / SegFormer have been trained
    and their per-class results saved.
"""

import gc
import glob
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Shared configuration
# ============================================================

CATEGORIES: List[str] = [
    "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear",
    "long_sleeved_outwear", "vest", "sling", "shorts", "trousers",
    "skirt", "short_sleeved_dress", "long_sleeved_dress",
    "vest_dress", "sling_dress",
]
# COCO ids are 1-indexed; 0 is reserved for background in semantic masks.
CAT_ID_TO_NAME: Dict[int, str] = {i + 1: c for i, c in enumerate(CATEGORIES)}
NAME_TO_CAT_ID: Dict[str, int] = {c: i + 1 for i, c in enumerate(CATEGORIES)}


# ============================================================
# SECTION 1: DATASET PREPARATION
# ============================================================
#
# DeepFashion2 ships annotations as one JSON per image with a per-item
# polygon list. We need:
#   (a) A single COCO-format JSON per split.
#   (b) Polygons in COCO's expected nesting depth (some DF2 polygons come
#       triple-nested as [[[x,y,...]]] instead of [[x,y,...]]), which silently
#       breaks Detectron2 / Mask2Former training.
#   (c) Reproducible train/val/test splits across all three models so the
#       comparison is on identical images.
# ============================================================


def generate_coco_json(
    image_dir: str,
    anno_dir: str,
    output_path: str,
    subset_name: str = "train",
) -> str:
    """Convert a directory of DeepFashion2 per-image JSONs into one COCO file.

    Args:
        image_dir: directory containing .jpg images.
        anno_dir:  directory containing matching .json annotations.
        output_path: where to write the COCO-format JSON.
        subset_name: label used in print output ("train", "validation", ...).

    Returns:
        The output path.
    """
    if not (os.path.isdir(image_dir) and os.path.isdir(anno_dir)):
        raise FileNotFoundError(
            f"Missing dirs for {subset_name}: {image_dir} / {anno_dir}"
        )

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "clothing"}
            for i, name in enumerate(CATEGORIES)
        ],
    }

    ann_id = 1
    img_files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")
    )

    skipped_no_anno = 0
    skipped_unreadable = 0

    for img_file in tqdm(img_files, desc=f"COCO/{subset_name}"):
        img_id = int(os.path.splitext(img_file)[0])
        anno_file = os.path.join(anno_dir, f"{img_id:06d}.json")
        if not os.path.exists(anno_file):
            skipped_no_anno += 1
            continue

        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            skipped_unreadable += 1
            continue
        h, w = img.shape[:2]

        coco["images"].append({
            "id": img_id,
            "file_name": img_file,
            "height": h,
            "width": w,
        })

        with open(anno_file) as f:
            ann_data = json.load(f)

        # DF2 stores items under keys "item1", "item2", ...
        for key, item in ann_data.items():
            if not key.startswith("item"):
                continue
            cat_id = item.get("category_id")
            if not cat_id or cat_id > len(CATEGORIES):
                continue

            seg = item.get("segmentation", [])
            bbox = item.get("bounding_box")
            if not seg or not bbox:
                continue

            # Polygon area from COCO bbox approximation (DF2 doesn't ship area).
            x, y, w_box, h_box = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": seg,
                "bbox": [x, y, w_box, h_box],
                "area": float(w_box * h_box),
                "iscrowd": 0,
            })
            ann_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f)

    print(
        f"[{subset_name}] images={len(coco['images'])} "
        f"annotations={len(coco['annotations'])} "
        f"skipped_no_anno={skipped_no_anno} skipped_unreadable={skipped_unreadable}"
    )
    return output_path


def repair_polygon_nesting(json_path: str) -> str:
    """Flatten triple-nested polygons to COCO's expected depth.

    DF2 occasionally writes `segmentation = [[[x1,y1,...]]]`. COCO expects
    `[[x1,y1,...]]`. Detectron2's data loader fails silently on the deeper
    form. We strip one level of nesting where present.

    Writes to <name>_repaired.json and returns that path.
    """
    with open(json_path) as f:
        data = json.load(f)

    fixed = 0
    for ann in data["annotations"]:
        seg = ann.get("segmentation", [])
        # Triple-nested case: seg[0] is a list whose first elt is also a list.
        if seg and isinstance(seg[0], list) and seg[0] and isinstance(seg[0][0], list):
            ann["segmentation"] = seg[0]
            fixed += 1

    out_path = json_path.replace(".json", "_repaired.json")
    with open(out_path, "w") as f:
        json.dump(data, f)

    print(f"Repaired {fixed} annotations -> {os.path.basename(out_path)}")
    return out_path


def make_eval_split(
    val_json_path: str,
    output_path: str,
    n_images: int = 500,
    seed: int = 42,
) -> str:
    """Sample a fixed subset of the val set used for cross-model evaluation.

    Locks the same image ids across Mask2Former / SAM / SegFormer so per-class
    numbers compare apples to apples. SAM is expensive per-image, so this
    keeps the eval set tractable.
    """
    with open(val_json_path) as f:
        data = json.load(f)

    rng = random.Random(seed)
    chosen = rng.sample(data["images"], min(n_images, len(data["images"])))
    chosen_ids = {img["id"] for img in chosen}

    subset = {
        "images": chosen,
        "annotations": [a for a in data["annotations"] if a["image_id"] in chosen_ids],
        "categories": data["categories"],
    }
    with open(output_path, "w") as f:
        json.dump(subset, f)

    print(
        f"Eval split: {len(subset['images'])} images, "
        f"{len(subset['annotations'])} annotations -> {os.path.basename(output_path)}"
    )
    return output_path


def run_dataset_prep(dataset_root: str = "/content/deepfashion2_dataset") -> Dict[str, str]:
    """Full pipeline: generate COCO, repair polygons, build eval split.

    Returns a dict of output paths used by the rest of the notebook.
    """
    paths = {}
    for split in ("train", "validation"):
        img_dir = os.path.join(dataset_root, split, "image")
        anno_dir = os.path.join(dataset_root, split, "annos")
        # Handle the "nested" structure (some downloads land at split/split/image)
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(dataset_root, split, split, "image")
            anno_dir = os.path.join(dataset_root, split, split, "annos")

        raw = os.path.join(dataset_root, f"{split}_coco.json")
        generate_coco_json(img_dir, anno_dir, raw, subset_name=split)
        paths[f"{split}_json"] = repair_polygon_nesting(raw)
        paths[f"{split}_imgs"] = img_dir

    paths["eval_subset_json"] = make_eval_split(
        paths["validation_json"],
        os.path.join(dataset_root, "val_eval_subset.json"),
        n_images=500,
    )
    return paths


# ============================================================
# SECTION 2: VISUALIZATION PIPELINE
# ============================================================
#
# Goal: produce a 2x2 grid (original + 3 model predictions) on the same
# validation images so the failure modes are visually comparable. We also
# emit a "rare category" gallery, which is the most useful artifact for
# the report: it shows where every model breaks down.
# ============================================================


# Distinct, color-blind-friendly palette for 13 clothing classes + bg.
_PALETTE = np.array([
    [  0,   0,   0],
    [230,  25,  75], [ 60, 180,  75], [255, 225,  25], [  0, 130, 200],
    [245, 130,  48], [145,  30, 180], [ 70, 240, 240], [240,  50, 230],
    [210, 245,  60], [250, 190, 212], [  0, 128, 128], [220, 190, 255],
    [170, 110,  40],
], dtype=np.uint8)


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend a binary mask onto an RGB image."""
    out = image.copy()
    out[mask] = (alpha * color + (1 - alpha) * image[mask]).astype(np.uint8)
    return out


def _render_semantic_label_map(image: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    """Overlay a semantic (per-pixel class id) label map onto an image."""
    out = image.copy()
    for cls_id in range(1, len(_PALETTE)):
        mask = label_map == cls_id
        if mask.any():
            out = _overlay_mask(out, mask, _PALETTE[cls_id])
    return out


def _render_instance_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
    class_ids: List[int],
    scores: Optional[List[float]] = None,
    score_thresh: float = 0.5,
) -> np.ndarray:
    """Render a list of instance masks with class-colored overlays."""
    out = image.copy()
    for i, (mask, cls_id) in enumerate(zip(masks, class_ids)):
        if scores is not None and scores[i] < score_thresh:
            continue
        if cls_id < 1 or cls_id >= len(_PALETTE):
            continue
        out = _overlay_mask(out, mask.astype(bool), _PALETTE[cls_id])
    return out


def _predict_mask2former(predictor, image_bgr: np.ndarray):
    """Run Mask2Former (Detectron2 predictor) -> (masks, class_ids, scores)."""
    out = predictor(image_bgr)
    inst = out["instances"].to("cpu")
    masks = inst.pred_masks.numpy() if inst.has("pred_masks") else np.zeros((0,))
    # Detectron2 class indices are 0-indexed within the registered dataset.
    # Add 1 so they line up with our 1-indexed CAT_ID_TO_NAME palette.
    class_ids = (inst.pred_classes.numpy() + 1).tolist() if len(inst) else []
    scores = inst.scores.numpy().tolist() if len(inst) else []
    return masks, class_ids, scores


def _predict_sam(predictor, image_rgb: np.ndarray,
                 gt_boxes_xyxy: np.ndarray, gt_class_ids: List[int]):
    """Run SAM with GT bbox prompts so its output is comparable to the others."""
    predictor.set_image(image_rgb)
    masks_out, ids_out = [], []
    for box, cid in zip(gt_boxes_xyxy, gt_class_ids):
        m, _, _ = predictor.predict(box=box, multimask_output=False)
        masks_out.append(m[0])
        ids_out.append(cid)
    return masks_out, ids_out


def _predict_segformer(model, processor, image_rgb: np.ndarray, device) -> np.ndarray:
    """Run SegFormer -> dense (H, W) label map at the original resolution."""
    import torch
    from PIL import Image as PILImage

    inputs = processor(images=PILImage.fromarray(image_rgb), return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # SegFormer outputs at 1/4 resolution; upsample back to (H, W).
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=image_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def render_comparison_grid(
    image_rgb: np.ndarray,
    m2f_pred: Tuple[np.ndarray, List[int], List[float]],
    sam_pred: Tuple[List[np.ndarray], List[int]],
    segformer_label_map: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot the 2x2 grid: original | M2F | SAM (prompted) | SegFormer."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original")

    m2f_masks, m2f_classes, m2f_scores = m2f_pred
    axes[0, 1].imshow(_render_instance_masks(image_rgb, m2f_masks, m2f_classes, m2f_scores))
    axes[0, 1].set_title(f"Mask2Former ({len(m2f_classes)} det)")

    sam_masks, sam_classes = sam_pred
    axes[1, 0].imshow(_render_instance_masks(image_rgb, sam_masks, sam_classes))
    axes[1, 0].set_title(f"SAM + GT boxes ({len(sam_masks)} masks)")

    axes[1, 1].imshow(_render_semantic_label_map(image_rgb, segformer_label_map))
    axes[1, 1].set_title("SegFormer")

    for ax in axes.flat:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def build_qualitative_gallery(
    val_json: str,
    val_imgs_dir: str,
    m2f_predictor,
    sam_predictor,
    segformer_model,
    segformer_processor,
    device,
    out_dir: str,
    n_random: int = 8,
    rare_classes: Tuple[str, ...] = ("sling", "short_sleeved_outwear",
                                     "long_sleeved_outwear", "vest_dress"),
) -> None:
    """Produce two galleries:
       (a) `random/`  — n random val images
       (b) `rare/`    — images containing at least one rare-category instance,
                        which is where every model struggles.

    Both galleries are saved as the same 2x2 grid format so we can paste
    them straight into the presentation deck.
    """
    os.makedirs(os.path.join(out_dir, "random"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "rare"), exist_ok=True)

    with open(val_json) as f:
        data = json.load(f)

    img_by_id = {img["id"]: img for img in data["images"]}
    anns_by_img = defaultdict(list)
    for a in data["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    rare_ids = {NAME_TO_CAT_ID[c] for c in rare_classes if c in NAME_TO_CAT_ID}
    rare_imgs = [
        img_by_id[i] for i, anns in anns_by_img.items()
        if any(a["category_id"] in rare_ids for a in anns)
    ]
    print(f"Found {len(rare_imgs)} images containing rare classes.")

    rng = random.Random(42)
    random_imgs = rng.sample(data["images"], min(n_random, len(data["images"])))

    def _process(img_info, bucket: str):
        img_path = os.path.join(val_imgs_dir, img_info["file_name"])
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Mask2Former
        m2f_pred = _predict_mask2former(m2f_predictor, image_bgr)

        # SAM (prompt with GT boxes from this image so the comparison is fair)
        anns = anns_by_img[img_info["id"]]
        boxes_xyxy = np.array([
            [a["bbox"][0], a["bbox"][1],
             a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
            for a in anns
        ], dtype=np.float32)
        cls_ids = [a["category_id"] for a in anns]
        sam_pred = _predict_sam(sam_predictor, image_rgb, boxes_xyxy, cls_ids) \
            if len(boxes_xyxy) else ([], [])

        # SegFormer
        sf_label = _predict_segformer(segformer_model, segformer_processor,
                                      image_rgb, device)

        gt_names = ", ".join(sorted({CAT_ID_TO_NAME[c] for c in cls_ids}))
        save_path = os.path.join(out_dir, bucket, f"cmp_{img_info['id']}.png")
        render_comparison_grid(
            image_rgb, m2f_pred, sam_pred, sf_label,
            title=f"GT: {gt_names}",
            save_path=save_path,
        )

    for img_info in tqdm(random_imgs, desc="random gallery"):
        _process(img_info, "random")
    for img_info in tqdm(rare_imgs[:n_random], desc="rare-class gallery"):
        _process(img_info, "rare")

    print(f"Galleries written to {out_dir}")


# ============================================================
# SECTION 3: CLASS-IMBALANCE ANALYSIS
# ============================================================
#
# The most interesting consistent finding across all three models is that
# rare categories collapse. This section quantifies that:
#   - Count training instances per category.
#   - Pull per-category scores from each model.
#   - Compute Spearman rank correlation between log(count) and score.
#   - Produce two plots: a sorted bar chart of class counts overlaid with
#     per-model scores, and a log-count-vs-score scatter with regression.
# ============================================================


def count_instances_per_class(train_json_path: str) -> Dict[str, int]:
    """Count how many training instances exist per clothing category."""
    with open(train_json_path) as f:
        data = json.load(f)
    counts = Counter(a["category_id"] for a in data["annotations"])
    return {CAT_ID_TO_NAME[cid]: counts.get(cid, 0) for cid in CAT_ID_TO_NAME}


def assemble_per_class_table(
    instance_counts: Dict[str, int],
    m2f_per_class_ap: Dict[str, float],
    sam_per_class_iou: Dict[str, float],
    segformer_per_class_iou: Dict[str, float],
) -> pd.DataFrame:
    """Build the master DataFrame used for the analysis + report table."""
    rows = []
    for name in CATEGORIES:
        rows.append({
            "category": name,
            "train_instances": instance_counts.get(name, 0),
            "mask2former_AP": m2f_per_class_ap.get(name, np.nan),
            "sam_IoU": sam_per_class_iou.get(name, np.nan),
            "segformer_IoU": segformer_per_class_iou.get(name, np.nan),
        })
    df = pd.DataFrame(rows).sort_values("train_instances", ascending=False)
    df["log_instances"] = np.log10(df["train_instances"].clip(lower=1))
    return df


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, NaN-safe, no scipy dependency."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return float("nan")
    xr = pd.Series(x[mask]).rank().to_numpy()
    yr = pd.Series(y[mask]).rank().to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def plot_counts_vs_scores(df: pd.DataFrame, save_path: str) -> None:
    """Bar chart of instance counts with per-model scores overlaid on twin axis."""
    fig, ax1 = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df))

    ax1.bar(x, df["train_instances"], color="#cccccc", label="Train instances")
    ax1.set_ylabel("Training instances")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["category"], rotation=45, ha="right")

    ax2 = ax1.twinx()
    # Mask2Former is in [0, 100] AP; rescale to [0, 1] so it shares an axis
    # with the IoU scores. We label the axis accordingly.
    ax2.plot(x, df["mask2former_AP"] / 100, marker="o",
             color="#1f77b4", label="Mask2Former AP / 100")
    ax2.plot(x, df["sam_IoU"], marker="s",
             color="#d62728", label="SAM IoU")
    ax2.plot(x, df["segformer_IoU"], marker="^",
             color="#2ca02c", label="SegFormer IoU")
    ax2.set_ylabel("Score (AP rescaled / IoU)")
    ax2.set_ylim(0, 1)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.title("Class imbalance vs per-category performance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_imbalance_correlation(df: pd.DataFrame, save_path: str) -> None:
    """Scatter of log(instances) vs score for each model, with Spearman rho."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    pairs = [
        ("Mask2Former (AP/100)", df["mask2former_AP"] / 100, "#1f77b4"),
        ("SAM (IoU)",            df["sam_IoU"],              "#d62728"),
        ("SegFormer (IoU)",      df["segformer_IoU"],        "#2ca02c"),
    ]

    for ax, (label, scores, color) in zip(axes, pairs):
        x = df["log_instances"].to_numpy()
        y = scores.to_numpy()
        ax.scatter(x, y, color=color, s=60)
        for xi, yi, name in zip(x, y, df["category"]):
            if not np.isnan(yi):
                ax.annotate(name, (xi, yi), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        # Linear fit on the non-NaN subset, drawn for visual reference only.
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept, "--", color=color, alpha=0.5)
        rho = spearman(x, y)
        ax.set_title(f"{label}\nSpearman rho = {rho:.2f}")
        ax.set_xlabel("log10(train instances)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Per-category score")

    plt.suptitle("Per-category score vs class frequency", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def run_imbalance_analysis(
    train_json_path: str,
    m2f_per_class_ap: Dict[str, float],
    sam_per_class_iou: Dict[str, float],
    segformer_per_class_iou: Dict[str, float],
    out_dir: str = "/content/imbalance_analysis",
) -> pd.DataFrame:
    """End-to-end pipeline. Returns the per-class DataFrame.

    The dicts should map category name (snake_case, matching CATEGORIES) to
    the per-class metric. Anything missing becomes NaN and is skipped in the
    correlation.
    """
    os.makedirs(out_dir, exist_ok=True)

    counts = count_instances_per_class(train_json_path)
    df = assemble_per_class_table(
        counts, m2f_per_class_ap, sam_per_class_iou, segformer_per_class_iou,
    )

    table_path = os.path.join(out_dir, "per_class_table.csv")
    df.to_csv(table_path, index=False)
    print(f"Per-class table -> {table_path}")
    print(df.to_string(index=False))

    plot_counts_vs_scores(df, os.path.join(out_dir, "counts_vs_scores.png"))
    plot_imbalance_correlation(df, os.path.join(out_dir, "log_count_correlation.png"))

    rhos = {
        "mask2former": spearman(
            df["log_instances"].to_numpy(), df["mask2former_AP"].to_numpy() / 100),
        "sam": spearman(
            df["log_instances"].to_numpy(), df["sam_IoU"].to_numpy()),
        "segformer": spearman(
            df["log_instances"].to_numpy(), df["segformer_IoU"].to_numpy()),
    }
    with open(os.path.join(out_dir, "spearman.json"), "w") as f:
        json.dump(rhos, f, indent=2)
    print(f"Spearman correlations vs log(train instances): {rhos}")
    return df


# ============================================================
# Sanity check entry point
# ============================================================
if __name__ == "__main__":
    # Smoke test: just verify the module parses cleanly and the category list
    # has the expected shape. Real pipelines run inside the Colab notebook.
    assert len(CATEGORIES) == 13
    assert len(CAT_ID_TO_NAME) == 13
    assert NAME_TO_CAT_ID["trousers"] == 8
    print("matthew_contributions.py: OK")
