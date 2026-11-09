"""Check the frozen SAR adaptation split for cross-partition rotations and reflections."""
import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import zlib

import numpy as np
from PIL import Image

from plan_sar_adaptation_data import MANIFESTS, OUTPUT as PLAN, ROOT, read_verified, sha256


ACQUISITION = MANIFESTS / "sar_adaptation_data_acquisition_20261107.json"
THUMBNAIL_SIDE = 64
NCC_THRESHOLD = 0.80
BOX_TOLERANCE = 3
ORIENTATIONS = [(k, flip) for k in range(4) for flip in (False, True)]


def orient(array, rotation, reflection):
    rotated = np.rot90(array, rotation)
    return np.fliplr(rotated) if reflection else rotated


def normalized_pixels(array):
    flat = array.astype(np.float32).ravel()
    flat -= flat.mean()
    norm = np.linalg.norm(flat)
    return flat / norm if norm else flat


def orientation_info(rotation, reflection):
    return {"rotation_counterclockwise_degrees": rotation * 90,
            "left_right_reflection_after_rotation": reflection}


def compare_boxes(train_boxes, validation_boxes, shape, rotation, reflection):
    """Match transformed axis-aligned boxes by sorted corner coordinates."""
    train_corners = sorted((x, y, x + w, y + h) for x, y, w, h in train_boxes)
    transformed = []
    for x, y, w, h in validation_boxes:
        height, width = shape
        x0, y0, x1, y1 = x, y, x + w, y + h
        for _ in range(rotation):
            x0, y0, x1, y1 = y0, width - x1, y1, width - x0
            height, width = width, height
        if reflection:
            x0, x1 = width - x1, width - x0
        transformed.append((x0, y0, x1, y1))
    if not train_corners or len(train_corners) != len(transformed):
        return None
    return float(np.abs(np.asarray(train_corners) - np.asarray(sorted(transformed))).max())


def audit():
    acquisition = json.loads(ACQUISITION.read_bytes())
    if acquisition["status"] != "passed":
        raise ValueError("The fixed data acquisition did not pass")
    records_path = ROOT / acquisition["records"]["path"]
    raw_records = read_verified(records_path, acquisition["records"]["sha256"])
    records = [json.loads(line) for line in raw_records.splitlines() if line.strip()]
    plan = json.loads(read_verified(PLAN, acquisition["source_plan_sha256"]))
    expected_counts = {(split, label): n for split, n in
                       (("adapt_train", 20), ("adapt_validation", 10)) for label in plan["labels"]}
    if (Counter((r["proposed_partition"], r["reference_answer"]) for r in records) != expected_counts
            or len({r["sample_id"] for r in records}) != len(records)):
        raise ValueError("Unexpected records in the fixed adaptation split")

    source = plan["sources"]["coco_train"]
    coco_raw = zlib.decompress(read_verified(ROOT / source["cache"], source["compressed_sha256"]), -15)
    if sha256(coco_raw) != source["uncompressed_sha256"]:
        raise ValueError("Decoded COCO cache differs from the fixed annotation")
    coco = json.loads(coco_raw)
    boxes = defaultdict(list)
    wanted = {r["coco_image_id"] for r in records}
    for annotation in coco["annotations"]:
        if annotation["image_id"] in wanted:
            boxes[annotation["image_id"]].append(annotation["bbox"])

    pixels, thumbnails = {}, {}
    for r in records:
        asset = r["asset"]
        with Image.open(io.BytesIO(read_verified(ROOT / asset["file"], asset["sha256"]))) as image:
            if image.size != (r["width"], r["height"]) or len(boxes[r["coco_image_id"]]) != r["annotation_count"]:
                raise ValueError(f"Image or box metadata differs for {r['sample_id']}")
            gray = image.convert("L")
            pixels[r["sample_id"]] = np.array(gray)
            thumbnails[r["sample_id"]] = np.array(gray.resize((THUMBNAIL_SIDE, THUMBNAIL_SIDE), Image.Resampling.BOX))

    train = [r for r in records if r["proposed_partition"] == "adapt_train"]
    validation = [r for r in records if r["proposed_partition"] == "adapt_validation"]
    left = np.stack([normalized_pixels(thumbnails[r["sample_id"]]) for r in train])
    scores = np.stack([
        left @ np.stack([normalized_pixels(orient(thumbnails[r["sample_id"]], *orientation))
                         for r in validation]).T for orientation in ORIENTATIONS
    ])
    candidates = []
    excluded = set()
    for i, j in zip(*np.where(scores.max(axis=0) >= NCC_THRESHOLD)):
        a, b = train[i], validation[j]
        a_id, b_id = a["sample_id"], b["sample_id"]
        thumbnail_index = int(scores[:, i, j].argmax())
        candidate = {
            "train_id": a_id, "train_image": a["image_name"], "train_label": a["reference_answer"],
            "validation_id": b_id, "validation_image": b["image_name"],
            "validation_label": b["reference_answer"],
            "thumbnail_ncc": round(float(scores[thumbnail_index, i, j]), 6),
            "thumbnail_orientation": orientation_info(*ORIENTATIONS[thumbnail_index]),
            "status": "conservative_review",
        }
        x, y = pixels[a_id], pixels[b_id]
        left_native = normalized_pixels(x)
        native_scores = []
        for rotation, reflection in ORIENTATIONS:
            oriented = orient(y, rotation, reflection)
            if oriented.shape == x.shape:
                native_scores.append((float(left_native @ normalized_pixels(oriented)), rotation, reflection))
        if native_scores:
            score, rotation, reflection = max(native_scores)
            box_error = compare_boxes(boxes[a["coco_image_id"]], boxes[b["coco_image_id"]],
                                      y.shape, rotation, reflection)
            candidate.update(native_ncc=round(score, 6), native_orientation=orientation_info(rotation, reflection),
                             train_boxes=len(boxes[a["coco_image_id"]]), validation_boxes=len(boxes[b["coco_image_id"]]),
                             max_matched_box_corner_error_px=box_error)
            if (score >= NCC_THRESHOLD and box_error is not None and box_error <= BOX_TOLERANCE
                    and a["reference_answer"] == b["reference_answer"]):
                candidate["status"] = "high_confidence_geometric_derivative"
        candidates.append(candidate)
        excluded.add(b_id)

    return {
        "status": "completed",
        "scope": "D4 rotation/reflection screening of the frozen adaptation split. Original scene lineage is not established.",
        "source_records_path": records_path.relative_to(ROOT).as_posix(),
        "source_records_sha256": sha256(raw_records),
        "source_coco_sha256": sha256(coco_raw),
        "thumbnail_side_px": THUMBNAIL_SIDE,
        "ncc_threshold": NCC_THRESHOLD,
        "box_corner_tolerance_px": BOX_TOLERANCE,
        "train_images": len(train), "validation_images": len(validation),
        "cross_partition_image_pairs": len(train) * len(validation),
        "orientation_comparisons": int(scores.size),
        "candidates": candidates,
        "excluded_validation_ids": [r["sample_id"] for r in validation if r["sample_id"] in excluded],
        "exclusion_policy": "All thumbnail candidates are excluded from the sensitivity subset. The original split is retained.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New report path. Defaults to standard output.")
    args = parser.parse_args()
    text = json.dumps(audit(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        print(f"completed: {args.output}")


if __name__ == "__main__":
    main()
