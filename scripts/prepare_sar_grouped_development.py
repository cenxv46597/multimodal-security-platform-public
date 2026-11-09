"""Build a development split from observed source and geometric groups."""

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import PurePosixPath
import re
import struct
import zlib

import numpy as np
from PIL import Image

from audit_sar_adaptation_near_duplicates import (
    ORIENTATIONS, THUMBNAIL_SIDE, compare_boxes, normalized_pixels, orient, orientation_info,
)
from plan_sar_adaptation_data import read_verified
from run_sar_projector_pilot import MANIFESTS, PLAN, PROTOCOL, RECORDS, verified_partitions
from run_vlm_clean_screen import RECORDS as HISTORICAL_RECORDS, ROOT, load_json, read_records, sha256

MAPPING = ROOT / ".work/sar-grouped-development-20261107/mapping.json"
MAPPING_SHA256 = "e0d814f6661657679ac355f52728df94cb0eb02d00756b7a0d97ce50aa787f55"
OUTPUT = MANIFESTS / "sar_grouped_development_plan_20261107.json"
GROUPED_RECORDS = ROOT / "data/processed/sar_grouped_development_v1/records.jsonl"
CROP_THRESHOLDS = ((100, 0.80), (90, 0.92), (80, 0.92))
SOURCE_SUFFIX = re.compile(r"__[0-9]+(?:\.[0-9]+)?__-?[0-9]+___-?[0-9]+$")


def source_parent(original_name):
    stem = PurePosixPath(original_name.replace("\\", "/")).with_suffix("").as_posix()
    return SOURCE_SUFFIX.sub("", stem)


def source_rows():
    train, validation, labels = verified_partitions(load_json(PROTOCOL))
    historical, task, _ = read_records()
    if labels != task["labels"]:
        raise ValueError("Historical and adaptation categories differ")
    original_names = json.loads(read_verified(MAPPING, MAPPING_SHA256))
    reverse = {filename: source for source, filename in original_names.items()}
    if len(reverse) != len(original_names):
        raise ValueError("Published source mapping is not one-to-one")
    source = load_json(PLAN)["sources"]["coco_train"]
    coco = json.loads(zlib.decompress(read_verified(ROOT / source["cache"], source["compressed_sha256"]), -15))
    images = {entry["id"]: entry for entry in coco["images"]}
    categories = {entry["id"]: entry["name"] for entry in coco["categories"]}
    boxes = defaultdict(list)
    selected_ids = {
        row.get("source_coco", row)["coco_image_id"] for row in [*train, *validation, *historical]
    }
    for box in coco["annotations"]:
        if box["image_id"] in selected_ids:
            boxes[box["image_id"]].append(box)
    rows = []
    for part, records in (("adapt_train", train), ("adapt_validation", validation), ("historical", historical)):
        for record in records:
            meta = record.get("source_coco", record)
            image = images[meta["coco_image_id"]]
            entries = boxes[image["id"]]
            width, height = image["width"], image["height"]
            if (image["file_name"] != record["image_name"] or record["asset"]["size"] != [width, height]
                    or len(entries) != meta["annotation_count"]
                    or {categories[box["category_id"]] for box in entries} != {record["reference_answer"]}):
                raise ValueError(f"Source annotation mismatch: {record['sample_id']}")
            max_fraction = max(
                max(0, min(width, x + w) - max(0, x)) * max(0, min(height, y + h) - max(0, y)) / (width * height)
                for x, y, w, h in (entry["bbox"] for entry in entries)
            )
            name = reverse[record["image_name"]]
            rows.append({
                "record": record, "role": part, "source_name": name, "source_parent": source_parent(name),
                "box_fraction": max_fraction, "scale_bin": "small" if max_fraction < .05 else
                "medium" if max_fraction < .20 else "large",
                "boxes": [entry["bbox"] for entry in entries],
            })
    if len({row["record"]["sample_id"] for row in rows}) != 240:
        raise ValueError("Expected 240 distinct local development records")
    return rows, labels, len(original_names)


def geometric_links(rows):
    arrays, crops = [], {fraction: [] for fraction, _ in CROP_THRESHOLDS}
    for row in rows:
        asset = row["record"]["asset"]
        path = (ROOT / asset["file"]).resolve()
        if not path.is_relative_to(ROOT / "data/raw"):
            raise ValueError("Image is outside the verified raw data directory")
        with Image.open(io.BytesIO(read_verified(path, asset["sha256"]))) as image:
            if list(image.size) != asset["size"]:
                raise ValueError("Image dimensions differ from verified metadata")
            rgb = image.convert("RGB")
            row["pixel_sha256"] = hashlib.sha256(struct.pack(">QQ", *image.size) + rgb.tobytes()).hexdigest()
            gray = image.convert("L")
            arrays.append(np.asarray(gray))
            for fraction in crops:
                width, height = gray.size
                crop_width, crop_height = max(1, width * fraction // 100), max(1, height * fraction // 100)
                x, y = (width - crop_width) // 2, (height - crop_height) // 2
                view = gray.crop((x, y, x + crop_width, y + crop_height))
                crops[fraction].append(np.asarray(view.resize((THUMBNAIL_SIDE, THUMBNAIL_SIDE), Image.Resampling.BOX)))
    left = np.stack([normalized_pixels(image) for image in crops[100]])
    hits = {}
    for fraction, threshold in CROP_THRESHOLDS:
        for rotation, flip in ORIENTATIONS:
            right = np.stack([normalized_pixels(orient(image, rotation, flip)) for image in crops[fraction]])
            scores = left @ right.T
            for i, j in zip(*np.where(scores >= threshold)):
                i, j = int(i), int(j)
                if i == j:
                    continue
                key = tuple(sorted((i, j)))
                candidate = (fraction == 100, round(float(scores[i, j]), 6))
                if key not in hits or candidate > hits[key][0]:
                    hits[key] = (candidate, i, j, fraction, rotation, flip)
    links = []
    for _, ((_, score), i, j, fraction, rotation, flip) in sorted(hits.items()):
        edge = {"left": i, "right": j, "reason": "geometric_candidate", "thumbnail_ncc": score,
                "right_center_crop_percent": fraction, "right_orientation": orientation_info(rotation, flip)}
        if fraction == 100:
            oriented = orient(arrays[j], rotation, flip)
            if oriented.shape == arrays[i].shape:
                edge["native_ncc"] = round(float(normalized_pixels(arrays[i]) @ normalized_pixels(oriented)), 6)
                edge["matched_box_corner_error_px"] = compare_boxes(
                    rows[i]["boxes"], rows[j]["boxes"], arrays[j].shape, rotation, flip)
        links.append(edge)
    return links


def assign_groups(rows, links):
    for field in ("source_parent", "pixel_sha256"):
        seen = {}
        for index, row in enumerate(rows):
            value = row[field]
            if value in seen:
                links.append({"left": seen[value], "right": index, "reason": field})
            else:
                seen[value] = index
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for link in links:
        parents[find(link["right"])] = find(link["left"])
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    for members in groups.values():
        key = "\n".join(sorted(row["record"]["sample_id"] for row in members))
        group = "sar-observed-" + hashlib.sha256(key.encode()).hexdigest()[:16]
        for row in members:
            row["group"] = group
    return list(groups.values())


def select_validation(rows):
    occupied = {row["group"] for row in rows if row["role"] == "adapt_train"}
    kept, removed = [], []
    for row in rows:
        if row["role"] != "adapt_validation":
            continue
        if row["group"] in occupied:
            removed.append(row)
        else:
            kept.append(row)
            occupied.add(row["group"])
    available = [row for row in rows if row["role"] == "historical" and row["group"] not in occupied]
    options = [
        sorted((candidate for candidate in available
                if candidate["record"]["reference_answer"] == slot["record"]["reference_answer"]
                and candidate["scale_bin"] == slot["scale_bin"]),
               key=lambda candidate: (abs(candidate["box_fraction"] - slot["box_fraction"]),
                                      candidate["record"]["sample_id"]))
        for slot in removed
    ]
    by_group, replacements = {}, {}

    def place(slot, visited):
        for candidate in options[slot]:
            group = candidate["group"]
            if group in visited:
                continue
            visited.add(group)
            if group not in by_group or place(by_group[group], visited):
                by_group[group] = slot
                replacements[slot] = candidate
                return True
        return False

    for slot in range(len(removed)):
        place(slot, set())
    return kept, removed, replacements


def describe(row):
    return {
        "sample_id": row["record"]["sample_id"], "image_name": row["record"]["image_name"],
        "label": row["record"]["reference_answer"], "role": row["role"], "source_name": row["source_name"],
        "source_parent": row["source_parent"], "observed_source_group": row["group"],
        "max_visible_box_fraction": row["box_fraction"], "scale_bin": row["scale_bin"],
    }


def save_fixed(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError(f"Existing frozen output differs: {path}") from None


def main():
    rows, labels, mapping_count = source_rows()
    links = geometric_links(rows)
    groups = assign_groups(rows, links)
    kept, removed, replacements = select_validation(rows)
    train = [row for row in rows if row["role"] == "adapt_train"]
    validation = kept + [replacements[i] for i in sorted(replacements)]
    validation_groups = {row["group"] for row in validation}
    if len(validation_groups) != len(validation) or validation_groups & {row["group"] for row in train}:
        raise ValueError("Observed group separation failed")
    complete = Counter(row["record"]["reference_answer"] for row in validation) == dict.fromkeys(labels, 10)
    encoded = "".join(
        json.dumps({**row["record"], "grouped_partition": partition, **{
            key: value for key, value in describe(row).items()
            if key in ("source_name", "source_parent", "observed_source_group", "max_visible_box_fraction", "scale_bin")
        }}, ensure_ascii=False) + "\n"
        for partition, members in (("train", train), ("validation", validation)) for row in members
    ).encode("utf-8")
    coverage = {
        name: dict(Counter(row["scale_bin"] for row in members if row["record"]["reference_answer"] == "harbor"))
        for name, members in (("training", train), ("original_validation", [*kept, *removed]),
                              ("grouped_validation", validation),
                              ("historical_pool", [row for row in rows if row["role"] == "historical"]))
    }
    manifest = {
        "status": "completed" if complete else "needs_additional_groups", "labels": labels,
        "scope": "Development split with observed source and geometric groups. Geographic scene lineage is unresolved.",
        "mapping": {"source_member": "SARDet_100K/mapping.json", "dataset": "greatbird/sardet-100k",
                    "dataset_version": 1, "storage_version_id": 11694364, "sha256": MAPPING_SHA256,
                    "entries": mapping_count, "cache": MAPPING.relative_to(ROOT).as_posix(),
                    "group_rule": "Shared mapped source name after removing the trailing crop suffix"},
        "original_records_sha256": sha256(RECORDS),
        "historical_records_sha256": sha256(HISTORICAL_RECORDS),
        "audit": {"images": len(rows), "groups": len(groups), "multi_image_groups": sum(len(g) > 1 for g in groups),
                  "cross_role_groups": sum(len({row["role"] for row in group}) > 1 for group in groups),
                  "thumbnail_side_px": THUMBNAIL_SIDE, "crop_and_ncc_thresholds": CROP_THRESHOLDS,
                  "links": [{**link, "left": rows[link["left"]]["record"]["sample_id"],
                             "right": rows[link["right"]]["record"]["sample_id"]} for link in links]},
        "selection": {"rule": "Keep safe original development rows first. Match each removed row to a historical row of the same label and size bin, using each observed group once.",
                      "original_retained": len(kept), "historical_replacements": len(replacements),
                      "removed": [describe(row) for row in removed],
                      "replacement_pairs": [{"removed": describe(removed[i]), "replacement": describe(replacements[i])}
                                            for i in sorted(replacements)],
                      "unfilled": [describe(row) for i, row in enumerate(removed) if i not in replacements]},
        "harbor_coverage": coverage,
        "pool": [describe(row) for row in rows],
        "validation_ids": [row["record"]["sample_id"] for row in validation],
        "records": {"path": GROUPED_RECORDS.relative_to(ROOT).as_posix(), "count": len(train) + len(validation),
                    "sha256": hashlib.sha256(encoded).hexdigest()},
    }
    save_fixed(GROUPED_RECORDS, encoded)
    save_fixed(OUTPUT, (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps({"status": manifest["status"], "groups": len(groups), "links": len(links),
                      "train": len(train), "validation": len(validation), "removed": len(removed),
                      "replacements": len(replacements), "harbor_coverage": coverage,
                      "unfilled": manifest["selection"]["unfilled"]}, ensure_ascii=False))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
