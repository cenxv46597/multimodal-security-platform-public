"""Read-only source-group audit. Does not create an experimental split."""

import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATTERN = re.compile(
    r"(?P<label>[^_]+)_real_(?P<collection>.*?)_elevDeg_(?P<elevation>\d+)_"
    r"azCenter_(?P<az_int>\d+)_(?P<az_fraction>\d+)_serial_(?P<serial>.*?)\.png"
)


def audit_sample(entries):
    by_class = defaultdict(list)
    hashes = defaultdict(list)
    for entry in entries:
        path = ROOT / entry["path"]
        match = SAMPLE_PATTERN.fullmatch(path.name)
        if not match or match["label"] != path.parent.name:
            raise ValueError(f"Unexpected SAMPLE filename: {path.name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"SAMPLE content changed: {path.name}")
        hashes[digest].append(entry["path"])
        row = match.groupdict()
        row["azimuth"] = float(row["az_int"] + "." + row["az_fraction"])
        by_class[row["label"]].append(row)
    counts = {label: len(rows) for label, rows in sorted(by_class.items())}
    classes = {}
    for label, rows in sorted(by_class.items()):
        classes[label] = {
            "files": len(rows), "serial_tokens": sorted({row["serial"] for row in rows}),
            "collection_tokens": sorted({row["collection"] for row in rows}),
            "elevations_degrees_from_filename": sorted({int(row["elevation"]) for row in rows}),
            "azimuth_degrees_from_filename": [min(row["azimuth"] for row in rows),
                                               max(row["azimuth"] for row in rows)],
            "ten_degree_bin_counts_for_audit_only": dict(sorted(Counter(
                int(row["azimuth"] // 10) * 10 for row in rows).items())),
        }
    return {
        "files": sum(counts.values()), "classes": classes,
        "unique_content_hashes": len(hashes),
        "duplicate_content_groups": [paths for paths in hashes.values() if len(paths) > 1],
        "class_count": len(classes), "uniform_guess_reference": 1 / len(classes),
        "majority_class_reference": max(counts.values()) / sum(counts.values()),
        "each_class_has_one_serial_token": all(len(v["serial_tokens"]) == 1 for v in classes.values()),
        "limits": [
            "Filename tokens are observed metadata, not a verified physical-instance inventory.",
            "No per-class unseen-serial train/test evaluation is possible with this selected batch.",
            "Nearby angles and repeated elevations require grouped/blocked design; bins above are an audit only.",
            "Ten fine classes are not silently mapped to SARChat six coarse categories.",
        ],
    }


def audit_hrrp(entries):
    metadata_path = ROOT / "data/raw/hrrp_pc/a1ab400c91f3/metadata.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    index = {row["hrrp_filename"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Nonunique HRRP metadata filenames")
    group_splits = defaultdict(set)
    split_angles = defaultdict(set)
    for row in rows:
        group_splits[(row["label_str"], int(row["theta"]))].add(row["split"])
        split_angles[row["split"]].add(int(row["theta"]))
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("Full metadata original group crosses split")
    local_groups = defaultdict(list)
    local_hashes = defaultdict(list)
    for entry in entries:
        path = ROOT / entry["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"HRRP content changed: {path.name}")
        row = index[path.name]
        local_groups[(row["label_str"], int(row["theta"]))].append(row)
        local_hashes[digest].append(row)
    return {
        "full_metadata_rows": len(rows), "full_metadata_original_groups": len(group_splits),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "existing_metadata_splits": {
            split: {"angle_min": min(angles), "angle_max": max(angles), "angle_count": len(angles),
                    "rows": sum(row["split"] == split for row in rows),
                    "original_groups": sum(next(iter(splits)) == split for splits in group_splits.values())}
            for split, angles in split_angles.items()
        },
        "local_files": len(entries), "local_original_groups": len(local_groups),
        "local_unique_content_hashes": len(local_hashes),
        "local_groups": [{"label": label, "theta": angle, "files": len(group),
                          "split": group[0]["split"]} for (label, angle), group in sorted(local_groups.items())],
        "local_original_aug0_files": sum(row["augmentation_type"] == "0"
                                         for group in local_groups.values() for row in group),
        "local_same_content_cross_split": any(len({r["split"] for r in group}) > 1
                                              for group in local_hashes.values()),
        "uniform_guess_reference": 1 / 3,
        "limits": [
            "Local files are a format subset: one original angle per class per split, not a formal benchmark.",
            "Original group means label/theta in this simulation, not an independent physical target.",
            "Existing metadata has adjacent split boundaries 100/101 and 140/141 degrees, without an angle guard band.",
            "No new split, model input, preprocessing, or train/test reassignment was created.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    entries = [json.loads(line) for line in
               (ROOT / "resources/manifests/resource_inventory.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    for entry in entries:
        if entry["resource"] == "HRRP_PC_metadata":
            if hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"HRRP metadata changed: {entry['path']}")
    report = {
        "checked_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
        "scope": "Read-only observed groups; no experimental split or model evaluation created",
        "sample": audit_sample([e for e in entries if e["resource"] == "SAMPLE_qpm_real" and e["path"].endswith(".png")]),
        "hrrp": audit_hrrp([e for e in entries if e["resource"] == "HRRP_PC_format_subset"]),
        "status": "passed",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = (ROOT / args.output).resolve()
        if not output.is_relative_to(ROOT / "resources/manifests") or not output.name.startswith("domain_group_audit_"):
            parser.error("Output must be resources/manifests/domain_group_audit_*.json")
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
