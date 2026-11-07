"""Audit a fixed single-category annotation task and prepare local link checks.

No inference, training, attack, formal split, or automatic stem matching.
"""

import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/protocols/sar_single_category_v0.json"
OUTPUT = ROOT / "data/processed/sar_single_category_v0"


def basename(path):
    return PurePosixPath(path.replace("\\", "/")).name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-processor", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    labels = set(config["labels"])
    source_question = config["source_selection"]["Question_after_outer_whitespace_trim"]
    inventory = json.loads((ROOT / "resources/manifests/sarlang_pilot_inventory_20261105.json").read_text(encoding="utf-8"))
    for row in inventory["files"]:
        data = (ROOT / row["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"] or len(data) != row["bytes"]:
            raise ValueError(f"Original resource changed: {row['path']}")
    development_report = json.loads((ROOT / "resources/manifests/sarlang_development_20261105.json").read_text(encoding="utf-8"))
    assets = development_report["assets"]
    candidates = []
    statistics = {}
    source_groups = set()
    source_group_overlap = set()
    with zipfile.ZipFile(ROOT / "data/raw/sarlang/0271408f71bc/Text.zip") as archive:
        for member in config["source_members"]:
            counts = Counter()
            questions = Counter()
            accepted = Counter()
            split_names = set()
            with archive.open(member) as stream:
                reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
                for number, raw in enumerate(reader, 1):
                    row = {key.strip(): value.strip() for key, value in raw.items() if key is not None}
                    if row["Application"] != config["source_selection"]["Application"]:
                        continue
                    counts["classification_rows"] += 1
                    questions[row["Question"]] += 1
                    if row["Question"] != source_question:
                        counts["excluded_question_template"] += 1
                        continue
                    if row["Answer"] not in labels:
                        counts["excluded_nonexact_single_label"] += 1
                        continue
                    counts["eligible_annotation_rows"] += 1
                    accepted[row["Answer"]] += 1
                    image_name = basename(row["Image Name"])
                    if image_name in split_names:
                        raise ValueError(f"Multiple selected annotation rows for {image_name}")
                    split_names.add(image_name)
                    if image_name in assets:
                        asset = assets[image_name]
                        candidates.append({
                            "sample_id": "sar-single-category-local-" + asset["sha256"][:16],
                            "task_id": config["task_id"],
                            "image_group_id": "image-bytes-sha256:" + asset["sha256"],
                            "group_scope": "Exact example image file bytes; original scene independence unresolved",
                            "asset": asset, "question": config["model_question"],
                            "reference_answer": row["Answer"],
                            "source_question": row["Question"],
                            "source_member": member, "source_record_index_1_based": number,
                            "source_image_name": row["Image Name"],
                            "author_split": "train" if "/train/" in member else "test",
                            "match": "Exact basename link only; no full upstream image-byte equivalence claim",
                            "formal_split": None, "usage": "development_source_link_check_only",
                        })
            source_group_overlap.update(source_groups & split_names)
            source_groups.update(split_names)
            statistics[member] = {**dict(counts), "question_templates": dict(questions),
                                  "eligible_unique_image_names": len(split_names),
                                  "eligible_label_counts": {label: accepted[label] for label in config["labels"]},
                                  "eligible_largest_class_fraction_descriptive": max(accepted.values()) / sum(accepted.values())}
    if source_group_overlap:
        raise ValueError("Selected author split image basenames overlap")
    candidates.sort(key=lambda row: row["asset"]["member"])
    if len({row["sample_id"] for row in candidates}) != len(candidates):
        raise ValueError("Duplicate local image or truncated ID collision")
    with zipfile.ZipFile(ROOT / "data/raw/sarlang/0271408f71bc/Examples.zip") as archive:
        for row in candidates:
            if hashlib.sha256(archive.read(row["asset"]["member"])).hexdigest() != row["asset"]["sha256"]:
                raise ValueError("Local image differs from development report")
    processor = {"status": "not_requested"}
    if args.check_processor:
        from prepare_sarlang_pilot import check_processor
        processor = check_processor(candidates)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "local_candidates.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates), encoding="utf-8")
    observed = Counter(row["reference_answer"] for row in candidates)
    constant_label = config["constant_label_reference"]["label"]
    train_members = [member for member in statistics if "/train/" in member]
    if len(train_members) != 1:
        raise ValueError("Expected one source training annotation file")
    training_counts = statistics[train_members[0]]["eligible_label_counts"]
    if training_counts[constant_label] != max(training_counts.values()):
        raise ValueError("Configured constant baseline is not selected by training distribution")
    report = {
        "checked_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
        "task_id": config["task_id"], "status": "passed_annotation_and_local_link_checks_only",
        "protocol_path": CONFIG.relative_to(ROOT).as_posix(),
        "protocol_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "source_statistics": statistics,
        "local_candidate_count": len(candidates), "local_label_counts": dict(observed),
        "local_missing_labels": [label for label in config["labels"] if label not in observed],
        "local_author_split_counts": dict(Counter(row["author_split"] for row in candidates)),
        "local_largest_class_fraction_descriptive": max(observed.values()) / sum(observed.values()) if observed else None,
        "constant_label_reference": {
            "label": constant_label, "selection_source": train_members[0],
            "scope": "Annotation-label arithmetic reference; not a model run",
            "source_annotation_fractions": {member: value["eligible_label_counts"][constant_label] /
                                            value["eligible_annotation_rows"] for member, value in statistics.items()},
            "local_correct_count": observed[constant_label], "local_total": len(candidates),
            "local_fraction": observed[constant_label] / len(candidates) if candidates else None,
        },
        "local_candidates": {"path": target.relative_to(ROOT).as_posix(),
                             "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
        "processor_check": processor,
        "formal_split_created": False, "normal_accuracy_evaluated": False,
        "normal_capability_pass_threshold_frozen": False,
        "limits": ["Annotation counts are not counts of locally acquired images.",
                   "Author single-category annotations do not prove a single physical instance or predominant class.",
                   "Exact basename pairing is not a full upstream image provenance audit.",
                   "Tiny local candidates mix author train/test and cannot represent a held-out six-class benchmark."]}
    path = ROOT / "resources/manifests/sar_single_category_v0_20261105.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "local_candidate_count", "local_label_counts", "local_missing_labels", "constant_label_reference")}, ensure_ascii=True))
    print(path)


if __name__ == "__main__":
    main()
