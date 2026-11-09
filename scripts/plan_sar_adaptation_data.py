"""Propose a fixed SAR adaptation split from existing local metadata."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "resources/manifests"
CACHE = ROOT / ".work/next-clean-screen-20261105/images"
TASK = ROOT / "experiments/protocols/sar_single_category_v0.json"
OUTPUT = MANIFESTS / "sar_adaptation_data_plan_20261107.json"
SEED = "sar-adaptation-data-v0"
TRAIN_MEMBER = "Text/VQA/train/SARVQA1_train.csv"
ZIP_TRAIN = PurePosixPath("SARDet_100K/JPEGImages/train")
ZIP_INDEX = CACHE / "kaggle-zip-index.json"
ZIP_INDEX_SHA256 = "a2a225097c6481d87035fa2b29f9a5732ed460abccbabdc6a1cca7739e32cb83"
DOWNLOAD_ROOT = "https://www.kaggle.com/api/v1/datasets/download/greatbird/sardet-100k"
PARTITIONS = {"adapt_train": 20, "adapt_validation": 10}
ATTEMPTS_PER_IMAGE = 2


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_verified(path, expected_sha256):
    data = path.read_bytes()
    if sha256(data) != expected_sha256:
        raise ValueError(f"Source differs from fixed evidence: {path.relative_to(ROOT)}")
    return data


def zip_fingerprint(entry):
    return entry["crc32"], entry["uncompressed_bytes"]


def load_image_metadata(evidence):
    """Join the pinned ZIP directory and cached COCO train annotation."""
    index = json.loads(read_verified(ZIP_INDEX, ZIP_INDEX_SHA256))
    if (index["dataset"], index["dataset_version"], index["storage_version_id"]) != (
        "greatbird/sardet-100k", 1, 11694364
    ):
        raise ValueError("Unexpected dataset version in the ZIP index")
    entries = {}
    for entry in index["entries"]:
        member = PurePosixPath(entry["name"])
        if member.parent == ZIP_TRAIN and entry["uncompressed_bytes"] > 0:
            if member.name in entries:
                raise ValueError(f"Duplicate train ZIP member: {member.name}")
            entries[member.name] = entry

    source = evidence["official_coco_validation"]
    compressed_path = CACHE / "train-json-deflate.bin"
    compressed = read_verified(compressed_path, source["compressed_sha256"])
    decoder = zlib.decompressobj(-15)
    raw = decoder.decompress(compressed, source["uncompressed_bytes"] + 1)
    if (not decoder.eof or decoder.unused_data or len(raw) != source["uncompressed_bytes"]
            or sha256(raw) != source["uncompressed_sha256"]
            or f"{zlib.crc32(raw) & 0xffffffff:08x}" != source["crc32"]):
        raise ValueError("COCO train metadata failed its length, SHA or CRC check")
    coco = json.loads(raw)
    images = {row["file_name"]: row for row in coco["images"]}
    ids = {row["id"] for row in coco["images"]}
    if len(images) != len(ids) or len(images) != source["total_images"]:
        raise ValueError("COCO train image names or IDs are not unique")
    categories = {row["id"]: row["name"] for row in coco["categories"]}
    annotations = defaultdict(Counter)
    for row in coco["annotations"]:
        annotations[row["image_id"]][row["category_id"]] += 1
    return entries, images, categories, annotations


def build_exclusions(evidence, entries, examples_path):
    """Keep the historical development screen and pilot examples separate."""
    records_meta = evidence["records"]
    records = [
        json.loads(line)
        for line in read_verified(ROOT / records_meta["path"], records_meta["sha256"]).splitlines()
        if line.strip()
    ]
    old_entries = evidence["official_file_map"]["selected"]
    old_names = {record["image_name"] for record in records}
    if (len(records) != 60 or len(old_entries) != 60 or len(old_names) != 60
            or old_names != {entry["image_name"] for entry in old_entries}):
        raise ValueError("Historical development sample identities differ")
    fingerprints = set()
    stems = {PurePosixPath(name).stem.casefold() for name in old_names}
    for old in old_entries:
        entry = entries[old["image_name"]]
        if entry["name"] != old["name"] or zip_fingerprint(entry) != zip_fingerprint(old):
            raise ValueError(f"Historical ZIP member differs: {old['image_name']}")
        fingerprints.add(zip_fingerprint(entry))
    with zipfile.ZipFile(examples_path) as archive:
        example_images = [
            info for info in archive.infolist()
            if PurePosixPath(info.filename).parent == PurePosixPath("Examples/images")
            and not info.is_dir()
        ]
        if len(example_images) != 20:
            raise ValueError("Expected 20 images in the fixed Examples.zip")
        for info in example_images:
            stems.add(PurePosixPath(info.filename).stem.casefold())
            fingerprints.add((f"{info.CRC:08x}", info.file_size))
    return stems, fingerprints


def sample_plan(task, text_path, metadata, exclusions):
    entries, images, categories, annotations = metadata
    excluded_stems, excluded_fingerprints = exclusions
    labels = task["labels"]
    rule = task["source_selection"]
    pools = defaultdict(list)
    seen_names = set()
    excluded = Counter()
    with zipfile.ZipFile(text_path) as archive, archive.open(TRAIN_MEMBER) as stream:
        reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
        for index, row in enumerate(reader, 1):
            application, question, label, source_name = (
                row[key].strip() for key in ("Application", "Question", "Answer", "Image Name")
            )
            if (application != rule["Application"] or label not in labels
                    or question != rule["Question_after_outer_whitespace_trim"]):
                continue
            name = PurePosixPath(source_name.replace("\\", "/")).name
            if name in seen_names:
                raise ValueError(f"Duplicate eligible SARVQA1 train name: {name}")
            seen_names.add(name)
            if PurePosixPath(name).stem.casefold() in excluded_stems:
                excluded["old_development_or_example_stem"] += 1
                continue
            image, entry = images.get(name), entries.get(name)
            if not image or not entry:
                raise ValueError(f"Missing exact train COCO/ZIP match: {name}")
            counts = annotations[image["id"]]
            category_ids = sorted(counts)
            if ([categories[key] for key in category_ids] != [label]
                    or min(image["width"], image["height"]) <= 0):
                raise ValueError(f"Invalid COCO dimensions or category for {name}")
            if zip_fingerprint(entry) in excluded_fingerprints:
                excluded["old_development_or_example_crc_and_length"] += 1
                continue
            digest = sha256("\n".join((SEED, task["revision"], name)).encode("utf-8"))
            pools[label].append({
                "sample_id": "sar-adapt-" + digest[:20], "selection_sha256": digest,
                "image_name": name, "reference_answer": label, "source_member": TRAIN_MEMBER,
                "source_record_index_1_based": index, "question": task["model_question"],
                "author_split": "train", "sarvqa_author_split": "train", "formal_split": None,
                "kaggle_image_directory": "train", "coco_image_id": image["id"],
                "width": image["width"], "height": image["height"],
                "annotation_count": sum(counts.values()), "category_ids": category_ids,
                "category_names": [label], **entry,
                "download_url": f"{DOWNLOAD_ROOT}/{quote(entry['name'], safe='')}?datasetVersionNumber=1",
            })
    for label in labels:
        pools[label].sort(key=lambda row: (row["selection_sha256"], row["image_name"]))

    selected = []
    fingerprints = set()
    stems = set()
    for partition, per_class in PARTITIONS.items():
        for label in labels:
            count = 0
            for row in pools[label]:
                fingerprint = zip_fingerprint(row)
                stem = PurePosixPath(row["image_name"]).stem.casefold()
                if fingerprint in fingerprints or stem in stems:
                    continue
                selected.append({**row, "proposed_partition": partition})
                fingerprints.add(fingerprint)
                stems.add(stem)
                count += 1
                if count == per_class:
                    break
            if count != per_class:
                raise ValueError(f"Insufficient unique images for {partition}/{label}")
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError("Selection IDs are not unique")
    return selected, dict(excluded), {label: len(pools[label]) for label in labels}


def main():
    acquisition_path = MANIFESTS / "sar_clean_development_acquisition_20261105.json"
    acquisition_bytes = acquisition_path.read_bytes()
    evidence = json.loads(acquisition_bytes)
    if evidence["status"] != "passed" or evidence["official_coco_validation"]["status"] != "passed":
        raise ValueError("Historical clean development source verification has not passed")
    old_plan = json.loads(read_verified(ROOT / evidence["source_plan"], evidence["source_plan_sha256"]))
    task = json.loads(read_verified(TASK, old_plan["task_protocol_sha256"]))

    inventory_path = MANIFESTS / "sarlang_pilot_inventory_20261105.json"
    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_bytes)
    if inventory["revision"] != task["revision"] or inventory["status"] != "passed":
        raise ValueError("SARLANG source inventory differs from the task revision")
    archives = {}
    for name in ("Text.zip", "Examples.zip"):
        entry = next(row for row in inventory["files"] if PurePosixPath(row["path"]).name == name)
        path = ROOT / entry["path"]
        read_verified(path, entry["sha256"])
        archives[name] = path, entry["sha256"]
    text_path, text_sha = archives["Text.zip"]
    examples_path, examples_sha = archives["Examples.zip"]
    metadata = load_image_metadata(evidence)
    exclusions = build_exclusions(evidence, metadata[0], examples_path)
    selected, excluded, pool_counts = sample_plan(task, text_path, metadata, exclusions)

    budgets = {}
    for partition, per_class in PARTITIONS.items():
        rows = [row for row in selected if row["proposed_partition"] == partition]
        budgets[partition] = {
            "image_count": len(rows), "images_per_class": per_class,
            "zip_compressed_member_bytes": sum(row["compressed_bytes"] for row in rows),
            "per_file_range_response_body_bytes": sum(row["uncompressed_bytes"] for row in rows),
        }
    response_bytes = sum(row["uncompressed_bytes"] for row in selected)
    manifest = {
        "plan_id": SEED, "schema_version": 1, "date": "2026-11-07", "status": "proposal",
        "task_id": task["task_id"], "labels": task["labels"], "model_question": task["model_question"],
        "sources": {
            "task_protocol": {"path": TASK.relative_to(ROOT).as_posix(), "sha256": old_plan["task_protocol_sha256"]},
            "sarlang_inventory": {"path": inventory_path.relative_to(ROOT).as_posix(), "sha256": sha256(inventory_bytes)},
            "sarlang_revision": task["revision"], "source_member": TRAIN_MEMBER,
            "text_zip": {"path": text_path.relative_to(ROOT).as_posix(), "sha256": text_sha},
            "examples_zip": {"path": examples_path.relative_to(ROOT).as_posix(), "sha256": examples_sha},
            "kaggle": {"dataset": "greatbird/sardet-100k", "dataset_version": 1,
                       "storage_dataset_id": 7339941, "storage_version_id": 11694364},
            "zip_index": {"path": ZIP_INDEX.relative_to(ROOT).as_posix(), "sha256": ZIP_INDEX_SHA256},
            "coco_train": {"member": "SARDet_100K/Annotations/train.json",
                           "cache": (CACHE / "train-json-deflate.bin").relative_to(ROOT).as_posix(),
                           "compressed_sha256": evidence["official_coco_validation"]["compressed_sha256"],
                           "uncompressed_sha256": evidence["official_coco_validation"]["uncompressed_sha256"],
                           "crc32": evidence["official_coco_validation"]["crc32"]},
            "historical_development_acquisition": {"path": acquisition_path.relative_to(ROOT).as_posix(),
                                                   "sha256": sha256(acquisition_bytes)},
            "historical_development_records": evidence["records"],
        },
        "selection": {
            "ranking": "SHA256 of UTF-8 seed + LF + SARLANG revision + LF + exact image name",
            "seed": SEED,
            "order": "Take train before development validation, in the fixed task label order, using ascending rank then image name.",
            "eligibility": "Fixed SARVQA1 author train rule, exact COCO and ZIP train name, positive COCO dimensions, and one COCO category matching the reference.",
            "exclusions": "Case-folded stems and ZIP CRC32 plus uncompressed length from the historical 60 and 20 examples.",
            "cross_partition_deduplication": "Globally unique case-folded stem and ZIP CRC32 plus uncompressed length.",
            "excluded_counts": excluded, "eligible_pool_counts": pool_counts,
            "model_outputs_used_for_selection": False,
        },
        "image_count": len(selected), "partition_budget": budgets,
        "download_budget": {
            "zip_compressed_member_bytes": sum(row["compressed_bytes"] for row in selected),
            "per_file_range_response_body_bytes_one_attempt_each": response_bytes,
            "max_attempts_per_image": ATTEMPTS_PER_IMAGE,
            "response_body_byte_limit": response_bytes * ATTEMPTS_PER_IMAGE,
            "metadata_download_bytes": 0,
            "method": "Individual pinned Kaggle file Range responses return identity-encoded image bytes. ZIP-compressed sizes are recorded separately.",
        },
        "validation_scope": "Development comparison at image level. Original scene and constituent source dataset lineage remain unresolved.",
        "checks_before_use": [
            "Confirm pinned storage version and Content-Range, then verify image byte length and ZIP CRC32.",
            "Fully decode each image and compare dimensions with COCO.",
            "Compare file SHA256 and canonical decoded RGB pixel hashes including dimensions across partitions, the historical 60, and Examples.",
            "Stop acquisition on source mismatch, budget breach, failed integrity or cross-partition duplicate. Retain fixed counts and report failures.",
        ],
        "acquired": False, "trained": False, "formal_test_created": False, "selected": selected,
    }
    rendered = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with OUTPUT.open("xb") as stream:
            stream.write(rendered)
    except FileExistsError:
        if OUTPUT.read_bytes() != rendered:
            raise ValueError(f"Existing proposal differs. Refusing to overwrite {OUTPUT.name}") from None
    print(json.dumps({"plan": OUTPUT.relative_to(ROOT).as_posix(), "sha256": sha256(rendered),
                      "image_count": len(selected), "response_body_bytes": response_bytes,
                      "response_body_byte_limit": response_bytes * ATTEMPTS_PER_IMAGE}))


if __name__ == "__main__":
    main()
