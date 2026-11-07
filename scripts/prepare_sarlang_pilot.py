"""Build development-only SARLANG records and inspect CPU processor inputs.

No train/test split, model weights, inference, training, or attack is created.
"""

import argparse
from collections import Counter, defaultdict
import copy
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "0271408f71bcd9afff69e4e7d324cae912461ebd"
BASE = ROOT / "data/raw/sarlang/0271408f71bc"
MODEL = ROOT / "models/SmolVLM-256M-Instruct/7e3e67edbbed"
DEFAULT_OUTPUT = ROOT / "data/processed/sarlang_pilot/0271408f71bc"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_CACHE"] = str(ROOT / ".cache/huggingface/hub")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def basename(path):
    return PurePosixPath(path.replace("\\", "/")).name


def normalized(text):
    return " ".join(text.split())


def question_text(text):
    if not text.startswith("<image>") or text.count("<image>") != 1:
        raise ValueError("Expected exactly one leading image marker")
    question = text[len("<image>"):].strip()
    if not question:
        raise ValueError("Empty question")
    return question


def verify_archives():
    inventory = json.loads((ROOT / "resources/manifests/sarlang_pilot_inventory_20261105.json").read_text(encoding="utf-8"))
    for entry in inventory["files"]:
        data = (ROOT / entry["path"]).read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Resource changed: {entry['path']}")
    return inventory


def decode_image(archive, member):
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(archive.read(member))) as image:
        image.load()
        array = np.asarray(image)
        if array.dtype != np.uint8 or image.mode not in ("L", "RGB"):
            raise ValueError(f"Unspecified image conversion for {member}: {image.mode}/{array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite image data")
        info = {"format": image.format, "mode": image.mode, "size": list(image.size),
                "dtype": str(array.dtype), "range": [int(array.min()), int(array.max())],
                "pixel_decoding": "passed", "model_conversion": "PIL convert RGB in memory"}
        rgb = image.convert("RGB")
    return rgb, info


def build_records(archive):
    examples = json.loads(archive.read("Examples/SARVQA_text.json").decode("utf-8-sig"))
    members = {basename(name): name for name in archive.namelist()
               if not name.endswith("/") and name.lower().endswith((".jpg", ".jpeg", ".png"))}
    if len(members) != 20:
        raise ValueError("Unexpected image count or ambiguous basenames")
    assets = {}
    for name, member in sorted(members.items()):
        digest = hashlib.sha256(archive.read(member)).hexdigest()
        image, info = decode_image(archive, member)
        image.close()
        assets[name] = {"archive": (BASE / "Examples.zip").relative_to(ROOT).as_posix(),
                        "member": member, "sha256": digest, **info}
    records = []
    for index, example in enumerate(examples, 1):
        if len(example["images"]) != 1 or [m["role"] for m in example["messages"]] != ["user", "assistant"]:
            raise ValueError("Unexpected demonstration schema")
        image_name = basename(example["images"][0])
        asset = assets[image_name]
        records.append({
            "sample_id": f"sarlang-demo-{index:03}",
            "image_group_id": "image-bytes-sha256:" + asset["sha256"],
            "group_scope": "Exact archive image bytes only; not audited original scene independence",
            "original_scene_group_id": None,
            "asset": asset,
            "question": question_text(example["messages"][0]["content"]),
            "reference_answer": example["messages"][1]["content"],
            "author_tasks": [], "author_annotation_matches": [],
            "usage": "development_format_pilot_only", "formal_split": None,
        })
    if len(records) != 30:
        raise ValueError("Unexpected demonstration record count")
    return records, assets


def attach_exact_task_matches(records):
    wanted = defaultdict(list)
    for record in records:
        wanted[(basename(record["asset"]["member"]), normalized(record["question"]),
                normalized(record["reference_answer"]))].append(record)
    with zipfile.ZipFile(BASE / "Text.zip") as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".csv")):
            with archive.open(member) as handle:
                for raw in csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")):
                    row = {key.strip(): value for key, value in raw.items() if key is not None}
                    question = row.get("Question", row.get("Prompt"))
                    answer = row.get("Answer", row.get("Caption"))
                    if question is None or answer is None:
                        raise ValueError(f"Unrecognized annotation columns in {member}")
                    key = (basename(row["Image Name"]), normalized(question), normalized(answer))
                    if key in wanted:
                        task = row.get("Application", row.get("Caption Type", "")).strip()
                        for record in wanted[key]:
                            record["author_annotation_matches"].append({
                                "archive_member": member, "application": task,
                                "match": "exact_basename_and_whitespace_normalized_question_answer",
                            })
    for record in records:
        record["author_tasks"] = sorted({m["application"] for m in record["author_annotation_matches"]})
        record["task_mapping_status"] = (
            "unresolved" if not record["author_tasks"] else
            "exact_author_task_match" if len(record["author_tasks"]) == 1 else "ambiguous_author_tasks")


def model_inputs(processor, archive, record):
    """Only image pixels and question are forwarded to the processor."""
    image, _ = decode_image(archive, record["asset"]["member"])
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": record["question"]},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    try:
        return processor(text=prompt, images=[image], return_tensors="pt")
    finally:
        image.close()


def check_processor(records):
    import torch
    from transformers import AutoProcessor

    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    processor.image_processor.size = {"longest_edge": 512}
    processor.image_processor.do_image_splitting = False
    allowed_keys = {"input_ids", "attention_mask", "pixel_values", "pixel_attention_mask"}
    checks = []
    group_pixels = {}
    with zipfile.ZipFile(BASE / "Examples.zip") as archive:
        for record in records:
            batch = model_inputs(processor, archive, record)
            if set(batch) != allowed_keys:
                raise ValueError(f"Unexpected model inputs: {set(batch)}")
            if list(batch["pixel_values"].shape) != [1, 1, 3, 512, 512]:
                raise ValueError("Unexpected image processor shape")
            if int((batch["input_ids"] == 49190).sum()) != 64:
                raise ValueError("Unexpected image token count")
            if not torch.isfinite(batch["pixel_values"]).all():
                raise ValueError("Nonfinite processor image values")
            group = record["image_group_id"]
            if group in group_pixels and not torch.equal(group_pixels[group], batch["pixel_values"]):
                raise ValueError("Same image routed to different processed pixels")
            group_pixels.setdefault(group, batch["pixel_values"].clone())
            checks.append({"sample_id": record["sample_id"], "input_keys": sorted(batch),
                           "text_and_image_token_length": batch["input_ids"].shape[1],
                           "pixel_shape": list(batch["pixel_values"].shape)})
        original = model_inputs(processor, archive, records[0])
        changed = copy.deepcopy(records[0])
        changed.update(sample_id="changed-id", image_group_id="changed-group",
                       reference_answer="CHANGED_EVALUATOR_REFERENCE", author_tasks=["changed-task"],
                       original_scene_group_id="changed-scene", formal_split="changed-bookkeeping")
        mutated = model_inputs(processor, archive, changed)
        metadata_invariant = all(torch.equal(original[key], mutated[key]) for key in allowed_keys)
        if not metadata_invariant:
            raise ValueError("Evaluation metadata affected encoded model inputs")
    return {"status": "passed", "records_checked": len(checks), "checks": checks,
            "model_weight_loaded": False, "model_inference_run": False,
            "metadata_mutation_leaves_all_model_input_tensors_identical": metadata_invariant,
            "all_shared_image_groups_have_identical_processed_pixels": True,
            "input_keys": sorted(allowed_keys),
            "gradient_scope": "No gradient computed; PIL/NumPy conversion remains outside autograd"}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-processor", action="store_true")
    args = parser.parse_args()
    inventory = verify_archives()
    with zipfile.ZipFile(BASE / "Examples.zip") as archive:
        records, assets = build_records(archive)
    attach_exact_task_matches(records)
    counts = Counter(record["image_group_id"] for record in records)
    processor = check_processor(records) if args.check_processor else {"status": "not_requested"}
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(DEFAULT_OUTPUT / "records.jsonl", records)
    write_jsonl(DEFAULT_OUTPUT / "evaluation_side.jsonl", [
        {key: record[key] for key in ("sample_id", "image_group_id", "reference_answer", "author_tasks", "usage")}
        for record in records])
    write_jsonl(DEFAULT_OUTPUT / "image_groups.jsonl", [
        {"image_group_id": group, "sample_ids": [r["sample_id"] for r in records if r["image_group_id"] == group],
         "record_count": count, "scope": "Exact image file bytes; original scene lineage unresolved"}
        for group, count in sorted(counts.items())])
    summary = {
        "checked_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
        "revision": REVISION, "scope": "Development records and offline CPU processor boundary only",
        "status": "passed", "source_inventory_status": inventory["status"],
        "records": len(records), "image_groups": len(counts),
        "group_size_distribution": dict(sorted(Counter(counts.values()).items())),
        "images_fully_decoded": len(assets),
        "image_formats": dict(Counter(a["format"] for a in assets.values())),
        "image_modes": dict(Counter(a["mode"] for a in assets.values())),
        "assets": assets,
        "task_mapping_status": dict(Counter(r["task_mapping_status"] for r in records)),
        "author_tasks_among_unambiguous_matches": dict(Counter(r["author_tasks"][0] for r in records if len(r["author_tasks"]) == 1)),
        "application_matching_rule": "Exact image basename + whitespace-normalized question and answer; no stem-only or semantic guesses",
        "processor_check": processor,
        "outputs": {name: {"path": (DEFAULT_OUTPUT / name).relative_to(ROOT).as_posix(),
                           "sha256": hashlib.sha256((DEFAULT_OUTPUT / name).read_bytes()).hexdigest()}
                    for name in ("records.jsonl", "evaluation_side.jsonl", "image_groups.jsonl")},
        "formal_split_created": False, "domain_accuracy_evaluated": False,
        "limits": ["Demo images mix author train/test and do not establish a held-out benchmark.",
                   "Exact file bytes identify shared images, not independent original scenes.",
                   "Author application classification does not establish a common fixed label taxonomy or single-label task.",
                   "No annotation correctness, domain model competence, gradient, training or attack result is claimed."],
    }
    target = ROOT / "resources/manifests/sarlang_development_20261105.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "records", "image_groups", "task_mapping_status", "author_tasks_among_unambiguous_matches")}, ensure_ascii=True))
    print(target)


if __name__ == "__main__":
    main()
